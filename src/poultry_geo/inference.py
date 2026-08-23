from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon

from poultry_geo.geotiff import find_geotiffs, iter_slice_geotiff


def _ensure_three_channels(image_array: np.ndarray) -> np.ndarray:
    if image_array.ndim != 3 or image_array.shape[0] == 0:
        raise ValueError("Expected image data in (bands, height, width) format.")
    if image_array.shape[0] >= 3:
        return image_array[:3]
    if image_array.shape[0] == 1:
        return np.repeat(image_array, 3, axis=0)
    return np.concatenate([image_array, image_array[-1:, :, :]], axis=0)


def _normalize_tile_to_pil(image_array: np.ndarray, value_range: tuple[float, float] | None = None) -> Image.Image:
    """Convert imagery without per-tile contrast stretching."""
    rgb = np.moveaxis(_ensure_three_channels(image_array), 0, -1)
    if rgb.dtype == np.uint8:
        return Image.fromarray(np.ascontiguousarray(rgb))
    if value_range is None:
        raise ValueError("Non-uint8 GeoTIFF input requires a fixed normalization_range=(min, max).")
    low, high = map(float, value_range)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("normalization_range must be finite with max > min.")
    return Image.fromarray(np.ascontiguousarray(np.clip((rgb.astype(np.float32) - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)))


def _iter_same_shape_batches(tiles: Iterable[Any], batch_size: int):
    batch: list[Any] = []
    shape: tuple[int, int] | None = None
    for tile in tiles:
        tile_shape = tile.image_array.shape[1:]
        if batch and (tile_shape != shape or len(batch) >= batch_size):
            yield batch
            batch, shape = [], None
        batch.append(tile)
        shape = tile_shape
    if batch:
        yield batch


def _project_points(points: np.ndarray, transform: Any) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in (transform * (float(px), float(py)) for px, py in points)]


def _to_wgs84(x: float, y: float, source_crs: Any) -> tuple[float | None, float | None]:
    if source_crs is None:
        return None, None
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True).transform(x, y)
    return float(lon), float(lat)


def _record(points: np.ndarray, box: np.ndarray, cls: Any, conf: Any, transform: Any, source_crs: Any,
            tile_name: str, offset_xy: tuple[int, int], task: str, index: int, source_id: str | None = None) -> dict[str, Any]:
    polygon = Polygon(points)
    if polygon.is_empty or not polygon.is_valid:
        raise ValueError("Prediction geometry is invalid.")
    x1, y1, x2, y2 = map(float, box.tolist())
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    native_x, native_y = transform * (center_x, center_y)
    longitude, latitude = _to_wgs84(float(native_x), float(native_y), source_crs)
    projected_points = _project_points(points, transform)
    return {
        "task": task, "prediction_index": index, "tile_name": tile_name, "source_id": source_id or tile_name,
        "class": int(cls.item() if hasattr(cls, "item") else cls),
        "confidence": float(conf.item() if hasattr(conf, "item") else conf),
        "global_x": float(offset_xy[0] + center_x), "global_y": float(offset_xy[1] + center_y),
        "x_center": center_x, "y_center": center_y, "bbox_x1": x1, "bbox_y1": y1,
        "bbox_x2": x2, "bbox_y2": y2, "bbox_width": x2 - x1, "bbox_height": y2 - y1,
        "native_x": float(native_x), "native_y": float(native_y), "source_crs": str(source_crs) if source_crs else None,
        "longitude_wgs84": longitude, "latitude_wgs84": latitude,
        "pixel_area": float(polygon.area), "geometry_pixel": json.dumps(points.tolist()),
        "geometry_native": json.dumps(projected_points),
        "mask_points": json.dumps(points.tolist()), "mask_points_gis": json.dumps(projected_points),
    }


def yolo_result_to_records(result: Any, transform: Any, tile_name: str, offset_xy: tuple[int, int],
                           source_crs: Any = None, source_id: str | None = None,
                           skipped: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """Convert detection, segmentation, or OBB Ultralytics output into GIS records."""
    records: list[dict[str, Any]] = []

    def append_record(points: Any, box: Any, cls: Any, conf: Any, task: str, index: int) -> None:
        try:
            records.append(_record(
                np.asarray(points.cpu() if hasattr(points, "cpu") else points, dtype=np.float32),
                np.asarray(box.cpu() if hasattr(box, "cpu") else box, dtype=np.float32),
                cls, conf, transform, source_crs, tile_name, offset_xy, task, index, source_id,
            ))
        except ValueError:
            if skipped is not None:
                skipped["invalid_geometry"] = skipped.get("invalid_geometry", 0) + 1

    if getattr(result, "obb", None) is not None and result.obb.data is not None:
        for i, (points, box, cls, conf) in enumerate(zip(result.obb.xyxyxyxy, result.obb.xyxy, result.obb.cls, result.obb.conf)):
            append_record(points, box, cls, conf, "obb", i)
        return records
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.data is None:
        return records
    mask_polygons = getattr(getattr(result, "masks", None), "xy", None)
    for i, (box, cls, conf) in enumerate(zip(boxes.xyxy, boxes.cls, boxes.conf)):
        box_array = np.asarray(box.cpu() if hasattr(box, "cpu") else box, dtype=np.float32)
        if mask_polygons is not None and i < len(mask_polygons):
            points, task = mask_polygons[i], "segment"
        else:
            x1, y1, x2, y2 = box_array.tolist()
            points, task = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "detect"
        append_record(points, box_array, cls, conf, task, i)
    return records

def _dedupe_records(records: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    """Class-aware polygon NMS isolated to one source raster and its native CRS."""
    kept: list[dict[str, Any]] = []
    groups: dict[tuple[str | None, str | None, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (record.get("source_id", record.get("tile_name")), record.get("source_crs"), int(record["class"]))
        groups.setdefault(key, []).append(record)
    for candidates in groups.values():
        accepted: list[Polygon] = []
        for record in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
            polygon = Polygon(json.loads(record["geometry_native"]))
            if polygon.is_empty or not polygon.is_valid:
                continue
            if any(polygon.intersection(previous).area / max(polygon.union(previous).area, 1e-12) > iou_threshold for previous in accepted):
                continue
            accepted.append(polygon)
            kept.append(record)
    return kept

def run_yolo_inference(geotiff_dir: str | Path, model_path: str | Path, output_csv: str | Path,
                       window_size: int = 3000, stride: int = 3000, batch_size: int = 8,
                       confidence_threshold: float = 0.0, device: str | None = None,
                       normalization_range: tuple[float, float] | None = None,
                       cross_tile_nms_iou: float | None = 0.5,
                       progress_callback: Callable[[int, int, str], None] | None = None) -> Path:
    """Run tiled YOLO inference, applying NMS only among tiles from each source raster."""
    import torch
    from ultralytics import YOLO

    geotiff_dir, output_csv = Path(geotiff_dir), Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if device in (None, "auto") else device
    model = YOLO(str(model_path)).to(resolved_device)
    geotiff_paths = find_geotiffs(geotiff_dir)
    skipped: dict[str, int] = {}
    wrote_records = False
    for index, geotiff_path in enumerate(geotiff_paths, start=1):
        source_records: list[dict[str, Any]] = []
        source_id = str(geotiff_path.resolve())
        for tiles in _iter_same_shape_batches(iter_slice_geotiff(geotiff_path, tile_size=window_size, stride=stride), batch_size):
            images = [_normalize_tile_to_pil(tile.image_array, normalization_range) for tile in tiles]
            results = model(images, device=resolved_device, batch=len(tiles), conf=confidence_threshold, verbose=False)
            for tile, result in zip(tiles, results):
                source_records.extend(yolo_result_to_records(
                    result, tile.transform, geotiff_path.name, tile.offset_xy, tile.crs,
                    source_id=source_id, skipped=skipped,
                ))
        if cross_tile_nms_iou is not None:
            source_records = _dedupe_records(source_records, cross_tile_nms_iou)
        if source_records:
            pd.DataFrame(source_records).to_csv(output_csv, index=False, mode="a" if wrote_records else "w", header=not wrote_records)
            wrote_records = True
        if progress_callback:
            progress_callback(index, len(geotiff_paths), geotiff_path.name)
    if not wrote_records:
        pd.DataFrame(columns=["task", "prediction_index", "tile_name", "source_id", "class", "confidence"]).to_csv(output_csv, index=False)
    output_csv.with_suffix(".metadata.json").write_text(json.dumps({
        "model_path": str(model_path), "device": str(resolved_device),
        "normalization_range": normalization_range, "cross_tile_nms_iou": cross_tile_nms_iou,
        "nms_scope": "same source raster and native CRS only",
        "input_band_policy": "first three bands; one/two-band inputs are replicated to RGB",
        "geotiffs": [str(path) for path in geotiff_paths], "skipped_predictions": skipped,
    }, indent=2), encoding="utf-8")
    return output_csv
