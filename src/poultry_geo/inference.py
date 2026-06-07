from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon

from poultry_geo.geotiff import find_geotiffs, slice_geotiff_to_memory


def _ensure_three_channels(image_array: np.ndarray) -> np.ndarray:
    if image_array.ndim != 3 or image_array.shape[0] == 0:
        raise ValueError("Expected image data in (bands, height, width) format.")
    if image_array.shape[0] >= 3:
        return image_array[:3]
    if image_array.shape[0] == 1:
        return np.repeat(image_array, 3, axis=0)
    return np.concatenate([image_array, image_array[-1:, :, :]], axis=0)


def _normalize_tile_to_pil(image_array: np.ndarray) -> Image.Image:
    rgb = np.moveaxis(_ensure_three_channels(image_array), 0, -1)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32)
        rgb_min = float(rgb.min())
        rgb_max = float(rgb.max())
        if rgb_max == rgb_min:
            rgb = np.zeros_like(rgb, dtype=np.uint8)
        else:
            rgb = np.clip((rgb - rgb_min) / (rgb_max - rgb_min) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


def _iter_same_shape_batches(tiles: list[Any], batch_size: int):
    current_batch: list[Any] = []
    current_shape: tuple[int, int] | None = None

    for tile in tiles:
        tile_shape = tile.image_array.shape[1:]
        if current_shape is None:
            current_shape = tile_shape

        if tile_shape != current_shape or len(current_batch) >= batch_size:
            yield current_batch
            current_batch = [tile]
            current_shape = tile_shape
            continue

        current_batch.append(tile)

    if current_batch:
        yield current_batch


def _project_mask_points(mask_pixel_coords: np.ndarray, transform: Any) -> list[list[float]]:
    coords: list[list[float]] = []
    for px, py in mask_pixel_coords:
        x_coord, y_coord = transform * (float(px), float(py))
        coords.append([float(x_coord), float(y_coord)])
    return coords


def yolo_result_to_records(
    result: Any,
    transform: Any,
    tile_name: str,
    offset_xy: tuple[int, int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if result.masks is None or result.boxes is None or result.boxes.data is None:
        return records

    for mask_index, (mask, box, cls, conf) in enumerate(
        zip(result.masks.xy, result.boxes.xyxy, result.boxes.cls, result.boxes.conf)
    ):
        mask_pixel_coords = np.asarray(mask, dtype=np.float32)
        if len(mask_pixel_coords) < 3:
            continue

        pixel_polygon = Polygon(mask_pixel_coords)
        if pixel_polygon.is_empty or not pixel_polygon.is_valid:
            continue

        x1, y1, x2, y2 = map(float, box.tolist())
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        longitude, latitude = transform * (x_center, y_center)
        bbox_lon_x1, bbox_lat_y1 = transform * (x1, y1)
        bbox_lon_x2, bbox_lat_y2 = transform * (x2, y2)
        mask_points_gis = _project_mask_points(mask_pixel_coords, transform)
        global_x = offset_xy[0] + x_center
        global_y = offset_xy[1] + y_center
        pixel_area = pixel_polygon.area
        area_geo = pixel_area * abs(float(transform.a) * float(transform.e))

        records.append(
            {
                "latitude": float(latitude),
                "longitude": float(longitude),
                "bbox_lat_y1": float(bbox_lat_y1),
                "bbox_lon_x1": float(bbox_lon_x1),
                "bbox_lat_y2": float(bbox_lat_y2),
                "bbox_lon_x2": float(bbox_lon_x2),
                "bbox_width_geo": abs(float(bbox_lon_x2) - float(bbox_lon_x1)),
                "bbox_height_geo": abs(float(bbox_lat_y2) - float(bbox_lat_y1)),
                "global_x": float(global_x),
                "global_y": float(global_y),
                "x_center": float(x_center),
                "y_center": float(y_center),
                "bbox_x1": float(x1),
                "bbox_y1": float(y1),
                "bbox_x2": float(x2),
                "bbox_y2": float(y2),
                "bbox_width": float(x2 - x1),
                "bbox_height": float(y2 - y1),
                "area_geo": float(area_geo),
                "pixel_count": float(pixel_area),
                "class": int(cls.item()),
                "confidence": float(conf.item()),
                "tile_name": tile_name,
                "mask_index": mask_index,
                "mask_points": json.dumps(mask_pixel_coords.tolist()),
                "mask_points_gis": json.dumps(mask_points_gis),
            }
        )

    return records


def _apply_confidence_threshold(result: Any, threshold: float) -> Any:
    if threshold <= 0.0 or result.boxes is None or result.boxes.data is None:
        return result

    keep = result.boxes.conf >= threshold
    result.boxes.data = result.boxes.data[keep]
    if result.masks is not None and result.masks.data is not None:
        result.masks.data = result.masks.data[keep]
    return result


def run_yolo_inference(
    geotiff_dir: str | Path,
    model_path: str | Path,
    output_csv: str | Path,
    window_size: int = 3000,
    stride: int = 3000,
    batch_size: int = 8,
    confidence_threshold: float = 0.0,
    device: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Inference requires torch. Install requirements-ml.txt.") from exc

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Inference requires ultralytics. Install requirements-ml.txt.") from exc

    geotiff_dir = Path(geotiff_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(model_path)).to(device)

    all_records: list[dict[str, Any]] = []
    geotiff_paths = find_geotiffs(geotiff_dir)
    for index, geotiff_path in enumerate(geotiff_paths, start=1):
        tiles = slice_geotiff_to_memory(geotiff_path, tile_size=window_size, stride=stride)
        for batch_tiles in _iter_same_shape_batches(tiles, batch_size):
            batch_images = [_normalize_tile_to_pil(tile.image_array) for tile in batch_tiles]
            results = model(batch_images, device=device, batch=len(batch_tiles), verbose=False)
            for tile, result in zip(batch_tiles, results):
                result = _apply_confidence_threshold(result, confidence_threshold)
                all_records.extend(
                    yolo_result_to_records(
                        result=result,
                        transform=tile.transform,
                        tile_name=geotiff_path.name,
                        offset_xy=tile.offset_xy,
                    )
                )
        if progress_callback is not None:
            progress_callback(index, len(geotiff_paths), geotiff_path.name)

    pd.DataFrame(all_records).to_csv(output_csv, index=False)
    return output_csv
