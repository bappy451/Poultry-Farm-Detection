from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
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


def _full_raster_to_bgr(full_image_array: np.ndarray) -> np.ndarray:
    image = np.moveaxis(_ensure_three_channels(full_image_array), 0, -1)
    image = image.astype(np.float32)
    image_min = float(image.min())
    image_max = float(image.max())
    if np.isclose(image_min, image_max):
        image = np.zeros_like(image, dtype=np.uint8)
    else:
        image = np.clip((image - image_min) / (image_max - image_min) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def draw_annotations_on_image(
    full_image_array: np.ndarray,
    slices: list[dict[str, Any]],
    object_infos: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    overlay = _full_raster_to_bgr(full_image_array)

    for obj in object_infos:
        offset_x = int(obj["global_x"] - obj["x_center"])
        offset_y = int(obj["global_y"] - obj["y_center"])

        x1_global = int(obj["bbox_x1"] + offset_x)
        y1_global = int(obj["bbox_y1"] + offset_y)
        x2_global = int(obj["bbox_x2"] + offset_x)
        y2_global = int(obj["bbox_y2"] + offset_y)

        confidence = float(obj["confidence"])
        class_id = int(obj["class"])
        label = f"Cls:{class_id} ({confidence:.2f})"
        cv2.rectangle(overlay, (x1_global, y1_global), (x2_global, y2_global), (0, 255, 0), 3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_origin = (x1_global, max(y1_global - 10, text_size[1]))
        cv2.putText(overlay, label, text_origin, font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(overlay, label, text_origin, font, font_scale, (255, 255, 255), thickness)

        try:
            mask_points = np.asarray(json.loads(obj["mask_points"]), dtype=np.int32)
            if len(mask_points) >= 3:
                mask_points_global = mask_points + np.array([offset_x, offset_y])
                mask_img = np.zeros_like(overlay)
                cv2.fillPoly(mask_img, [mask_points_global], (0, 0, 255))
                cv2.polylines(mask_img, [mask_points_global], isClosed=True, color=(0, 0, 255), thickness=2)
                overlay = cv2.addWeighted(overlay, 1.0, mask_img, 0.7, 0)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

        try:
            latitude = float(obj["latitude"])
            longitude = float(obj["longitude"])
            pixel_x = None
            pixel_y = None
            for slice_data in slices:
                try:
                    row, col = rowcol(slice_data["transform"], longitude, latitude)
                    pixel_x = col + slice_data["offset_xy"][0]
                    pixel_y = row + slice_data["offset_xy"][1]
                    break
                except Exception:
                    continue
            if pixel_x is not None and pixel_y is not None:
                cv2.circle(overlay, (int(pixel_x), int(pixel_y)), 8, (0, 0, 255), -1)
                latlon_label = f"({latitude:.5f}, {longitude:.5f})"
                latlon_origin = (int(pixel_x + 10), int(pixel_y - 10))
                cv2.putText(overlay, latlon_label, latlon_origin, font, 0.7, (0, 0, 0), 3)
                cv2.putText(overlay, latlon_label, latlon_origin, font, 0.7, (255, 255, 255), 2)
        except (KeyError, TypeError, ValueError):
            pass

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        output_path = output_path.with_suffix(".jpg")
    cv2.imwrite(str(output_path), overlay)
    return output_path


def draw_annotations_with_rectangles(
    full_image_array: np.ndarray,
    slices: list[dict[str, Any]],
    object_infos: list[dict[str, Any]],
    output_path: str | Path,
    geometry_column: str = "rect_mask_points",
) -> Path:
    overlay = _full_raster_to_bgr(full_image_array)
    for obj in object_infos:
        try:
            rect_points = np.asarray(json.loads(obj[geometry_column]), dtype=np.int32)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue

        if len(rect_points) < 3:
            continue
        polygon = Polygon(rect_points)
        if polygon.is_empty or not polygon.is_valid:
            continue

        offset_x = int(obj.get("global_x", 0) - obj.get("x_center", 0))
        offset_y = int(obj.get("global_y", 0) - obj.get("y_center", 0))
        rect_points_global = rect_points + np.array([offset_x, offset_y])
        mask_img = np.zeros_like(overlay)
        cv2.fillPoly(mask_img, [rect_points_global], (255, 0, 0))
        overlay = cv2.addWeighted(overlay, 1.0, mask_img, 0.4, 0)
        cv2.polylines(overlay, [rect_points_global], isClosed=True, color=(255, 255, 0), thickness=3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        output_path = output_path.with_suffix(".jpg")
    cv2.imwrite(str(output_path), overlay)
    return output_path


def visualize_detection_csv(
    geotiff_dir: str | Path,
    detections_csv: str | Path,
    output_dir: str | Path,
    *,
    window_size: int = 3000,
    stride: int = 3000,
    rectangle_geometry_column: str | None = None,
) -> list[Path]:
    geotiff_dir = Path(geotiff_dir)
    detections_df = pd.read_csv(detections_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for geotiff_path in find_geotiffs(geotiff_dir):
        tile_name = geotiff_path.name
        matched = detections_df[detections_df["tile_name"].astype(str) == tile_name]
        if matched.empty:
            continue
        with rasterio.open(geotiff_path) as src:
            full_image_array = src.read()
        tiles = [
            {
                "transform": tile.transform,
                "offset_xy": tile.offset_xy,
            }
            for tile in slice_geotiff_to_memory(geotiff_path, tile_size=window_size, stride=stride)
        ]
        output_path = output_dir / f"{geotiff_path.stem}_annotated.jpg"
        if rectangle_geometry_column:
            draw_annotations_with_rectangles(
                full_image_array,
                tiles,
                matched.to_dict(orient="records"),
                output_path,
                geometry_column=rectangle_geometry_column,
            )
        else:
            draw_annotations_on_image(
                full_image_array,
                tiles,
                matched.to_dict(orient="records"),
                output_path,
            )
        written.append(output_path)
    return written


def draw_annotations_on_jpg_image(
    jpg_image_path: str | Path,
    object_infos: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    image = cv2.imread(str(jpg_image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {jpg_image_path}")

    overlay = image.copy()
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    for obj in object_infos:
        x1 = int(round(float(obj["bbox_x1"])))
        y1 = int(round(float(obj["bbox_y1"])))
        x2 = int(round(float(obj["bbox_x2"])))
        y2 = int(round(float(obj["bbox_y2"])))
        if not (0 <= x1 < width and 0 <= y1 < height and 0 <= x2 <= width and 0 <= y2 <= height):
            continue

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"Cls:{obj.get('class', '')} ({float(obj.get('confidence', 0.0)):.2f})"
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_origin = (x1, max(y1 - 5, text_size[1]))
        cv2.putText(overlay, label, text_origin, font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(overlay, label, text_origin, font, font_scale, (255, 255, 255), thickness)

        mask_points_raw = obj.get("mask_points")
        if mask_points_raw is None or (isinstance(mask_points_raw, float) and np.isnan(mask_points_raw)):
            continue
        try:
            mask_points = np.asarray(json.loads(mask_points_raw), dtype=np.int32)
            if mask_points.ndim == 2 and mask_points.shape[0] >= 3:
                mask_img = np.zeros_like(image)
                cv2.fillPoly(mask_img, [mask_points], (0, 0, 255))
                cv2.polylines(mask_img, [mask_points], isClosed=True, color=(0, 0, 255), thickness=2)
                overlay = cv2.addWeighted(overlay, 1.0, mask_img, 0.3, 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    return output_path


def visualize_detection_csv_on_jpgs(
    detections_csv: str | Path,
    jpg_dir: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    detections_df = pd.read_csv(detections_csv)
    jpg_dir = Path(jpg_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def extract_base_name(tile_name: str) -> str:
        tile_name = str(tile_name)
        if ".tif" in tile_name:
            return tile_name.split(".tif")[0]
        return Path(tile_name).stem

    detections_df["base_name"] = detections_df["tile_name"].apply(extract_base_name)
    written: list[Path] = []
    for jpg_path in sorted(path for path in jpg_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}):
        matched = detections_df[detections_df["base_name"] == jpg_path.stem]
        if matched.empty:
            continue
        output_path = output_dir / f"{jpg_path.stem}_annotated.jpg"
        draw_annotations_on_jpg_image(
            jpg_path,
            matched.to_dict(orient="records"),
            output_path,
        )
        written.append(output_path)
    return written
