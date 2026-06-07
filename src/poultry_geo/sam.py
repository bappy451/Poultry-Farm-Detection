from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image


def lonlat_to_pixel(lon: float, lat: float, transform: object) -> tuple[int, int]:
    row, col = rasterio.transform.rowcol(transform, lon, lat)
    return int(col), int(row)


def proj_bbox_to_pixel(
    bbox_proj: tuple[float, float, float, float] | list[float],
    transform: object,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_proj
    row1, col1 = rasterio.transform.rowcol(transform, x1, y1)
    row2, col2 = rasterio.transform.rowcol(transform, x2, y2)
    px1, px2 = sorted([int(col1), int(col2)])
    py1, py2 = sorted([int(row1), int(row2)])
    return px1, py1, px2, py2


def build_sam_predictor(
    checkpoint_path: str | Path,
    model_type: str = "vit_h",
    device: str | None = None,
):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("SAM refinement requires torch. Install requirements-ml.txt.") from exc

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "SAM refinement requires segment_anything. Install it before running sam-refine."
        ) from exc

    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path)).to(device=resolved_device)
    return SamPredictor(sam)


def build_sam2_mask_generator(
    checkpoint_path: str | Path,
    model_config: str,
    device: str | None = None,
):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("SAM2 requires torch. Install requirements-ml.txt.") from exc

    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
    except ImportError as exc:
        raise RuntimeError("SAM2 support requires the sam2 package to be installed.") from exc

    resolved_device = device
    if resolved_device is None:
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"

    sam2_model = build_sam2(
        model_config,
        str(checkpoint_path),
        device=resolved_device,
        apply_postprocessing=False,
    )
    return SAM2AutomaticMaskGenerator(sam2_model)


def run_sam_geo_csv_batch(
    csv_path: str | Path,
    geotiff_dir: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    model_type: str = "vit_h",
    tile_col: str = "tile_name",
    point_lon_col: str = "longitude",
    point_lat_col: str = "latitude",
    box_y1_col: str = "bbox_lat_y1",
    box_x1_col: str = "bbox_lon_x1",
    box_y2_col: str = "bbox_lat_y2",
    box_x2_col: str = "bbox_lon_x2",
    class_label_col: str = "class",
    conf_col: str = "confidence",
) -> list[Path]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("SAM refinement requires opencv-python. Install requirements-ml.txt.") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = build_sam_predictor(checkpoint_path=checkpoint_path, model_type=model_type)
    df = pd.read_csv(csv_path)
    df[tile_col] = df[tile_col].astype(str)

    geotiff_dir = Path(geotiff_dir)
    tiff_files = {
        path.name: path
        for path in geotiff_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    }

    written_paths: list[Path] = []
    for tile_name, geotiff_path in sorted(tiff_files.items()):
        if tile_name not in df[tile_col].values:
            continue

        objects = df[df[tile_col] == tile_name].to_dict(orient="records")
        if not objects:
            continue

        with rasterio.open(geotiff_path) as src:
            image = src.read([1, 2, 3])
            image = np.transpose(image, (1, 2, 0))
            transform = src.transform

        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            image_min = float(image.min())
            image_max = float(image.max())
            if image_max == image_min:
                image = np.zeros_like(image, dtype=np.uint8)
            else:
                image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)

        overlay = image.copy()

        for index, obj in enumerate(objects):
            try:
                lon = float(obj[point_lon_col])
                lat = float(obj[point_lat_col])
                point_x, point_y = lonlat_to_pixel(lon, lat, transform)

                box_x1 = float(obj[box_x1_col])
                box_y1 = float(obj[box_y1_col])
                box_x2 = float(obj[box_x2_col])
                box_y2 = float(obj[box_y2_col])
                x1, y1 = lonlat_to_pixel(box_x1, box_y1, transform)
                x2, y2 = lonlat_to_pixel(box_x2, box_y2, transform)
                x1, x2 = np.clip([x1, x2], 0, overlay.shape[1] - 1)
                y1, y2 = np.clip([y1, y2], 0, overlay.shape[0] - 1)
                x1, x2 = sorted([int(x1), int(x2)])
                y1, y2 = sorted([int(y1), int(y2)])

                if x2 <= x1 or y2 <= y1:
                    continue

                roi_rgb = overlay[y1:y2, x1:x2]
                if roi_rgb.shape[0] < 2 or roi_rgb.shape[1] < 2:
                    continue

                predictor.set_image(roi_rgb)
                point_coords = np.array([[point_x - x1, point_y - y1]])
                point_labels = np.array([1])
                box_coords = np.array([0, 0, x2 - x1, y2 - y1])
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_coords,
                    multimask_output=True,
                )
                if masks is None or len(masks) == 0:
                    continue

                mask = masks[int(np.argmax(scores))]
                mask_full = np.zeros(overlay.shape[:2], dtype=np.uint8)
                resized = cv2.resize(
                    mask.astype(np.uint8),
                    (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask_full[y1:y2, x1:x2] = resized

                mask_img = np.zeros_like(overlay)
                mask_img[mask_full > 0] = [0, 0, 255]
                overlay = cv2.addWeighted(overlay, 1.0, mask_img, 0.4, 0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(overlay, (int(point_x), int(point_y)), 4, (0, 255, 255), -1)
                label = f"Cls:{obj.get(class_label_col, '')} ({float(obj.get(conf_col, 0.0)):.2f})"
                origin = (x1, max(y1 - 5, 20))
                cv2.putText(overlay, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            except Exception as exc:
                print(f"[sam-refine] Skipped {tile_name} row {index}: {exc}")

        output_path = output_dir / f"{geotiff_path.stem}_sam_annotated.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        written_paths.append(output_path)

    return written_paths


def run_sam_on_bbox_csv(
    csv_path: str | Path,
    geotiff_dir: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    model_type: str = "vit_h",
    class_label_col: str = "class",
    conf_col: str = "confidence",
    tile_col: str = "tile_name",
    xmin_col: str = "bbox_xmin",
    ymin_col: str = "bbox_ymin",
    xmax_col: str = "bbox_xmax",
    ymax_col: str = "bbox_ymax",
) -> list[Path]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("SAM refinement requires opencv-python. Install requirements-ml.txt.") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = build_sam_predictor(checkpoint_path=checkpoint_path, model_type=model_type)
    df = pd.read_csv(csv_path)
    geotiff_dir = Path(geotiff_dir)
    tiff_files = {
        path.stem: path
        for path in geotiff_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    }

    written_paths: list[Path] = []
    for tile_base, geotiff_path in sorted(tiff_files.items()):
        objects = df[df[tile_col].astype(str) == f"{tile_base}.tif"].to_dict(orient="records")
        if not objects:
            continue

        with rasterio.open(geotiff_path) as src:
            image = src.read([1, 2, 3])
            image = np.transpose(image, (1, 2, 0))
            transform = src.transform

        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            image_min = float(image.min())
            image_max = float(image.max())
            if image_max == image_min:
                image = np.zeros_like(image, dtype=np.uint8)
            else:
                image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)

        overlay = image.copy()
        for obj in objects:
            try:
                bbox_proj = (
                    float(obj[xmin_col]),
                    float(obj[ymin_col]),
                    float(obj[xmax_col]),
                    float(obj[ymax_col]),
                )
                x1, y1, x2, y2 = proj_bbox_to_pixel(bbox_proj, transform)
                x1, x2 = np.clip([x1, x2], 0, overlay.shape[1] - 1)
                y1, y2 = np.clip([y1, y2], 0, overlay.shape[0] - 1)
                if x2 <= x1 or y2 <= y1:
                    continue

                roi_rgb = overlay[y1:y2, x1:x2]
                if roi_rgb.size == 0 or roi_rgb.shape[0] < 2 or roi_rgb.shape[1] < 2:
                    continue

                predictor.set_image(roi_rgb)
                box = np.array([0, 0, x2 - x1, y2 - y1])
                masks, scores, _ = predictor.predict(box=box, multimask_output=True)
                if masks is None or len(masks) == 0:
                    continue

                mask = masks[int(np.argmax(scores))]
                mask_full = np.zeros(overlay.shape[:2], dtype=np.uint8)
                resized = cv2.resize(
                    mask.astype(np.uint8),
                    (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask_full[y1:y2, x1:x2] = resized
                mask_img = np.zeros_like(overlay)
                mask_img[mask_full > 0] = [0, 0, 255]
                overlay = cv2.addWeighted(overlay, 1.0, mask_img, 0.4, 0)

                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"Cls:{obj.get(class_label_col, '')} ({float(obj.get(conf_col, 0.0)):.2f})"
                origin = (int(x1), max(int(y1) - 5, 20))
                cv2.putText(overlay, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except Exception as exc:
                print(f"[sam-bbox-refine] Skipped {tile_base}: {exc}")

        output_path = output_dir / f"{tile_base}_sam_annotated.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        written_paths.append(output_path)

    return written_paths


def batch_sam2_visualize(
    image_dir: str | Path,
    checkpoint_path: str | Path,
    model_config: str,
    *,
    output_dir: str | Path | None = None,
    save_images: bool = False,
    device: str | None = None,
) -> list[Path]:
    try:
        import cv2
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "SAM2 visualization requires matplotlib and opencv-python. Install the research extras.",
        ) from exc

    image_dir = Path(image_dir)
    output_root = Path(output_dir) if output_dir else None
    if save_images and output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    generator = build_sam2_mask_generator(
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        device=device,
    )
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    written: list[Path] = []

    def show_annotations(annotations: list[dict[str, object]]) -> None:
        if not annotations:
            return
        ax = plt.gca()
        ax.set_autoscale_on(False)
        sorted_annotations = sorted(annotations, key=lambda item: float(item["area"]), reverse=True)
        first_mask = sorted_annotations[0]["segmentation"]
        overlay = np.ones((first_mask.shape[0], first_mask.shape[1], 4), dtype=np.float32)
        overlay[:, :, 3] = 0
        for annotation in sorted_annotations:
            mask = annotation["segmentation"]
            color = np.concatenate([np.random.random(3), [0.5]])
            overlay[mask] = color
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(overlay, contours, -1, (0, 0, 1, 0.4), thickness=1)
        ax.imshow(overlay)

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        masks = generator.generate(image_np)

        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(image_np)
        plt.title("Original")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(image_np)
        show_annotations(masks)
        plt.title("Segmented")
        plt.axis("off")
        plt.suptitle(image_path.name)

        if save_images and output_root is not None:
            output_path = output_root / f"{image_path.stem}_sam2.png"
            plt.savefig(output_path, bbox_inches="tight")
            written.append(output_path)
        plt.close()

    return written
