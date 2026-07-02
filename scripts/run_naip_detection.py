"""
run_naip_detection.py
=====================
Detect poultry farm buildings in NAIP GeoTIFF imagery using YOLO11l (COCO pretrained
or fine-tuned). No segmentation / SAM — bounding-box detection only.

Output
------
A CSV containing one row per detected object with:
  - Source file and tile metadata
  - Pixel-space bounding box (global image coordinates)
  - Coordinates in the source GeoTIFF CRS
  - WGS-84 latitude / longitude
  - NAD83(2011) Georgia State Plane Easting / Northing (or whichever CRS is configured)

Usage
-----
    python scripts/run_naip_detection.py --config configs/detect_naip_config.yaml

    # Override any scalar value on the command line:
    python scripts/run_naip_detection.py --config configs/detect_naip_config.yaml \\
        --image-dir E:/Test --output-csv outputs/my_run.csv --conf-threshold 0.4
"""

from __future__ import annotations  # enables list[str] etc. on Python 3.8/3.9

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Hard dependency checks — fail immediately with a clear message
# ---------------------------------------------------------------------------
try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import xy as rasterio_xy
except ImportError:
    sys.exit("ERROR: rasterio not installed.  Run:  pip install rasterio")

try:
    from pyproj import Transformer
except ImportError:
    sys.exit("ERROR: pyproj not installed.  Run:  pip install pyproj")

try:
    import torch
    import torchvision.ops as tvops
    from ultralytics import YOLO
    from PIL import Image
except ImportError as exc:
    sys.exit(
        f"ERROR: Missing dependency: {exc}\n"
        "Run:  pip install ultralytics pillow torch torchvision"
    )

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level_str: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    level = getattr(logging, level_str.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    # Remove any handlers that might have been added by previous basicConfig calls
    root = logging.getLogger()
    root.handlers.clear()
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    
    # Suppress verbose GDAL/Rasterio ExtraSamples warnings
    logging.getLogger("rasterio").setLevel(logging.ERROR)
    
    return logging.getLogger("naip_detection")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Merge command-line overrides into the loaded config dict."""
    if args.image_dir:
        cfg.setdefault("input", {})["image_dir"] = args.image_dir
    if args.output_csv:
        cfg.setdefault("output", {})["csv_path"] = args.output_csv
    if args.model_weights:
        cfg.setdefault("model", {})["weights"] = args.model_weights
    if args.conf_threshold is not None:
        cfg.setdefault("model", {})["conf_threshold"] = args.conf_threshold
    if args.target_crs:
        cfg.setdefault("output", {})["target_crs"] = args.target_crs
    return cfg


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def discover_images(image_dir: str, extensions: List[str], recursive: bool) -> List[Path]:
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    # On Windows, glob is case-insensitive, so *.tif == *.TIF.
    # We use a set of resolved paths to deduplicate regardless of OS.
    found: set = set()
    glob_fn = root.rglob if recursive else root.glob
    for ext in extensions:
        for p in glob_fn(f"*{ext}"):
            found.add(p.resolve())
    return sorted(found)


# ---------------------------------------------------------------------------
# Sliding-window tile generation
# ---------------------------------------------------------------------------

def generate_tiles(
    img_height: int,
    img_width: int,
    window_size: int,
    overlap: float,
) -> List[Dict]:
    """
    Generate all tile dicts that cover the full image.
    Each dict has: col_off, row_off, width, height (pixel space, full-image coords).
    Border tiles are clipped to the image boundary (no padding needed for rasterio).
    """
    stride = max(int(window_size * (1.0 - overlap)), 1)
    tiles = []
    row_off = 0
    while row_off < img_height:
        col_off = 0
        while col_off < img_width:
            w = min(window_size, img_width - col_off)
            h = min(window_size, img_height - row_off)
            tiles.append({
                "col_off": col_off,
                "row_off": row_off,
                "width": w,
                "height": h,
            })
            col_off += stride
        row_off += stride
    return tiles


# ---------------------------------------------------------------------------
# Array → PIL conversion
# ---------------------------------------------------------------------------

def array_to_pil(arr: np.ndarray) -> Image.Image:
    """
    Convert a (C, H, W) rasterio array to a PIL RGB image ready for YOLO.
    Handles 1-, 2-, 3-, and 4+ band images. Normalises non-uint8 dtypes.
    """
    # -- Band selection -------------------------------------------------
    if arr.shape[0] > 3:
        arr = arr[:3]                                    # drop NIR / extra bands
    elif arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)                 # grey → RGB
    elif arr.shape[0] == 2:
        arr = np.concatenate([arr, arr[:1]], axis=0)    # RG → RGB (pad)

    # -- dtype normalisation -------------------------------------------
    if arr.dtype != np.uint8:
        a_min, a_max = arr.min(), arr.max()
        if a_max > a_min:
            arr = ((arr.astype(np.float32) - a_min) / (a_max - a_min) * 255.0).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

    # (C, H, W) → (H, W, C) for PIL
    return Image.fromarray(np.moveaxis(arr, 0, -1))


# ---------------------------------------------------------------------------
# YOLO inference over tiles
# ---------------------------------------------------------------------------

def run_yolo_on_tiles(
    model: YOLO,
    src: rasterio.DatasetReader,
    tiles: List[Dict],
    cfg_model: dict,
    device_str: str,
    logger: logging.Logger,
) -> List[Dict]:
    """
    Runs YOLO detection on every tile of the open rasterio dataset.
    Returns raw detections with pixel coords already mapped to full-image space.
    """
    conf_thr   = cfg_model.get("conf_threshold", 0.35)
    iou_thr    = cfg_model.get("iou_threshold", 0.45)
    classes    = cfg_model.get("classes", None)
    imgsz      = cfg_model.get("imgsz", 1024)
    batch_size = cfg_model.get("batch_size", 8)
    verbose    = cfg_model.get("verbose", False)

    detections: List[Dict] = []

    # -- tile iterator with optional tqdm progress bar -----------------
    tile_iter = (
        tqdm(range(0, len(tiles), batch_size), desc="  Tiles", unit="batch", leave=False)
        if HAS_TQDM else range(0, len(tiles), batch_size)
    )

    for batch_start in tile_iter:
        batch_tiles = tiles[batch_start: batch_start + batch_size]
        pil_imgs: List[Image.Image] = []
        valid_tiles: List[Dict] = []

        for tile in batch_tiles:
            window = Window(tile["col_off"], tile["row_off"], tile["width"], tile["height"])
            try:
                arr = src.read(window=window)
            except Exception as exc:
                logger.warning(f"    Skipping tile {tile}: read error — {exc}")
                continue
            pil_imgs.append(array_to_pil(arr))
            valid_tiles.append(tile)

        if not pil_imgs:
            continue

        try:
            with torch.inference_mode():
                # stream=True yields results one by one, saving significant RAM/VRAM
                results_generator = model.predict(
                    pil_imgs,
                    conf=conf_thr,
                    iou=iou_thr,
                    classes=classes,
                    imgsz=imgsz,
                    device=device_str,
                    verbose=verbose,
                    stream=True,
                )
                
                for result, tile in zip(results_generator, valid_tiles):
                    if result.boxes is None or len(result.boxes) == 0:
                        continue

                    # result.orig_shape is (H, W) of the image passed to YOLO.
                    orig_h, orig_w = result.orig_shape
                    tile_w, tile_h = tile["width"], tile["height"]

                    # Scale factors from YOLO's result space → actual tile pixel space
                    scale_x = tile_w / orig_w
                    scale_y = tile_h / orig_h

                    # Convert entire batch of boxes to numpy at once (faster, avoids tensor leaks)
                    xyxys = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    clss  = result.boxes.cls.cpu().numpy()

                    for i in range(len(xyxys)):
                        xyxy = xyxys[i]
                        conf = float(confs[i])
                        cls  = int(clss[i])

                        # Map to full-image pixel coordinates
                        x1 = float(xyxy[0]) * scale_x + tile["col_off"]
                        y1 = float(xyxy[1]) * scale_y + tile["row_off"]
                        x2 = float(xyxy[2]) * scale_x + tile["col_off"]
                        y2 = float(xyxy[3]) * scale_y + tile["row_off"]

                        detections.append({
                            "x1_px": x1,
                            "y1_px": y1,
                            "x2_px": x2,
                            "y2_px": y2,
                            "cx_px": (x1 + x2) / 2.0,
                            "cy_px": (y1 + y2) / 2.0,
                            "width_px":  x2 - x1,
                            "height_px": y2 - y1,
                            "confidence": conf,
                            "class_id": cls,
                            "tile_col_off": tile["col_off"],
                            "tile_row_off": tile["row_off"],
                        })

        except Exception as exc:
            logger.warning(f"    YOLO inference failed on batch starting at tile {batch_start}: {exc}")
            continue
        finally:
            # Prevent VRAM fragmentation/leaks on massive images with many tiles
            if "cuda" in device_str:
                torch.cuda.empty_cache()

    return detections


# ---------------------------------------------------------------------------
# Cross-tile NMS
# ---------------------------------------------------------------------------

def apply_global_nms(detections: List[Dict], iou_threshold: float) -> List[Dict]:
    """Remove duplicate detections that span overlapping tile boundaries."""
    if len(detections) <= 1:
        return detections

    boxes = torch.tensor(
        [[d["x1_px"], d["y1_px"], d["x2_px"], d["y2_px"]] for d in detections],
        dtype=torch.float32,
    )
    scores = torch.tensor([d["confidence"] for d in detections], dtype=torch.float32)
    keep = tvops.nms(boxes, scores, iou_threshold)
    return [detections[i] for i in keep.tolist()]


# ---------------------------------------------------------------------------
# Coordinate enrichment
# ---------------------------------------------------------------------------

def pixel_to_crs(affine, row: float, col: float) -> Tuple[float, float]:
    """
    Convert pixel (col, row) in the full image to (x, y) in the dataset's native CRS.
    rasterio_xy signature: xy(transform, rows, cols) -> (xs, ys)
    Note: rows=y-axis, cols=x-axis.
    """
    xs, ys = rasterio_xy(affine, row, col)
    # rasterio_xy returns numpy scalars when given scalar inputs; ensure plain float
    return float(xs), float(ys)


def enrich_with_coordinates(
    detections: List[Dict],
    affine,
    to_target: Transformer,
    to_wgs84: Optional[Transformer],
    include_source_crs: bool,
) -> List[Dict]:
    """Attach geospatial columns (source CRS, target CRS, WGS-84) to every detection."""
    for d in detections:
        # ---- pixel → native CRS (e.g., UTM metres) -------------------
        cx_src, cy_src = pixel_to_crs(affine, d["cy_px"], d["cx_px"])
        x1_src, y1_src = pixel_to_crs(affine, d["y1_px"], d["x1_px"])
        x2_src, y2_src = pixel_to_crs(affine, d["y2_px"], d["x2_px"])

        if include_source_crs:
            d["cx_src"] = round(cx_src, 4)
            d["cy_src"] = round(cy_src, 4)
            d["x1_src"] = round(x1_src, 4)
            d["y1_src"] = round(y1_src, 4)
            d["x2_src"] = round(x2_src, 4)
            d["y2_src"] = round(y2_src, 4)

        # ---- native CRS → target CRS (e.g., GA State Plane) ----------
        cx_tgt, cy_tgt = to_target.transform(cx_src, cy_src)
        x1_tgt, y1_tgt = to_target.transform(x1_src, y1_src)
        x2_tgt, y2_tgt = to_target.transform(x2_src, y2_src)

        d["cx_easting"]   = round(cx_tgt, 2)
        d["cy_northing"]  = round(cy_tgt, 2)
        d["x1_easting"]   = round(x1_tgt, 2)
        d["y1_northing"]  = round(y1_tgt, 2)
        d["x2_easting"]   = round(x2_tgt, 2)
        d["y2_northing"]  = round(y2_tgt, 2)
        d["bbox_width_ft"]  = round(abs(x2_tgt - x1_tgt), 2)
        d["bbox_height_ft"] = round(abs(y2_tgt - y1_tgt), 2)
        d["bbox_area_sqft"] = round(d["bbox_width_ft"] * d["bbox_height_ft"], 2)

        # ---- native CRS → WGS-84 lat/lon ----------------------------
        if to_wgs84 is not None:
            lon, lat = to_wgs84.transform(cx_src, cy_src)
            d["longitude_wgs84"] = round(lon, 7)
            d["latitude_wgs84"]  = round(lat, 7)

    return detections


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

CSV_FIELDNAMES = [
    # -- File info
    "image_file", "image_crs",
    # -- Pixel space (global image coordinates)
    "cx_px", "cy_px",
    "x1_px", "y1_px", "x2_px", "y2_px",
    "width_px", "height_px",
    # -- Source / native CRS  (e.g. UTM metres)
    "cx_src", "cy_src",
    "x1_src", "y1_src", "x2_src", "y2_src",
    # -- Target CRS  (e.g. GA State Plane, feet)
    "cx_easting", "cy_northing",
    "x1_easting", "y1_northing", "x2_easting", "y2_northing",
    "bbox_width_ft", "bbox_height_ft", "bbox_area_sqft",
    # -- WGS-84
    "latitude_wgs84", "longitude_wgs84",
    # -- Detection metadata
    "confidence", "class_id", "class_name",
    "tile_col_off", "tile_row_off",
]


def open_csv_writer(
    csv_path: str,
    target_crs_label: str,
    model_weights: str,
) -> Tuple[object, csv.DictWriter]:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    # Open in write mode; comment line is a proper CSV comment (starts with #).
    # We keep it because pandas read_csv(..., comment='#') skips it cleanly.
    f = open(csv_path, "w", newline="", encoding="utf-8")
    f.write(f"# NAIP Detection | model={model_weights} | target_crs={target_crs_label}\n")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    return f, writer


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(
    image_path: Path,
    model: YOLO,
    device_str: str,
    cfg: dict,
    csv_writer: csv.DictWriter,
    csv_file,                    # file handle for mid-image flush
    logger: logging.Logger,
) -> int:
    """Process one GeoTIFF. Returns number of detections written to CSV."""
    cfg_model  = cfg.get("model", {})
    cfg_tiling = cfg.get("tiling", {})
    cfg_nms    = cfg.get("nms", {})
    cfg_output = cfg.get("output", {})

    window_size        = cfg_tiling.get("window_size", 1024)
    overlap            = float(cfg_tiling.get("overlap", 0.20))
    nms_enabled        = cfg_nms.get("enabled", True)
    nms_iou            = float(cfg_nms.get("iou_threshold", 0.40))
    target_crs         = cfg_output.get("target_crs", "EPSG:6444")
    include_wgs84      = cfg_output.get("include_wgs84", True)
    include_source_crs = cfg_output.get("include_source_crs", True)

    logger.info(f"  Opening: {image_path.name}")

    try:
        with rasterio.open(image_path) as src:
            img_h, img_w = src.height, src.width
            src_crs_str  = src.crs.to_string() if src.crs else "UNKNOWN"
            affine       = src.transform

            logger.info(f"  Size: {img_w}×{img_h}px | CRS: {src_crs_str} | Bands: {src.count}")

            # -- Build CRS transformers --------------------------------
            try:
                to_target = Transformer.from_crs(src_crs_str, target_crs, always_xy=True)
            except Exception as exc:
                logger.error(f"  Cannot build {src_crs_str} → {target_crs} transformer: {exc}")
                return 0

            to_wgs84: Optional[Transformer] = None
            if include_wgs84:
                try:
                    to_wgs84 = Transformer.from_crs(src_crs_str, "EPSG:4326", always_xy=True)
                except Exception as exc:
                    logger.warning(f"  WGS-84 transformer failed: {exc}  (WGS-84 columns will be empty)")

            # -- Generate tiles ----------------------------------------
            tiles = generate_tiles(img_h, img_w, window_size, overlap)
            logger.info(
                f"  Tiles: {len(tiles)} "
                f"(window={window_size}px, stride={int(window_size*(1-overlap))}px, overlap={overlap:.0%})"
            )

            # -- YOLO inference ----------------------------------------
            t0 = time.time()
            raw_dets = run_yolo_on_tiles(model, src, tiles, cfg_model, device_str, logger)
            elapsed = time.time() - t0
            logger.info(f"  Raw detections: {len(raw_dets)} in {elapsed:.1f}s")

            # -- Cross-tile NMS ----------------------------------------
            if nms_enabled and len(raw_dets) > 1:
                raw_dets = apply_global_nms(raw_dets, nms_iou)
                logger.info(f"  After NMS: {len(raw_dets)} detections kept")

            # -- Coordinate enrichment --------------------------------
            enriched = enrich_with_coordinates(
                raw_dets, affine, to_target, to_wgs84, include_source_crs
            )

            # -- Write to CSV -----------------------------------------
            class_names = model.names or {}
            for det in enriched:
                det["image_file"] = image_path.name
                det["image_crs"]  = src_crs_str
                det["class_name"] = class_names.get(det["class_id"], f"cls_{det['class_id']}")
                csv_writer.writerow(det)

            # Flush after each image so partial results survive a crash
            csv_file.flush()
            return len(enriched)

    except rasterio.errors.RasterioIOError as exc:
        logger.error(f"  Cannot open {image_path.name}: {exc}")
        return 0


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_detection(cfg: dict, logger: logging.Logger) -> None:
    cfg_model  = cfg.get("model", {})
    cfg_input  = cfg.get("input", {})
    cfg_output = cfg.get("output", {})

    # -- Discover images -----------------------------------------------
    image_dir  = cfg_input.get("image_dir", "E:/Test")
    extensions = cfg_input.get("extensions", [".tif", ".tiff"])
    recursive  = cfg_input.get("recursive", True)

    logger.info(f"Scanning for images in: {image_dir}")
    image_paths = discover_images(image_dir, extensions, recursive)
    if not image_paths:
        logger.error(f"No images found in '{image_dir}' with extensions {extensions}")
        sys.exit(1)
    logger.info(f"Found {len(image_paths)} image(s)\n")

    # -- Load YOLO model once ------------------------------------------
    weights    = cfg_model.get("weights", "yolo11l.pt")
    device_cfg = cfg_model.get("device", "auto")
    device_str = ("cuda" if torch.cuda.is_available() else "cpu") if device_cfg == "auto" else device_cfg

    logger.info(f"Loading model: {weights}  (device={device_str})")
    model = YOLO(weights)
    # Warm up the model on device so the first real tile isn't slow
    _ = model.predict(
        [Image.new("RGB", (64, 64))],
        device=device_str,
        verbose=False,
        imgsz=64,
    )
    logger.info(f"Model ready. {len(model.names)} classes.")

    # -- Open CSV ------------------------------------------------------
    csv_path         = cfg_output.get("csv_path", "outputs/naip_detections.csv")
    target_crs_label = cfg_output.get("target_crs_label", cfg_output.get("target_crs", "EPSG:6444"))
    csv_file, csv_writer = open_csv_writer(csv_path, target_crs_label, weights)

    # -- Process every image -------------------------------------------
    total_dets = 0
    t_total    = time.time()

    try:
        for idx, image_path in enumerate(image_paths, start=1):
            logger.info(f"\n[{idx}/{len(image_paths)}] {image_path.name}")
            n = process_image(
                image_path, model, device_str, cfg, csv_writer, csv_file, logger
            )
            total_dets += n
            logger.info(f"  → Wrote {n} rows  (running total: {total_dets})")
    finally:
        csv_file.flush()
        csv_file.close()

    elapsed_total = time.time() - t_total
    sep = "=" * 60
    logger.info(
        f"\n{sep}\n"
        f"Finished! Processed {len(image_paths)} image(s) in {elapsed_total/60:.1f} min\n"
        f"Total detections : {total_dets}\n"
        f"Output CSV       : {Path(csv_path).resolve()}\n"
        f"Load in pandas   : pd.read_csv('{csv_path}', comment='#')\n"
        f"{sep}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run YOLO11l (COCO-pretrained or fine-tuned) on NAIP GeoTIFF imagery.\n"
            "Outputs a georeferenced CSV with bounding boxes in pixel space,\n"
            "native CRS, NAD83(2011) GA State Plane, and WGS-84 lat/lon."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", "-c",
        default="configs/detect_naip_config.yaml",
        help="Path to YAML config file  (default: configs/detect_naip_config.yaml)",
    )
    p.add_argument("--image-dir",      default=None, help="Override input.image_dir")
    p.add_argument("--output-csv",     default=None, help="Override output.csv_path")
    p.add_argument("--model-weights",  default=None, help="Override model.weights")
    p.add_argument("--conf-threshold", type=float, default=None, help="Override model.conf_threshold")
    p.add_argument("--target-crs",     default=None, help="Override output.target_crs  e.g. EPSG:6445")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    log_cfg = cfg.get("logging", {})
    logger  = setup_logging(
        level_str=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file", None),
    )

    sep = "=" * 60
    logger.info(sep)
    logger.info("NAIP Poultry Farm Detection — YOLO (bbox only, no SAM)")
    logger.info(sep)
    logger.info(f"Config      : {args.config}")
    logger.info(f"Image dir   : {cfg['input']['image_dir']}")
    logger.info(f"Output CSV  : {cfg['output']['csv_path']}")
    logger.info(f"Target CRS  : {cfg['output'].get('target_crs', 'EPSG:6444')}")
    logger.info(f"Model       : {cfg['model'].get('weights', 'yolo11l.pt')}")
    logger.info(f"Confidence  : {cfg['model'].get('conf_threshold', 0.35)}")

    run_detection(cfg, logger)


if __name__ == "__main__":
    main()
