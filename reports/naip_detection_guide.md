# NAIP Detection Script — User Guide
# `scripts/run_naip_detection.py`

## Overview

This script runs **YOLO11l** (COCO-pretrained or fine-tuned) on large **NAIP GeoTIFF** aerial images to detect poultry farm buildings. It produces a georeferenced CSV with bounding-box detections projected to:

- **NAD83(2011) / Georgia State Plane East** (EPSG:6444, units: US survey feet) — the primary output
- **WGS-84 Latitude / Longitude** — for convenience and web mapping
- **Source image CRS** (UTM Zone 17N for the Georgia NAIP imagery, EPSG:26917)

No segmentation or SAM processing is performed.

---

## Quick Start

```powershell
# Basic run — uses all defaults from config file
.\.venv\Scripts\python.exe scripts\run_naip_detection.py --config configs\detect_naip_config.yaml

# Override individual parameters without editing the config
.\.venv\Scripts\python.exe scripts\run_naip_detection.py `
    --config configs\detect_naip_config.yaml `
    --image-dir E:\Test `
    --output-csv outputs\naip_detections.csv `
    --conf-threshold 0.4
```

---

## Command-Line Arguments

| Argument | Description | Default |
|---|---|---|
| `--config`, `-c` | Path to YAML config file | `configs/detect_naip_config.yaml` |
| `--image-dir` | Override `input.image_dir` | *(from config)* |
| `--output-csv` | Override `output.csv_path` | *(from config)* |
| `--model-weights` | Override `model.weights` | *(from config)* |
| `--conf-threshold` | Override confidence threshold | *(from config)* |
| `--target-crs` | Override output CRS (e.g. `EPSG:6445`) | *(from config)* |

---

## Configuration File: `configs/detect_naip_config.yaml`

All parameters are documented inline in the YAML. Key sections:

### `model`
| Key | Description | Default |
|---|---|---|
| `weights` | YOLO weights path or name. `"yolo11l.pt"` auto-downloads COCO weights. | `yolo11l.pt` |
| `device` | `"auto"` / `"cuda"` / `"cpu"` | `auto` |
| `conf_threshold` | Min detection confidence (0–1) | `0.35` |
| `iou_threshold` | Per-tile NMS IoU threshold | `0.45` |
| `classes` | COCO class IDs to keep. `null` = all | `null` |
| `imgsz` | Input image size for YOLO | `1024` |
| `batch_size` | Tiles per GPU batch | `8` |

### `tiling`
| Key | Description | Default |
|---|---|---|
| `window_size` | Tile size in pixels | `1024` |
| `overlap` | Fractional tile overlap (0.0–0.5) | `0.20` |
| `pad_border` | Pad edge tiles to cover full image | `true` |

### `nms`
| Key | Description | Default |
|---|---|---|
| `enabled` | Apply cross-tile NMS | `true` |
| `iou_threshold` | IoU threshold for deduplication | `0.40` |

### `output`
| Key | Description | Default |
|---|---|---|
| `csv_path` | Output CSV path | `outputs/naip_detections.csv` |
| `target_crs` | Primary output CRS | `EPSG:6444` |
| `include_wgs84` | Add lat/lon columns | `true` |
| `include_source_crs` | Add native-CRS coordinate columns | `true` |

---

## Output CSV Schema

Each row represents one detected bounding box:

| Column | Description |
|---|---|
| `image_file` | Source GeoTIFF filename |
| `image_crs` | Native CRS of the GeoTIFF (e.g. `EPSG:26917`) |
| **Pixel Space** | |
| `cx_px`, `cy_px` | Detection centre (pixels, global image space) |
| `x1_px`, `y1_px`, `x2_px`, `y2_px` | Bounding box corners (pixels) |
| `width_px`, `height_px` | Bounding box size (pixels) |
| **Source CRS** (UTM / native) | |
| `cx_src`, `cy_src` | Centre in source CRS (meters) |
| `x1_src`, `y1_src`, `x2_src`, `y2_src` | BBox corners in source CRS |
| **Target CRS** (Georgia State Plane) | |
| `cx_easting`, `cy_northing` | Centre in GA State Plane (feet) |
| `x1_easting`, `y1_northing` | BBox top-left in GA State Plane |
| `x2_easting`, `y2_northing` | BBox bottom-right in GA State Plane |
| `bbox_width_ft`, `bbox_height_ft` | Box dimensions in feet |
| `bbox_area_sqft` | Box area in square feet |
| **WGS-84** | |
| `latitude_wgs84` | Centroid latitude (decimal degrees) |
| `longitude_wgs84` | Centroid longitude (decimal degrees) |
| **Detection** | |
| `confidence` | YOLO confidence score (0–1) |
| `class_id` | COCO class integer ID |
| `class_name` | COCO class name (e.g. `car`, `airplane`) |
| `tile_col_off`, `tile_row_off` | Origin tile pixel offset |

---

## Coordinate Reference Systems

| CRS | EPSG | Units | Notes |
|---|---|---|---|
| Source (NAIP E:\Test) | 26917 | meters | NAD83 / UTM Zone 17N |
| Target — GA East | **6444** | US survey ft | NAD83(2011) / Georgia East *(default)* |
| Target — GA West | 6445 | US survey ft | NAD83(2011) / Georgia West |
| WGS-84 | 4326 | decimal degrees | Supplementary columns |

> [!NOTE]
> The `E:\Test` imagery uses UTM Zone 17N (EPSG:26917). If you process images spanning both UTM Zone 16 and 17 (which your dataset contains), the script handles each file independently using its own native CRS, so reprojection is always correct.

---

## Switching to a Fine-Tuned Model

Once you have a fine-tuned YOLO model that only detects poultry houses, update the config:

```yaml
model:
  weights: "runs/downstream/yolo11l_coco/weights/best.pt"
  classes: [0]    # Set to [0] if your fine-tuned model uses class 0 for poultry house
  conf_threshold: 0.5
```

---

## Performance Notes (RTX 4090, 60cm NAIP)

A single 10710×12300 pixel GeoTIFF at 1024px tiles with 20% overlap generates ~208 tiles. At batch_size=8, expect roughly **~30–60 seconds per file** on GPU. For 30 files that is approximately **15–30 minutes total**.

---

## Loading Output in GIS Software

### ArcGIS Pro
1. `Insert → Add Data → XY Table Data`
2. Set X field = `longitude_wgs84`, Y field = `latitude_wgs84`, CRS = GCS WGS 1984
3. Or use `cx_easting` / `cy_northing` with NAD83(2011) Georgia East coordinate system

### QGIS
1. `Layer → Add Layer → Add Delimited Text Layer`
2. Set X = `longitude_wgs84`, Y = `latitude_wgs84`, CRS = EPSG:4326
3. Right-click layer → `Export → Save Features As` to reproject to EPSG:6444 if needed
