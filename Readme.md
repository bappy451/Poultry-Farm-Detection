# Poultry Farm Geospatial Detection Pipeline

This `Code` directory now packages the notebook workflow as a structured Python project: GeoTIFF slicing, YOLO inference, georeferenced CSV generation, duplicate suppression, rectangle and mask merging, farm clustering, GIS export, shapefile conversion, KML export, SAM or SAM2 helpers, county analytics, transformer-model experiments, model export, and config-driven orchestration.

## Status

The projectized workflow is now runnable from the terminal and covers the major notebook families in `notebook/` and `notebook_new/`.

What is complete now:

- installable Python package under `src/poultry_geo/`
- CLI entrypoint for the operational and research-support workflows
- unit tests for the structured modules and notebook-derived helpers
- real-asset smoke testing against the provided training data and NAIP GeoTIFFs
- a local `data.yaml` wired to the actual dataset in `../Data/poultry_houses_db-2`
- code-backed replacements for the notebook utilities around conversion, visualization, shapefile analysis, DETR or Mask2Former training, and CSV or KML export

What is still intentionally outside the structured package:

- one-off manuscript figures, exploratory plots, and audit notebooks
- advanced SAM2 runs when the external `sam2` package and checkpoints are not installed locally
- any notebook cell that depends on hard-coded personal paths, ad hoc credentials, or manual interactive inspection

So the archive notebooks remain the research record, but the reusable functionality has been pulled into the package surface under `src/poultry_geo/`.

## Local Assets

These local assets are already present in this workspace:

- training dataset: [../Data/poultry_houses_db-2](</Users/abedinm/Desktop/01_MS_Code/Data/poultry_houses_db-2>)
- dataset YAML: [../Data/poultry_houses_db-2/data.yaml](/Users/abedinm/Desktop/01_MS_Code/Data/poultry_houses_db-2/data.yaml)
- test GeoTIFF folder: [../Data/Test_ms_ga](</Users/abedinm/Desktop/01_MS_Code/Data/Test_ms_ga>)
- trained poultry weights: [../01 Data & Code/runs/poultry](</Users/abedinm/Desktop/01_MS_Code/01 Data & Code/runs/poultry>)

The dataset YAML now uses the real local layout:

```yaml
path: .
train: train/images
val: valid/images
test: test/images
nc: 1
names:
  - poultry house
```

## Project Layout

- `src/poultry_geo/`
  Structured production package.
- `scripts/poultry_geo.py`
  Direct terminal launcher.
- `scripts/poultry_geo.ps1` and `scripts/poultry_geo.cmd`
  Windows launchers for the full CLI with append-only logging.
- `scripts/slice_tiles.sh`
  Bash wrapper for slicing one GeoTIFF or a directory of GeoTIFFs into `tif`, `png`, or `jpg` tiles.
- `scripts/slice_tiles.ps1` and `scripts/slice_tiles.cmd`
  Windows launchers for the common slicing workflow with append-only logging.
- `configs/pipeline.example.json`
  Example config for `run-pipeline`.
- `tests/`
  Unit tests for the extracted code.
- `notebook/`
  Flattened notebook mirror.
- `notebook_new/`
  Newer working notebooks and autosaves.
- `pyproject.toml`
  Package metadata and console entrypoint.
- `requirements.txt`
  Core install for package development and tests.
- `requirements-ml.txt`
  Heavier ML install for YOLO, export, and SAM.
- `Makefile`
  Convenience setup and test commands.

## Modules

- `geotiff.py`
  GeoTIFF discovery, edge-snapped no-padding slicing, and stitching back to original extent.
- `inference.py`
  Tiled YOLO inference and geospatial detection CSV export.
- `postprocess.py`
  Centroid suppression, mask IoU suppression, rectangle creation, polygon merging, and farm clustering.
- `gis.py`
  Boundary shapefile export, polygon or bbox shapefile export, shapefile merging, dimension analysis, and farm shapefile creation.
- `conversion.py`
  GeoTIFF to image conversion, shapefile to CSV conversion, row sampling, and CSV to KML export.
- `sam.py`
  SAM refinement using georeferenced point and box prompts, plus optional SAM2 visualization helpers.
- `visualization.py`
  Detection overlays for GeoTIFFs and standard JPG or PNG imagery.
- `training.py`
  Ultralytics training wrapper for YOLO and RT-DETR family models.
- `transformer_models.py`
  HuggingFace DETR and Mask2Former dataset wrappers, training helpers, and DETR inference.
- `exporting.py`
  Ultralytics model export helpers.
- `analytics.py`
  County aggregation, NASS correction, runoff screening, and bootstrap helpers.
- `pipeline.py`
  JSON-configured orchestration for end-to-end runs.
- `cli.py`
  Terminal interface over the modules above.

## Setup

Use Python 3.12 on this machine. The current environment was created with `/opt/anaconda3/bin/python3.12`.

```bash
cd /Users/abedinm/Desktop/01_MS_Code/Code
/opt/anaconda3/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-ml.txt
```

Core package checks:

```bash
.venv/bin/python -m poultry_geo --help
.venv/bin/python -m pytest -q
```

## Windows Quick Start

On Windows, the easiest path is to use the new `*.cmd` wrappers from PowerShell or Command Prompt. They automatically invoke the PowerShell scripts with execution-policy bypass and append every run to [command.log](/Users/abedinm/Desktop/01_MS_Code/Code/command.log).

From the `Code` directory on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\pip.exe install -e .
.\.venv\Scripts\pip.exe install -r requirements-ml.txt
```

If you want NVIDIA GPU support on Windows or Linux, install the official CUDA-enabled
PyTorch wheel first as shown above. Otherwise `requirements-ml.txt` may resolve to a
CPU-only Torch build.

Then run either the full CLI wrapper:

```powershell
.\scripts\poultry_geo.cmd --help
.\scripts\poultry_geo.cmd create-boundaries ..\Data\Test_ms_ga .\boundary_shapes
```

Or the dedicated slicer wrapper:

```powershell
.\scripts\slice_tiles.cmd --input-dir ..\Data\Test_ms_ga --output-dir .\tmp\tiles_256_png --tile-size 256 --format png --workers auto
```

If you prefer PowerShell directly instead of the `.cmd` wrappers:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\slice_tiles.ps1 --input-dir ..\Data\Test_ms_ga --output-dir .\tmp\tiles_256_png --tile-size 256 --format png --workers auto
```

The Windows wrappers support the same append-only log controls:

- `--log-file PATH`
  Write to a different log file instead of `Code\command.log`.
- `--no-log`
  Disable wrapper logging for one run.
- `--python PATH`
  Force a specific Python executable.

Each logged command records its command line, working directory, start and finish
timestamps, exit code, and elapsed duration. Python failures retain the complete
traceback. PowerShell wrapper failures also record the exception type, category,
source position, and PowerShell stack trace.

Commands also display progress and write the same progress records to the log.
File-oriented operations report a percentage, item count, current filename, and
estimated time remaining. Operations without a reliable total display an animated
`working` bar with elapsed time. The terminal updates one progress row in place;
the log stores only meaningful progress events. Pass `--no-progress` to disable
progress output.

## CLI Workflows

The package can be run either as the installed console script or directly through Python:

```bash
.venv/bin/python -m poultry_geo --help
.venv/bin/python scripts/poultry_geo.py --help
```

## NAIP Self-Supervised Data Loading

`poultry_geo.naip_data` provides a PyTorch input pipeline for recursively nested
NAIP tiles:

```python
from poultry_geo.naip_data import NAIPDataset, get_dataloaders

train_loader, val_loader = get_dataloaders(
    "tmp/test_jpg",
    batch_size=64,
    crop_size=224,
    precision="bf16",
    num_workers=8,
    prefetch_factor=4,
)

batch = next(iter(train_loader))
images = batch["image"]  # [B, C, H, W], CPU tensor
paths = batch["path"]
```

The dataset supports RGB JPG/PNG, multi-band GeoTIFF, and CHW/HWC NPY arrays.
Training samples receive synchronized random crops and flips across all bands,
plus channel-aware color jitter. Validation uses deterministic center crops.
Train/validation splitting occurs by parent source directory to prevent adjacent
tiles from the same source raster leaking across splits. Distributed samplers
are selected automatically after `torch.distributed` is initialized.

Input precision may be `fp32`, `fp16`, or `bf16`. NF4 is model-weight
quantization and is intentionally not used as an image tensor dtype.

## Inspect SSL Backbone

`poultry_geo.ssl_models` provides a lightweight 1x1 `BandAdapter` and a
custom `RemoteSensingViT` backbone for self-supervised inspection runs:

```python
from poultry_geo.ssl_models import BandAdapter, RemoteSensingViT

adapter = BandAdapter(in_channels=4, precision="bf16")
backbone = RemoteSensingViT(
    image_size=224,
    patch_size=16,
    embed_dim=192,
    depth=12,
    num_heads=3,
    precision="bf16",
)
```

To run one real batch through the adapter and backbone and print tensor shapes:

macOS/Linux:

```bash
./scripts/inspect_backbone.sh --data-root ./tmp/test_jpg --batch-size 2 --crop-size 224 --channels 3 --patch-size 16 --embed-dim 192 --depth 12 --num-heads 3 --precision bf16
```

Windows:

```powershell
.\scripts\inspect_backbone.cmd --data-root .\tmp\test_jpg --batch-size 2 --crop-size 224 --channels 3 --patch-size 16 --embed-dim 192 --depth 12 --num-heads 3 --precision bf16
```

Direct CLI on either platform:

```bash
python -m poultry_geo inspect-backbone tmp/test_jpg --batch-size 2 --crop-size 224 --channels 3 --patch-size 16 --embed-dim 192 --depth 12 --num-heads 3 --precision bf16
```

This prints:

- input tensor shape `[B, n, H, W]`
- adapted tensor shape `[B, 3, H, W]`
- token tensor shape `[B, num_patches, embed_dim]`
- patch grid, requested precision, resolved precision, and device

On Windows, use:

```powershell
.\scripts\poultry_geo.cmd --help
```

Main commands:

```bash
# Slice and stitch GeoTIFFs
.venv/bin/python -m poultry_geo slice input.tif output_tiles --tile-size 3000
.venv/bin/python -m poultry_geo stitch output_tiles rebuilt.tif

# Shell wrapper for directory-based slicing into image tiles
./scripts/slice_tiles.sh --input-dir geotiffs/ --output-dir output_tiles/ --tile-size 256 --format png

# Convert GeoTIFFs and convert shapefiles
.venv/bin/python -m poultry_geo convert-geotiff geotiffs/ converted_images/ --format jpg
.venv/bin/python -m poultry_geo shapefile-to-csv polygons.shp polygons.csv
.venv/bin/python -m poultry_geo shapefile-to-projected-csv polygons.shp polygons_projected.csv
.venv/bin/python -m poultry_geo csv-to-kml detections.csv detections.kml

# Create GeoTIFF boundary shapefiles
.venv/bin/python -m poultry_geo create-boundaries geotiffs/ boundary_shapes/

# YOLO inference and post-processing
.venv/bin/python -m poultry_geo infer-yolo geotiffs/ best.pt detections.csv --window-size 3000 --stride 3000
.venv/bin/python -m poultry_geo centroid-suppress detections.csv detections_dedup.csv
.venv/bin/python -m poultry_geo mask-nms detections.csv detections_mask_nms.csv
.venv/bin/python -m poultry_geo create-rectangles detections.csv detections_rect.csv
.venv/bin/python -m poultry_geo merge-polygons detections.csv detections_merged.csv
.venv/bin/python -m poultry_geo merge-adjacent-masks detections_rect.csv detections_rect_merged.csv
.venv/bin/python -m poultry_geo merge-masks-by-edge detections_rect.csv detections_edge_merged.csv
.venv/bin/python -m poultry_geo cluster-farms detections_dedup.csv farm_centroids.csv

# GIS export and shapefile analysis
.venv/bin/python -m poultry_geo export-polygons detections.csv polygon_shapes/
.venv/bin/python -m poultry_geo export-bboxes detections.csv bbox_shapes/
.venv/bin/python -m poultry_geo merge-overlapping-shapefile polygon_shapes/polygon_shapes.shp merged.shp
.venv/bin/python -m poultry_geo process-shapefile-dimensions merged.shp merged_with_dims.shp
.venv/bin/python -m poultry_geo filter-shapefile merged_with_dims.shp filtered.shp
.venv/bin/python -m poultry_geo identify-farms-shapefile filtered.shp poultry_farms.shp

# SAM refinement
.venv/bin/python -m poultry_geo sam-refine detections.csv geotiffs/ sam_output/ weights/sam_vit_h_4b8939.pth
.venv/bin/python -m poultry_geo sam-bbox-refine bbox_projected.csv geotiffs/ sam_bbox_output/ weights/sam_vit_h_4b8939.pth

# Visualization
.venv/bin/python -m poultry_geo visualize-geotiffs geotiffs/ detections.csv annotated_geotiffs/
.venv/bin/python -m poultry_geo visualize-jpgs detections_rect.csv converted_images/ annotated_jpgs/

# Training and export
.venv/bin/python -m poultry_geo train-yolo data.yaml runs/poultry --models yolo11n-seg.pt
.venv/bin/python -m poultry_geo train-yolo data.yaml runs/poultry --models rtdetr-l.pt --family rtdetr
.venv/bin/python -m poultry_geo train-detr train/images train/_annotations.coco.json runs/detr
.venv/bin/python -m poultry_geo train-mask2former train/images train/_annotations.coco.json runs/mask2former
.venv/bin/python -m poultry_geo export-model weights/best.pt --format torchscript

# County analytics
.venv/bin/python -m poultry_geo aggregate-counties detections_with_county.csv county_summary.csv
.venv/bin/python -m poultry_geo runoff-screening county_summary.csv county_precipitation.csv county_runoff.csv

# Config-driven orchestration
.venv/bin/python -m poultry_geo run-pipeline configs/pipeline.example.json
```

## Local Usage Examples

From `Code/`, the real dataset and test rasters can be used directly:

```bash
# Verify tests
.venv/bin/python -m pytest -q

# Train from the provided poultry dataset
.venv/bin/python -m poultry_geo train-yolo \
  ../Data/poultry_houses_db-2/data.yaml \
  runs/poultry \
  --models ../01\ Data\ \&\ Code/runs/poultry/yolov8n-seg-bs16-ep200/weights/best.pt \
  --epochs 1 --batch-size 1 --image-size 256 --device mps

# Create boundaries for the provided test GeoTIFFs
.venv/bin/python -m poultry_geo create-boundaries \
  ../Data/Test_ms_ga \
  boundary_shapes

# Run inference on the provided GeoTIFF folder with an existing poultry checkpoint
.venv/bin/python -m poultry_geo infer-yolo \
  ../Data/Test_ms_ga \
  ../01\ Data\ \&\ Code/runs/poultry/yolov8s-seg-bs16-ep200/weights/best.pt \
  detections.csv \
  --window-size 3000 --stride 3000 --confidence-threshold 0.01 --device mps
```

## Validation

Current verified state on this machine:

- full unit suite passes: `47 passed`
- all CLI subcommands parse and expose help
- `slice` ran on a real GeoTIFF from `../Data/Test_ms_ga`
- `stitch` rebuilt the original raster extent correctly
- `create-boundaries` generated `11` shapefiles from the `Test_ms_ga` folder
- `infer-yolo` ran on a real 3000x3000 crop from the provided NAIP imagery and produced `10` detections
- `centroid-suppress`, `mask-nms`, `cluster-farms`, and `export-polygons` all ran successfully on that real detection output
- notebook-derived helpers for rectangles, bbox export, shapefile dimensions, polygon merging, and shapefile filtering are covered by unit tests
- `run-pipeline` completed successfully on the real inference crop
- `aggregate-counties` and `runoff-screening` passed on smoke-test county inputs built from real detections
- `export-model` succeeded to TorchScript
- `train-yolo` completed a 1-epoch dry run on a tiny subset built from the real poultry dataset

Smoke outputs from the latest run are stored under [tmp/smoke_real](</Users/abedinm/Desktop/01_MS_Code/Code/tmp/smoke_real>).

## Pipeline Outputs

`run-pipeline` writes a working directory with artifacts such as:

- boundary shapefiles
- raw detection CSV
- centroid-suppressed CSV
- mask-NMS CSV when enabled
- farm centroid CSV
- polygon shapefile
- SAM overlays when enabled
- `pipeline_outputs.json` summary file

## Known Prerequisites

- `sam-refine` requires a local SAM checkpoint such as `sam_vit_h_4b8939.pth`; that checkpoint is not bundled in this workspace
- training and inference expect the heavier ML stack from `requirements-ml.txt`
- some examples assume Apple Metal (`--device mps`) on this machine; use `cpu` or `cuda` as appropriate on other systems

## Notebook Archive

The notebook folders remain useful as the research record:

- [notebook](</Users/abedinm/Desktop/01_MS_Code/Code/notebook>)
- [notebook_new](</Users/abedinm/Desktop/01_MS_Code/Code/notebook_new>)

They still include exploratory work, checkpoints, manuscript analysis, and one-off experiments. The structured package in `src/poultry_geo/` is the maintained terminal-facing project surface that recreates the reusable notebook functionality.
