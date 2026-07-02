# 08 - Codebase Runbook: How to Run, Test, and Review Outputs

This runbook is the operational guide for the current `Code` project. It explains what to run, what each command does, what files to expect, and where the important outputs are written.

The project is a notebook-derived, Python-packaged geospatial detection pipeline for poultry-house and farm analysis. The production code lives under `src/poultry_geo`; the original research notebooks are preserved under `notebook` and `notebook_new`.

## 1. Current Verified Baseline

During the latest audit, the full automated suite was run from the `Code` directory:

```bash
cd /Users/abedinm/Desktop/01_MS_Code/Code
./.venv/bin/python -m pytest -q --tb=short --disable-warnings
```

Expected result from the verified run:

```text
177 passed, 11 skipped, 8 warnings
```

The skipped tests are environment-gated checks such as CUDA-only tests and DDP socket smoke tests when the local machine does not allow binding a localhost port. They are not skipped because of known functional failures.

## 2. Repository Map

Important paths:

```text
Code/
  src/poultry_geo/              Production geospatial, ML, GIS, SSL, and analytics package
  src/downstream_finetune/      Downstream YOLO/ResNet fine-tuning helpers
  scripts/                      Cross-platform wrappers and helper scripts
  configs/                      YAML/JSON configs for training and pipelines
  tests/                        Pytest suite
  notebook/                     Flattened mirror of the original notebook tree
  notebook_new/                 Newer/renamed notebook working set plus checkpoint autosaves
  report/                       Technical documentation and this runbook
  Readme.md                     Main project overview
  command.log                   Append-only command log; ignored by Git
  tmp/                          Scratch output area; ignored by Git

../Data/
  Test_ms_ga/                   Provided GeoTIFF files for slicing, conversion, and inference testing
  poultry_houses_Training/      YOLO-format training and validation data
  poultry_houses_test/          YOLO-format test data
```

Notebook status:

```text
notebook + notebook_new contain 125 .ipynb files.
```

`notebook` is the traceable flattened copy of the original notebook tree. `notebook_new` is the expanded working set, including Jupyter `*-checkpoint.ipynb` autosaves. Treat checkpoint notebooks as autosaves, not primary workflows.

## 3. Environment Setup

Use Python 3.10 or newer. The current local environment is `.venv` under `Code`.

macOS or Linux:

```bash
cd /Users/abedinm/Desktop/01_MS_Code/Code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt
python -m pip install -e .
```

Windows PowerShell:

```powershell
cd C:\path\to\01_MS_Code\Code
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

For NVIDIA GPU training on Windows or Linux, install the official CUDA-enabled PyTorch wheel before `requirements-ml.txt`. The requirements file includes a CUDA 12.6 example, but match the command to the local CUDA driver.

## 4. Command Logging

The main CLI and wrapper scripts append command output to one project log file:

```text
Code/command.log
```

This log is append-only by default and is ignored by Git. You can disable logging for one run:

```bash
poultry-geo --no-log slice ../Data/Test_ms_ga ./tmp/tiles --tile-size 256 --format png
```

You normally do not need `scripts/run_logged.sh`. Use the real command directly:

```bash
./scripts/slice_tiles.sh \
  --input-dir ../Data/Test_ms_ga \
  --output-dir ./tmp/tiles_256_png \
  --tile-size 256 \
  --format png \
  --workers auto
```

Expected behavior:

- terminal output is still visible
- the same run is appended to `command.log`
- logs and `tmp/` outputs stay out of Git

## 5. Data Checks Before Running

The YOLO dataset YAML currently points to the actual data directories under `/Users/abedinm/Desktop/01_MS_Code/Data`.

Verified dataset counts:

```text
train images: 4342
val images:   482
test images:  702
classes:      poultry_house, broiler breeder, broiler house
```

Run this quick check after moving the project to another machine:

```bash
python - <<'PY'
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("../Data/poultry_houses_Training/data.yaml").read_text())
for key in ("train", "val", "test"):
    path = Path(cfg[key])
    images = [p for p in path.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print(key, path, "exists=", path.exists(), "images=", len(images))
print("classes:", cfg["names"])
PY
```

If any path prints `exists=False`, edit `../Data/poultry_houses_Training/data.yaml` so `train`, `val`, and `test` point to the local image folders.

## 6. GeoTIFF Slicing

Use this when you want to turn large GeoTIFFs into square TIFF/PNG/JPG tiles for training, inspection, or downstream batch processing.

Recommended wrapper command:

```bash
cd /Users/abedinm/Desktop/01_MS_Code/Code
./scripts/slice_tiles.sh \
  --input-dir ../Data/Test_ms_ga \
  --output-dir ./tmp/tiles_256_png \
  --tile-size 256 \
  --format png \
  --workers auto
```

Equivalent package command:

```bash
poultry-geo slice ../Data/Test_ms_ga ./tmp/tiles_256_png --tile-size 256 --format png --workers auto
```

Expected behavior:

- directory input writes one output subdirectory per source GeoTIFF
- single-file input writes tiles directly under the output directory
- default slicing is non-overlapping: `stride = tile_size`
- final windows snap to the image edge
- no zero padding is written
- if the image size is not divisible by `tile_size`, only the last edge window overlaps
- `--stride` smaller than `--tile-size` creates intentional overlap
- `--stride` larger than `--tile-size` is rejected because it leaves gaps
- `--workers auto` uses file-level multiprocessing for directory input
- one worker owns one full GeoTIFF from open through final tile write
- if the host environment blocks process pools, the command falls back to serial file-level slicing with a warning

Examples:

```bash
# 64x64 overlapping JPG tiles
./scripts/slice_tiles.sh \
  --input-dir ../Data/Test_ms_ga \
  --output-dir ./tmp/tiles_64_jpg \
  --tile-size 64 \
  --stride 32 \
  --format jpg \
  --jpeg-quality 92 \
  --workers 4

# 256x256 GeoTIFF tiles
./scripts/slice_tiles.sh \
  --input-dir ../Data/Test_ms_ga \
  --output-dir ./tmp/tiles_256_tif \
  --tile-size 256 \
  --format tif \
  --workers auto
```

## 7. GeoTIFF Conversion to PNG or JPG

Use conversion when you need one display-friendly image per GeoTIFF rather than many tiles.

```bash
poultry-geo convert-geotiff ../Data/Test_ms_ga ./tmp/geotiff_jpg --format jpg --jpeg-quality 90
```

Expected behavior:

- input folder structure is mirrored under the output directory
- single-band images are expanded to RGB
- multiband images use the first three bands for display
- non-uint8 rasters are normalized into display range

## 8. Dataset Catalogs

For large tile folders, build a CSV catalog once to avoid repeated recursive scans:

```bash
poultry-geo catalog-dataset ./tmp/tiles_256_png ./trained_models/data_catalog.csv
```

Expected output:

```text
trained_models/data_catalog.csv
```

Point SSL configs at the tile directory and, where supported, at the catalog path.

## 9. Supervised YOLO or RT-DETR Training

The corrected data YAML is:

```text
../Data/poultry_houses_Training/data.yaml
```

YOLO smoke run:

```bash
poultry-geo train-yolo \
  ../Data/poultry_houses_Training/data.yaml \
  ./trained_models/yolo_smoke \
  --models yolo11s-seg.pt \
  --epochs 1 \
  --batch-size 2 \
  --image-size 640 \
  --device cpu
```

Production-style YOLO run:

```bash
poultry-geo train-yolo \
  ../Data/poultry_houses_Training/data.yaml \
  ./trained_models/yolo_runs \
  --models yolo11l-seg.pt \
  --epochs 50 \
  --batch-size 8 \
  --image-size 640 \
  --device cuda
```

RT-DETR experiment:

```bash
poultry-geo train-yolo \
  ../Data/poultry_houses_Training/data.yaml \
  ./trained_models/rtdetr_runs \
  --models rtdetr-l.pt \
  --family rtdetr \
  --epochs 50 \
  --batch-size 4 \
  --image-size 640 \
  --device cuda
```

Expected outputs:

```text
trained_models/<run_name>/
  weights/best.pt
  weights/last.pt
  results.csv
  args.yaml
```

Notes:

- the first run may download model weights if they are not already local
- CPU smoke runs are slow but useful for validating paths
- GPU production runs need CUDA-enabled PyTorch
- training output folders are ignored by Git

## 10. ResNet50 Segmentation Fine-Tuning

The ResNet50 fine-tuning helper lives in `src/downstream_finetune/finetune_resnet50.py`.

CPU-safe smoke command:

```bash
python src/downstream_finetune/finetune_resnet50.py \
  --data ../Data/poultry_houses_Training/data.yaml \
  --project ./runs/segment/resnet50_smoke \
  --epochs 1 \
  --batch 1 \
  --imgsz 64 \
  --num-workers 0
```

Expected outputs:

```text
runs/segment/resnet50_smoke/
  args.yaml
  results.csv
  confusion_matrix.png
  weights/best.pt
  weights/last.pt
```

Important behavior:

- `pretrained=False` does not download ResNet50 backbone weights
- `--num-workers 0` is safest on CPU and restricted environments
- CUDA runs can increase `--num-workers`

## 11. Self-Supervised Learning: MIM and SparK

MIM uses ViT-style masked image modeling. SparK uses sparse masked modeling for hierarchical CNN-style encoders.

Before running either, edit these config files so `data.image_dir` points to your tile folder:

```text
configs/mim_grid_config.yaml
configs/spark_grid_search.yaml
```

MIM grid search:

```bash
./scripts/train_ssl_grid.sh \
  --training-name mim-search \
  --config configs/mim_grid_config.yaml \
  --verbose
```

SparK grid search:

```bash
./scripts/spark_grid_train.sh \
  --training-name spark-search \
  --config configs/spark_grid_search.yaml \
  --verbose
```

Expected outputs:

```text
runs/ssl-grid-search/ or runs/spark-grid-search/
trained_models/<training-name>/
  grid_search_results.csv
  state.json
  <experiment_id>_.../
    last.pt
    best.pt
    metrics.json
```

Expected behavior:

- grid search runs sequentially to avoid GPU memory oversubscription
- interrupted runs can resume from checkpoints
- `pretrained` is optional in SSL grid configs and defaults to `false` when absent
- `spconv` is optional; sparse paths are used when it is installed, and dense/test fallbacks remain available

## 12. MississippiFM Workflow

MississippiFM is the overlap-aware foundation-model path. It uses adjacency between existing overlapping tiles instead of treating every crop as independent.

Create adjacency metadata:

```bash
python scripts/create_adjacency_index.py \
  ./tmp/tiles_256_png \
  ./trained_models/mississippifm/adjacency.csv \
  --overlap-ratio 0.5 \
  --tile-size 256
```

Pretrain:

```bash
poultry-geo mississippi-pretrain \
  configs/mississippi_pretrain.yaml \
  ./tmp/tiles_256_png \
  ./trained_models/mississippifm/adjacency.csv \
  ./trained_models/mississippifm/run_a \
  --model-size vit_b \
  --resume
```

Evaluate probes:

```bash
poultry-geo mississippi-eval-probes \
  configs/mississippi_probes.yaml \
  ./trained_models/mississippifm/run_a/best.pt \
  ./trained_models/mississippifm/run_a/probes \
  --resume
```

Run geographic benchmark:

```bash
poultry-geo mississippi-eval-benchmark \
  configs/mississippi_benchmark.yaml \
  ./trained_models/mississippifm/run_a/best.pt \
  ./trained_models/mississippifm/run_a/benchmark \
  --resume
```

Expected outputs:

```text
trained_models/mississippifm/
  adjacency.csv
  run_a/
    last.pt
    best.pt
    metrics.json
    probes/
    benchmark/
```

## 13. Inference on GeoTIFFs

After training, run tiled inference on the provided GeoTIFF folder:

```bash
poultry-geo infer-yolo \
  ../Data/Test_ms_ga \
  ./trained_models/yolo_runs/<run>/weights/best.pt \
  ./tmp/raw_detections.csv \
  --window-size 3000 \
  --stride 3000 \
  --batch-size 4 \
  --confidence-threshold 0.4 \
  --device cuda
```

Expected behavior:

- GeoTIFFs are streamed by window instead of loaded fully into memory
- no-padding, edge-snapped windows are used
- variable-size edge windows are shape-bucketed for batching
- detections are projected from pixel coordinates into geospatial coordinates

Expected output:

```text
tmp/raw_detections.csv
```

## 14. Post-Processing

Typical cleanup sequence:

```bash
poultry-geo centroid-suppress \
  ./tmp/raw_detections.csv \
  ./tmp/detections_centroid_clean.csv \
  --distance-threshold 10

poultry-geo mask-nms \
  ./tmp/detections_centroid_clean.csv \
  ./tmp/detections_mask_clean.csv \
  --iou-threshold 0.5

poultry-geo cluster-farms \
  ./tmp/detections_mask_clean.csv \
  ./tmp/farms.csv \
  --eps-house 50 \
  --min-samples 2
```

Expected outputs:

```text
tmp/detections_centroid_clean.csv
tmp/detections_mask_clean.csv
tmp/farms.csv
```

Use centroid suppression for nearby duplicate boxes. Use mask NMS when polygon geometry is available.

## 15. GIS Export

Export polygon detections:

```bash
poultry-geo export-polygons \
  ./tmp/detections_mask_clean.csv \
  ./tmp/shapefiles/polygons \
  --crs EPSG:26916
```

Export bounding boxes:

```bash
poultry-geo export-bboxes \
  ./tmp/detections_mask_clean.csv \
  ./tmp/shapefiles/bboxes \
  --crs EPSG:26916
```

Convert CSV polygons to KML:

```bash
poultry-geo csv-to-kml ./tmp/detections_mask_clean.csv ./tmp/detections.kml
```

Expected outputs:

```text
tmp/shapefiles/
  *.shp
  *.shx
  *.dbf
  *.prj
tmp/detections.kml
```

Open shapefiles in QGIS or ArcGIS. Open KML files in Google Earth for quick visual review.

## 16. Visualization and SAM Refinement

Overlay detections on GeoTIFFs:

```bash
poultry-geo visualize-geotiffs \
  ../Data/Test_ms_ga \
  ./tmp/detections_mask_clean.csv \
  ./tmp/overlays
```

SAM refinement with a local SAM checkpoint:

```bash
poultry-geo sam-refine \
  ../Data/Test_ms_ga \
  ./tmp/detections_mask_clean.csv \
  ./tmp/sam_refined.csv \
  --sam-checkpoint ./weights/sam_vit_h_4b8939.pth \
  --model-type vit_h
```

Expected outputs:

```text
tmp/overlays/
tmp/sam_refined.csv
```

SAM/SAM2 commands require local checkpoints and the relevant packages. If checkpoints are missing, the command should fail early with a path error.

## 17. End-to-End Pipeline Config

Start from:

```text
configs/pipeline.example.json
```

Edit these fields before running:

```json
{
  "working_dir": "runs/my_pipeline",
  "geotiff_dir": "../Data/Test_ms_ga",
  "model_path": "./trained_models/yolo_runs/<run>/weights/best.pt",
  "reference_crs": "EPSG:26916"
}
```

Run:

```bash
poultry-geo run-pipeline configs/pipeline.example.json
```

Expected behavior:

- the selected `steps` in the JSON decide which phases run
- intermediate CSVs and exports are written under `working_dir`
- command output is appended to `command.log`

## 18. Plotting and Review

Plot a single training run:

```bash
./scripts/plot_run.sh "trained_models/mim-search/<experiment_folder>"
```

Plot a grid search campaign:

```bash
./scripts/plot_ssl_grid.sh --training-name mim-search
```

Expected outputs:

```text
trained_models/<training-name>/figures/
  *.png
  summary_data.csv
```

## 19. Testing

Run the whole suite:

```bash
./.venv/bin/python -m pytest -q --tb=short --disable-warnings
```

Run important focused suites:

```bash
./.venv/bin/python -m pytest -q tests/test_geotiff.py tests/test_conversion.py
./.venv/bin/python -m pytest -q tests/test_cli.py tests/test_command_logging.py
./.venv/bin/python -m pytest -q tests/test_inference.py tests/test_postprocess.py tests/test_gis.py
./.venv/bin/python -m pytest -q tests/test_ssl_models.py tests/test_ssl_training.py
./.venv/bin/python -m pytest -q tests/test_mississippi_*.py
```

Expected result from the latest full local run:

```text
177 passed, 11 skipped, 8 warnings
```

## 20. Common Problems

`Permission denied: ./scripts/slice_tiles.sh`

The shell wrappers should be executable. Fix once:

```bash
chmod +x scripts/*.sh run_campaign.sh
```

`data.yaml` points to `Code/Data/...`

Edit `../Data/poultry_houses_Training/data.yaml` so it points to the actual `../Data/...` directories.

PyTorch tries to download ResNet50 weights during a no-pretrained run

The current code explicitly disables torchvision backbone weights when `pretrained=False`. If this reappears, check `build_resnet50_segmentation_model`.

PyTorch dataloader worker or shared-memory errors on CPU

Use `--num-workers 0` for CPU smoke runs.

DDP tests skip on local socket restrictions

This environment may block `127.0.0.1` socket binding. On normal training machines the DDP smoke tests should run.

GeoTIFF slicing seems to create overlap even without `--stride`

That is expected only at the final edge window when dimensions are not divisible by the tile size. It is edge snapping, not padding.

## 21. What Should Not Go Into Git

These are intentionally ignored:

```text
command.log
*.log
tmp/
temp/
runs/
trained_models/
output_slices/
shapefile_output/
*.tif, *.jpg, *.png, *.csv
```

Keep source code, tests, configs, and report markdown in Git. Keep generated tiles, model weights, logs, and GIS exports out of Git.
