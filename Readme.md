# Poultry Farm Geospatial Detection Pipeline

<div align="center">
  <p><strong>A production-grade, end-to-end geospatial pipeline for detecting poultry houses in high-resolution NAIP imagery.</strong></p>
</div>

## 📖 Overview

This project bridges the gap between exploratory data science and robust production engineering. It automates the entire lifecycle of geospatial object detection—from raw GeoTIFF preprocessing to advanced machine learning inference, and finally to GIS-ready vector exports and county-level analytics. 

**Context:** The primary goal of this tool is to identify and map poultry houses using high-resolution aerial imagery (NAIP). By extracting precise footprints and clustering them into active farm units, researchers and policymakers can model agricultural density, estimate roof runoff, and assess potential environmental impacts on local watersheds.

Whether you are training custom self-supervised models on remote sensing data or running bulk inference across terabytes of imagery, this pipeline provides a highly optimized, CLI-driven interface.

---

## 📑 Table of Contents
1. [Key Capabilities](#-key-capabilities)
2. [Installation & Setup](#-installation--setup)
3. [Project Layout](#-project-layout)
4. [Architecture & Modules](#-architecture--modules)
5. [Cross-Platform Script Execution](#-cross-platform-script-execution)
6. [Foundation Model Pretraining (SSL)](#-foundation-model-pretraining-ssl)
7. [Monitoring & Visualization](#-monitoring--visualization)
8. [CLI Reference Guide](#-cli-reference-guide)
9. [Configuration-Driven Execution](#️-configuration-driven-execution)
10. [Troubleshooting & Known Issues](#-troubleshooting--known-issues)
11. [Testing & Validation](#-testing--validation)

---

## ✨ Key Capabilities

### 🌍 Geospatial Data Processing
- **High-Performance Slicing:** Efficiently slice massive GeoTIFFs into manageable tiles. Built-in **tile-level multiprocessing** ensures maximum CPU utilization even on single, monolithic files.
- **Memory-Efficient Stitching:** Rebuild massive rasters from smaller tiles using **windowed writing**, circumventing the out-of-memory (OOM) errors common in naive GIS scripts.
- **Format Conversion:** Rapidly convert GeoTIFFs to display-friendly formats (JPG/PNG) while normalizing data types.

### 🧠 Advanced Machine Learning
- **Tiled Inference:** Stream large rasters iteratively through YOLO (v8/v11) or RT-DETR models. This lazy-loading approach allows inference on multi-gigabyte files on standard hardware.
- **Self-Supervised Learning (SSL):** Native support for Masked Image Modeling (MIM) using Vision Transformers (ViT) and Sparse Masked Modeling (SparK) for CNNs (ResNet, CSPDarknet).
- **Sequential Grid Search:** Conduct massive hyperparameter sweeps for pretraining and segmentation tasks. Experiments are run reliably one after another, ensuring stability and accurate metric tracking without the overhead of multiprocessing.
- **Rich Progress Reporting:** Real-time verbose feedback showing current batch loss, running averages, learning rates, and pixel-level accuracy.
- **Stateful Resumption:** If interrupted, training resumes exactly where it left off, including optimizer and scheduler states.

### 🗺️ GIS Post-Processing & Analytics
- **Spatial Deduplication:** Suppress overlapping detections using Centroid Distance or Mask Intersection-over-Union (IoU).
- **Farm Clustering:** Automatically group individual poultry houses into farm entities using DBSCAN spatial clustering.
- **Vector Export:** Direct, frictionless conversion of tabular detection data to ESRI Shapefiles (Polygons/Bounding Boxes) and KML for Google Earth.
- **County-Level Analytics:** Calculate aggregated metrics, including NASS-corrected runoff volume screening based on precipitation data.

---

## 🛠 Installation & Setup

The project supports macOS, Linux, and Windows. Python 3.10 or higher is required.

### 1. POSIX (macOS / Linux)

```bash
# Clone the repository
git clone <repo_url> && cd <repo>

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install core and machine learning dependencies
pip install -r requirements.txt
pip install -r requirements-ml.txt

# Install the package in editable mode
pip install -e .
```

### 2. Windows (PowerShell)

We provide wrapper scripts (`*.cmd`) to ensure smooth execution and bypass policy issues on Windows. For local setup with GPU support:

```powershell
# Create environment
py -3.12 -m venv .venv

# Install CUDA-enabled Torch (Adjust versions to match your local CUDA toolkit)
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126

# Install dependencies and the package
.\.venv\Scripts\pip.exe install -e .
.\.venv\Scripts\pip.exe install -r requirements-ml.txt
```

---

## 🗂️ Project Layout

- `src/poultry_geo/`: Structured production Python package.
- `scripts/poultry_geo.py`: Direct terminal launcher.
- `scripts/poultry_geo.*`: CLI entry point wrappers for Windows (`.cmd`/`.ps1`) and Linux (`.sh`).
- `scripts/slice_tiles.*`: Wrappers for slicing GeoTIFFs into image tiles.
- `scripts/train_mim.*`: Wrappers for running self-supervised MAE pretraining on NAIP imagery.
- `scripts/train_ssl_grid.*`: Wrappers for stateful SSL (MIM) grid search training.
- `scripts/plot_ssl_grid.*`: Wrappers for generating grid search summary plots.
- `scripts/plot_run.*`: Wrappers for plotting individual run metrics (loss curves).
- `scripts/train_segmentation.*`: Wrappers for running stateful grid-search segmentation training (fine-tuning).
- `scripts/spark_grid_train.*`: Wrappers for SparK SSL grid search training.
- `scripts/train_spark.*`: Wrappers for single SparK SSL pretraining using CNNs.
- `configs/`: YAML and JSON configuration files for automated pipelines and grid searches.
- `tests/`: Unit tests for the extracted code and pipeline logic.
- `pyproject.toml`: Package metadata and console entrypoint definitions.
- `requirements.txt`: Core installation dependencies.
- `requirements-ml.txt`: Heavier ML installation dependencies (Torch, YOLO, etc.).

---

## 🏗️ Architecture & Modules

The core functionality lives under `src/poultry_geo/`:

| Module | Description |
|---|---|
| `geotiff.py` | Handles all raster I/O. Implements edge-snapped slicing and window-based stitching. |
| `inference.py` | Orchestrates sliding-window inference, generating georeferenced bounding boxes and polygons. |
| `gis.py` | Manages vector transformations, projecting CRS, and exporting ESRI Shapefiles. |
| `postprocess.py` | Applies NMS, centroid suppression, geometry merging, and DBSCAN clustering. |
| `training.py` | Wraps Ultralytics engines for streamlined YOLO/RT-DETR training. |
| `ssl_models.py` | Implements the adapters, ViTs, and decoders for MAE and SparK self-supervised learning. |
| `analytics.py` | Performs county-level aggregations and statistical bootstrapping for runoff. |

---

## 🖥️ Cross-Platform Script Execution

While you can run the application directly using the Python module (`python -m poultry_geo <command>`), we provide dedicated wrapper scripts in the `scripts/` directory to simplify execution across different operating systems. These wrappers handle environment paths, bypass Windows execution policies safely, and ensure consistent logging.

### Understanding the Wrappers
- **`.sh` scripts:** Bash wrappers for macOS and Linux.
- **`.cmd` scripts:** Command Prompt wrappers for Windows. These act as launchers that call the `.ps1` files.
- **`.ps1` scripts:** PowerShell scripts containing the actual Windows execution logic. (You rarely need to call these directly; use the `.cmd` files instead).

**Important Windows Note:** Always use the `.cmd` scripts from the Command Prompt or PowerShell, as they handle invoking PowerShell with `ExecutionPolicy Bypass` automatically.

---

## 🧪 Foundation Model Pretraining (SSL)

This pipeline supports automated, stable, and resume-aware Self-Supervised Learning (SSL). We support two primary methods: **MIM** (Vision Transformers) and **SparK** (Hierarchical CNNs).

### 1. Config-Driven Grid Search (Recommended)
Define your hyperparameters in a YAML file, and the pipeline will run experiments sequentially, maintaining full state for easy resumption.

#### Option A: MIM (MAE-style ViT)
*   **Config:** `ssl_config.yaml`
*   **Windows:** `.\scripts\train_ssl_grid.cmd --training-name "mim-search" --config ssl_config.yaml --verbose`
*   **Linux:** `./scripts/train_ssl_grid.sh --training-name "mim-search" --config ssl_config.yaml --verbose`

#### Option B: SparK (Sparse Masked Modeling)
*   **Supported Backbones:** `resnet50`, `yolo11l`, `yolov26l`
*   **Config:** `configs/spark_grid_search.yaml`
*   **Windows:** `.\scripts\spark_grid_train.cmd --training-name "spark-search" --config configs/spark_grid_search.yaml --verbose`
*   **Linux:** `./scripts/spark_grid_train.sh --training-name "spark-search" --config configs/spark_grid_search.yaml --verbose`

> **Note:** Set `image_dir` in your config to the directory containing your tiles (e.g., `./temp/test_512`).

### 2. Automated Sequential Campaign
If you want to run both SparK and MIM pretraining back-to-back without manual intervention, use the provided campaign scripts in the root directory. These scripts will automatically:
1.  Run the full SparK grid search.
2.  **Generate summary plots** for SparK in `trained_models/spark-search/figures`.
3.  Run the full MIM grid search.
4.  **Generate summary plots** for MIM in `trained_models/mim-search/figures`.

*   **Windows:** `.\run_campaign.cmd`
*   **Linux:** `chmod +x run_campaign.sh && ./run_campaign.sh`

### 3. Single Run CLI Training
If you want to run a single experiment without a config file:

#### MIM Single Run
```bash
# Windows (Example: Small ViT, 50 epochs)
.\scripts\train_mim.cmd --data-root ".\temp\test_512" --output-dir ".\runs\mim-single" --model-name S --epochs 50 --verbose

# Linux
./scripts/train_mim.sh --data-root "./temp/test_512" --output-dir "./runs/mim-single" --model-name S --epochs 50 --verbose
```

#### SparK Single Run
```bash
# Windows (Example: ResNet50)
.\scripts\train_spark.cmd --data-root ".\temp\test_512" --output-dir ".\runs\spark-single" --backbone-type resnet50 --epochs 50 --verbose

# Linux
./scripts/train_spark.sh --data-root "./temp/test_512" --output-dir "./runs/spark-single" --backbone-type resnet50 --epochs 50 --verbose
```

---

## 📊 Monitoring & Visualization

### 1. Real-Time Metrics
When running with `--verbose`, the terminal provides a multi-line progress bar showing:
- **Batch Loss:** Real-time reconstruction error.
- **Average Loss:** Epoch-to-date loss.
- **Learning Rate:** Current LR (including warmup/decay).
- **ETA:** Estimated time remaining for the full campaign.

### 2. Plotting Results
Generate summary visualizations after a grid search:

*   **Windows:** `.\scripts\plot_ssl_grid.cmd --training-name "mim-search"`
*   **Linux:** `./scripts/plot_ssl_grid.sh --training-name "mim-search"`

To visualize the loss curve for a specific experiment:
*   **Windows:** `.\scripts\plot_run.cmd "trained_models/mim-search/<EXP_FOLDER>"`
*   **Linux:** `./scripts/plot_run.sh "trained_models/mim-search/<EXP_FOLDER>"`

---

## 💻 CLI Reference Guide

The primary entry point is the `poultry-geo` command. 

> **Logging Note:** Every command executed is automatically appended to `command.log` with timestamps, arguments, and execution duration. Disable this per-run with `--no-log`.

### Data Preparation
```bash
# Slice a directory of GeoTIFFs into 256x256 PNGs, utilizing all CPU cores
poultry-geo slice ../Data/Raw ./tiles --tile-size 256 --format png --workers auto

# Reconstruct a full GeoTIFF from a directory of tiles
poultry-geo stitch ./tiles reconstructed.tif

# Extract boundary shapefiles from raw imagery extents
poultry-geo create-boundaries ../Data/Raw ./boundaries
```

### Inference & Post-Processing
```bash
# 1. Run inference (Streams tiles directly from the GeoTIFFs)
poultry-geo infer-yolo ../Data/Rasters weights/best.pt raw_detections.csv --window-size 3000 --stride 3000 --device cuda

# 2. Suppress overlapping/duplicate predictions
poultry-geo mask-nms raw_detections.csv clean_detections.csv --iou-threshold 0.5

# 3. Cluster individual houses into Farm units
poultry-geo cluster-farms clean_detections.csv farms.csv --eps-house 50.0 --min-samples 2
```

### GIS Export & Analytics
```bash
# Export precise detection polygons to an ESRI Shapefile
poultry-geo export-polygons clean_detections.csv ./shapefiles --crs EPSG:26916

# Convert detections to KML for stakeholder review in Google Earth
poultry-geo csv-to-kml clean_detections.csv export.kml

# Aggregate data to the county level
poultry-geo aggregate-counties clean_detections.csv county_summary.csv
```

---

## ⚙️ Configuration-Driven Execution

For complex workflows, the CLI supports JSON/YAML configuration files to orchestrate the pipeline automatically.

### End-to-End Pipeline
Run an entire pipeline (slicing, inference, post-processing, and export) via a single config file:
```bash
poultry-geo run-pipeline configs/pipeline.example.json
```

---

## 🛑 Troubleshooting & Known Issues

### 1. `command.log` File Lock (Windows)
If a background training process crashes or is forcefully terminated, PowerShell may leave a write-lock on the `command.log` file. You will see an error like: *The process cannot access the file because it is being used by another process.*
**Fix:** The wrappers are designed to fail gracefully and continue printing to the terminal, but to restore logging, find and terminate any orphaned `pwsh.exe` or `python.exe` processes, then delete the `command.log` file.

### 2. YAML Schema Drift
When running grid searches, ensure your configuration files strictly adhere to the schema expected by the pipeline. For example, use `image_dir` instead of `root`, and `image_size` instead of `crop_size`. If you encounter a `ValueError: config.yaml is missing data.image_dir`, double-check your parameter names against the provided examples in `ssl_config.yaml`.

### 3. Command Line Argument Parsing
When passing paths or string values to the wrapper scripts, it is highly recommended to use the equals sign (`=`) syntax to prevent parsing errors due to line breaks or spaces in PowerShell.
**Example:** `--training-name="my-run"` instead of `--training-name "my-run"`.

---

## 🧪 Testing & Validation

The codebase maintains rigorous quality standards, informed by Test-Driven Development (TDD). 

Run the test suite:
```bash
pytest
```
*The suite includes over 140 unit tests validating geospatial logic, coordinate transformations, multiprocessing locks, ML tensor shapes, and grid-search statefulness.*

---

## 📝 License & Contact
*Add your licensing and maintainer contact information here.*