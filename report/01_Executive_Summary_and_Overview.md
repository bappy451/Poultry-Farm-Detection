# 01 - Executive Summary and Project Overview

## 1. Project Domain and Challenge
The **Poultry Farm Geospatial Detection Pipeline** operates at the intersection of deep learning, geospatial analysis (GIS), and agricultural environmental monitoring. 

Monitoring concentrated animal feeding operations, specifically poultry farms, is critical for understanding their impact on local watersheds and ecosystems. Traditional on-the-ground surveys are slow, labor-intensive, and difficult to scale. Raw aerial imagery, such as the National Agriculture Imagery Program (NAIP) dataset, provides massive amounts of data. However, extracting meaningful, actionable insights from multi-gigabyte raster files requires specialized, robust, and scalable software architecture capable of handling the extreme memory and processing demands.

## 2. The Solution
This project provides a production-grade, end-to-end geospatial machine learning pipeline. It successfully transitions exploratory data science from brittle, ad-hoc Jupyter Notebooks into a highly robust, modular Python package (`poultry_geo`) accompanied by a Command-Line Interface (CLI).

It completely automates the workflow:
1. **Ingestion & Preprocessing:** Handling massive GeoTIFFs, slicing them without edge-padding, and using file-level multiprocessing for high-throughput tile extraction across GeoTIFF collections.
2. **Self-Supervised Learning (SSL):** Training foundation models on unannotated imagery to learn semantic representations of geography.
3. **Segmentation Fine-Tuning:** Specializing pretrained foundation models (YOLO variants, ResNet, DETR) to detect precise poultry house footprints.
4. **Streaming Inference:** Running inferences across entire regions using strided, overlapping sliding windows on disk-backed rasters to prevent out-of-memory errors.
5. **Post-Processing & GIS Transformation:** Cleaning bounding box artifacts, suppressing overlapping masks, and converting pixel-space predictions into standardized GIS vectors (Shapefiles, KML).
6. **Clustering & Analytics:** Grouping individual detections into "farm" entities using DBSCAN and running county-level environmental runoff analytics.

## 3. High-Level Pipeline Architecture
The system is divided into well-defined, testable modules under `src/poultry_geo/`:
- **`geotiff.py`**: Handles raster I/O, window-based slicing, and stitching.
- **`naip_data.py`**: A specialized PyTorch DataLoader capable of handling recursive directories, multi-band inputs, and synchronized multi-channel augmentations.
- **`ssl_models.py`**: Wrappers for Masked Image Modeling (MIM) and Sparse Masked Modeling (SparK).
- **`training.py`**: Orchestrates stateful grid searches and downstream Ultralytics YOLO/RT-DETR training.
- **`inference.py`**: Manages sliding-window inference on massive rasters.
- **`postprocess.py` & `gis.py`**: Execute NMS, centroid suppression, spatial clustering, and vector projection.
- **`analytics.py`**: Computes NASS-corrected runoff volumes and county-level statistical bootstrapping.

## 4. Current Status of the Project
The project has achieved its primary structural goals:
- **Fully Projectized**: The R&D workflow is successfully packaged as `poultry_geo`, installable via `pip` and executable natively via a CLI on Windows, macOS, and Linux.
- **Stable Tooling**: CLI wrappers ensure logging and cross-platform policy bypass without compromising system security.
- **Operational Documentation**: `08_Codebase_Runbook.md` now provides step-by-step commands, expected outputs, and troubleshooting for the full codebase.
- **Current Data Wiring**: `../Data/poultry_houses_Training/data.yaml` has been corrected to point to the actual training, validation, and test image folders under `/Users/abedinm/Desktop/01_MS_Code/Data`.
- **Model Success**: Downstream evaluation confirms that Self-Supervised Learning (particularly YOLO11l via SparK) provides superior F1 scores (~0.9845) over baseline architectures.
- **Robustness**: The latest local audit ran the full test suite with `177 passed, 11 skipped, 8 warnings`, covering coordinate projections, command logging, no-padding slicing, conversion, GIS export, tensor geometry, stateful training, and MississippiFM modules.

## 5. Documentation Structure
To fully document the depth of this pipeline, the report is split into the following detailed sections:
- `02_Geospatial_Data_Processing.md`: Detailed mechanics of GeoTIFF handling and data loading.
- `03_Machine_Learning_and_SSL.md`: In-depth analysis of MIM vs. SparK methodologies.
- `04_Downstream_Fine_Tuning.md`: YOLO evaluation, metrics, and transfer learning results.
- `05_GIS_PostProcessing_and_Analytics.md`: Spatial clustering, deduplication, and environmental calculations.
- `06_Engineering_Design_and_Future_Plans.md`: Architectural choices, hardware constraints, and roadmap.
- `07_Dataset_Description.md`: Data layout, YOLO data YAML, provided GeoTIFFs, and notebook collections.
- `08_Codebase_Runbook.md`: Concrete setup and run instructions with expected artifacts.
