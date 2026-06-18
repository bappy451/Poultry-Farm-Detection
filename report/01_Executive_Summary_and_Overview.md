# Executive Summary and Project Overview

## Project Domain and Challenge
The **Poultry Farm Geospatial Detection Pipeline** operates at the intersection of deep learning, geospatial analysis (GIS), and agricultural environmental monitoring. 

Monitoring concentrated animal feeding operations, specifically poultry farms, is critical for understanding their impact on local watersheds and ecosystems. Traditional on-the-ground surveys are slow and difficult to scale. Raw aerial imagery, such as the National Agriculture Imagery Program (NAIP) dataset, provides massive amounts of data, but extracting meaningful insights from multi-gigabyte raster files requires specialized, robust, and scalable software architecture.

## The Solution
This project provides a production-grade, end-to-end geospatial machine learning pipeline. It transitions exploratory data science from brittle Jupyter Notebooks into a highly robust, modular Python package (`poultry_geo`) accompanied by a Command-Line Interface (CLI).

It completely automates the workflow:
1. Ingesting and preprocessing massive GeoTIFFs.
2. Running Self-Supervised Learning (SSL) to build domain-specific foundation models.
3. Fine-tuning models (YOLO, DETR) to detect poultry houses.
4. Streaming high-resolution imagery through sliding-window inference.
5. Converting raw pixel predictions into georeferenced GIS vectors (Shapefiles, KML).
6. Conducting county-level environmental analytics (e.g., roof runoff estimation).

## Pipeline Flow

1. **Slicing:** The pipeline ingests large multi-band GeoTIFFs and slices them into smaller, uniform tiles (e.g., 256x256 or 512x512) required by neural networks, utilizing multiprocessing to handle the extreme I/O loads.
2. **Foundation Model Pretraining (SSL):** Unannotated imagery is used to train base models (Vision Transformers or CNNs) using Masked Image Modeling or SparK.
3. **Segmentation Fine-Tuning:** The pretrained models are trained on specific, human-annotated datasets to delineate the precise roof boundaries of poultry houses.
4. **Streaming Inference:** Trained models process incoming unseen imagery using an overlapping sliding window.
5. **Post-Processing & Deduplication:** The raw overlapping detections undergo Centroid Suppression and Mask Non-Maximum Suppression (NMS) to eliminate duplicate predictions across tile boundaries.
6. **Farm Entity Clustering:** Individual, densely packed poultry houses are grouped into unified "Farm" entities using spatial density algorithms (DBSCAN).
7. **GIS Export:** Data is projected to local coordinate reference systems (e.g., EPSG:26916) and exported as Polygons, Bounding Boxes, or KMLs.
8. **County Analytics:** Generates statistical environmental screening data, including estimated rainfall runoff corrected against official NASS (National Agricultural Statistics Service) baselines.

## Repository Layout
- **`src/poultry_geo/`**: The structured Python library containing the core logic (`geotiff.py`, `inference.py`, `gis.py`, `ssl_models.py`, etc.).
- **`scripts/`**: OS-specific wrappers (`.sh`, `.cmd`, `.ps1`) that orchestrate safe environment execution and enforce append-only system logging.
- **`configs/`**: YAML/JSON schemas that drive the stateful grid searches and end-to-end pipelines.
- **`tests/`**: Over 140 unit tests guaranteeing the safety of coordinate transformations, tensor mathematics, and multiprocessing locks.
- **`notebook/` & `notebook_new/`**: The historical R&D records containing exploratory data analysis, experimental manuscript plots, and hard-coded workflow drafts.