# 07 - Dataset Description and Data Engineering

This document outlines the datasets used in the Poultry Farm Geospatial Detection Pipeline, detailing how data is sourced, created, formatted, and utilized across both self-supervised pretraining and downstream fine-tuning workflows.

---

## 1. NAIP Imagery for Self-Supervised Learning (SSL)

To train robust foundation models for geospatial features, the pipeline leverages the **National Agriculture Imagery Program (NAIP)** high-resolution aerial imagery. 

### What is in the Dataset?
- **Raw Data:** Multi-gigabyte, multi-band GeoTIFF files capturing large swathes of agricultural landscapes.
- **Processed Tiles:** Smaller, manageable image tiles (e.g., 256x256 or 512x512 pixels) extracted from the monolithic GeoTIFFs. These tiles serve as the unlabeled input corpus for self-supervised pretraining.

### How is it Created?
1. **Slicing (`poultry-geo slice`):** The `geotiff.py` module orchestrates an edge-snapped, window-based slicing process. It utilizes multi-processing to chunk massive GeoTIFFs into thousands of JPG/PNG or NPY tiles without triggering Out-Of-Memory (OOM) faults.
2. **Cataloging:** To optimize loading, a CSV catalog mapping file paths can be generated via `create_dataset_catalog`, preventing slow recursive directory scans during model training.

### How is it Used?
- **Custom `NAIPDataset`:** PyTorch `DataLoader` mechanics (`naip_data.py`) handle reading these nested tiles dynamically.
- **On-the-fly Augmentations:** The loader applies random center-crops, horizontal/vertical flips, brightness/contrast color jitter, and hue adjustments directly to the tensors to encourage robust representation learning.
- **Pretraining Pipeline:** The tiles are streamed to Vision Transformers (using MIM) and CNNs (using SparK) to learn the underlying structural and textural context of the rural landscape.

---

## 2. Poultry House Segmentation Dataset for Downstream Fine-Tuning

Once the base model has learned to interpret NAIP imagery, we fine-tune it specifically for instance segmentation of poultry houses.

### What is in the Dataset?
- **Input Images:** High-resolution rural aerial imagery targeting active farm units.
- **Annotations:** Polygon masks closely hugging the footprint of poultry houses. 
- **Format Structure:** The structured dataset follows the YOLO standard, splitting images and labels into `train`, `val`, and `test` directories. Text files accompanying each image contain normalized coordinate arrays for the object polygons.

### How is it Created?
1. **COCO Origin:** The annotations are originally sourced and labeled in the standard **COCO JSON format** (`_annotations.coco.json`).
2. **Automated Conversion Pipeline (`prepare_data.py`):**
   - The script parses the COCO JSON and extracts image metadata and segmentation polygons.
   - It filters out invalid polygons and normalizes the coordinates to a `[0, 1]` scale based on image dimensions.
   - It converts the COCO structure into the `YOLO` `txt` label format.
3. **Data Splitting:** By default, it automatically splits the raw training directory into an isolated training and validation set (e.g., 90/10 split) to track out-of-sample performance without data leakage.
4. **Metadata Generation:** It programmatically generates the required `data.yaml` defining the paths and class indices.

### How is it Used?
- **Supervised Fine-Tuning:** The prepared YOLO directory and `data.yaml` file are directly ingested by the downstream training scripts (e.g., `train_segmentation` scripts for YOLO11/YOLO26 architectures).
- **Loss Calculation:** The precise polygons enable the model to calculate boundary losses and mask Intersect-over-Union (IoU) during backpropagation, leading to highly accurate, non-rectangular building footprints.
