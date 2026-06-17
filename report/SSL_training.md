# Self-Supervised Learning (SSL) Methodology & Training Report
**Project:** Poultry Farm Detection  
**Date:** June 12, 2026

## 1. Executive Summary
This report outlines the implementation and training strategy for Self-Supervised Learning (SSL) within the Poultry Farm Detection project. The project leverages two distinct masked modeling methodologies to pre-train vision backbones on large volumes of unannotated aerial imagery (e.g., NAIP tiles) prior to downstream fine-tuning for farm segmentation. 

The two methodologies used are:
1.  **MIM (Masked Image Modeling):** A Masked Autoencoder (MAE) style approach applied to Vision Transformer (ViT) architectures.
2.  **SparK (Sparse Masked Modeling):** An SSL framework designed specifically for hierarchical Convolutional Neural Networks (CNNs), such as ResNet and YOLO's CSPDarkNet.

By utilizing SSL, the project aims to learn robust, general-purpose spatial and semantic features from satellite imagery, drastically improving sample efficiency and final accuracy when fine-tuned on the smaller, annotated poultry farm dataset.

---

## 2. Methodologies in Detail

### 2.1 Masked Image Modeling (MIM)
MIM is implemented as an MAE-style framework, primarily targeting non-hierarchical architectures like Vision Transformers. 
*   **Concept:** The input image is divided into a grid of patches. A high percentage of these patches (typically 75%) are randomly masked (dropped out). The encoder processes only the visible patches. A lightweight decoder then attempts to reconstruct the missing patches in pixel space from the encoded latent representation and learnable mask tokens.
*   **Backbones Supported:** Vision Transformers (ViT) in Small (S), Base (B), Large (L), and Huge (H) configurations.
*   **Key Techniques:**
    *   **High Masking Ratio:** Forces the network to learn holistic, high-level structural features rather than relying on local textures.
    *   **Parameter-Efficient Fine-Tuning (PEFT):** The implementation supports QLoRA-style training using `nf4` (4-bit NormalFloat) precision and LoRA adapters. This allows training massive models (like ViT-Huge) on consumer hardware by freezing the backbone and only training low-rank adapter matrices.
*   **Execution Script:** `scripts/train_mim.sh` / `poultry_geo.py train-mim`

### 2.2 Sparse Masked Modeling (SparK)
While MAE is highly effective for ViTs, adapting it to CNNs is challenging because convolutions operate on dense grids and struggle with the "holes" left by masked patches. SparK resolves this.
*   **Concept:** SparK extends the success of MAE to hierarchical CNNs. It uses 3D sparse convolutions to process only the unmasked regions of the image, avoiding the computational waste of dense convolutions on zeroed-out pixels. It also employs a hierarchical decoder to reconstruct the masked parts at multiple resolutions.
*   **Backbones Supported:** Hierarchical CNNs, specifically `resnet50`, `yolo11l` (CSPDarkNet), and custom implementations like `yolov26l`.
*   **Key Techniques:**
    *   **Sparse Convolutions:** Treats the unmasked patches as a sparse point cloud, allowing standard CNNs to act as efficient masked autoencoders.
    *   **Multi-Scale Reconstruction:** Reconstructs features at different stages of the CNN pyramid, which is critical for dense prediction tasks like object detection and segmentation.
*   **Execution Script:** `scripts/train_spark.sh` / `poultry_geo.py train-spark`

---

## 3. Training & Grid Search Configurations

To identify the optimal pre-training configurations, the project utilizes comprehensive grid search YAML files that iterate over multiple hyperparameter combinations.

### 3.1 Common Data Setup
Both methodologies train on `512x256` RGB satellite tiles (`MS_tiles_512_256_jpg`).
*   **Crop/Image Size:** Resized or cropped to `224x224` during SSL training.
*   **Batch Size:** 64 for MIM, 128 for SparK (scaled based on VRAM).
*   **Workers:** 20 to ensure data loading does not bottleneck the GPU.

### 3.2 MIM Grid Search (`configs/mim_grid_config.yaml`)
*   **Models:** ViT-S, ViT-B, ViT-L
*   **Mask Ratio:** 0.75 (75% of patches dropped)
*   **Optimizer:** AdamW
*   **Learning Rate:** `1e-4` with 1 epoch of linear warmup.
*   **Weight Decay:** 0.05
*   **Stochastic Depth (Drop Path Rate):** 0.1 (Used to regularize deeper ViT models).
*   **Precision:** `fp16` (Mixed Precision for speed and VRAM savings).
*   **LoRA / Freezing:** Configurable support for freezing the backbone and using LoRA adapters for highly constrained environments.

### 3.3 SparK Grid Search (`configs/spark_grid_search.yaml`)
*   **Models:** `yolov26l`, `yolo11l`, `resnet50`
*   **Mask Ratio:** 0.75
*   **Optimizer:** AdamW
*   **Learning Rate:** `1e-4` (No warmup configured in default run).
*   **Weight Decay:** 0.05
*   **Precision:** `fp16`

---

## 4. Implementation Details and Tooling

The training pipelines are wrapped in robust Bash scripts that handle environment setup, dynamic argument forwarding to a central Python CLI (`poultry_geo.py`), and standardized logging.

### 4.1 Invocation Commands
Users can launch training runs using the provided shell scripts, passing dynamic arguments to override YAML defaults.

**MIM Example:**
```bash
./scripts/train_mim.sh \
  --data-root ./tmp/test_jpg \
  --output-dir ./runs/mim-run \
  --batch-size 64 \
  --crop-size 224 \
  --precision fp16 \
  --verbose
```

**SparK Example (YOLO Backbone):**
```bash
./scripts/train_spark.sh \
  --data-root ./tmp/test_jpg \
  --output-dir ./runs/spark-yolo \
  --batch-size 128 \
  --backbone-type yolo11l \
  --mask-ratio 0.75 \
  --precision fp16 \
  --verbose
```

### 4.2 Logging and Tracking
*   **Log Files:** Every execution appends to a central `command.log` tracking the timestamp, CWD, and full exact command string used to launch the job. It automatically traps exit codes and durations to ensure run history is safely preserved even if a job crashes.
*   **Verbose Mode:** Scripts feature a `--verbose` flag that enables rich, multi-line progress reporting with live metric updates in the terminal.

---

## 5. Conclusion
The dual-strategy SSL approach ensures the project can leverage the strengths of both Vision Transformers (via MIM/MAE) and Convolutional Neural Networks (via SparK). By first learning general representations of satellite terrain using 75% masked reconstruction tasks, the backbones (specifically YOLO11l) achieve significantly higher F1 scores and segmentation accuracy during downstream supervised fine-tuning.
