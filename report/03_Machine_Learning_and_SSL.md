# 03 - Machine Learning and Self-Supervised Learning (SSL)

To minimize the project's reliance on expensive, human-annotated datasets, `poultry_geo` leverages Self-Supervised Learning (SSL). By training foundation models on vast quantities of unannotated NAIP satellite imagery, the models learn rich, domain-specific spatial features—resulting in superior performance, faster convergence, and reduced overfitting when later fine-tuned on the task of poultry house segmentation.

The project implements two state-of-the-art SSL paradigms, targeting both Vision Transformers (ViTs) and Convolutional Neural Networks (CNNs).

## 1. Masked Image Modeling (MIM)
**Target Architectures:** Vision Transformers (ViT-Tiny, Small, Base, Large)
**Configuration Module:** `configs/mim_grid_config.yaml`
**Implementation:** `poultry_geo.ssl_models.RemoteSensingMIMModel`

Masked Image Modeling is a Masked Autoencoder (MAE) approach suitable for non-hierarchical architectures.

### Mechanism
The input tile is divided into a grid of patches (e.g., 16x16 pixels). During training, a large portion of these patches are randomly dropped out. A lightweight decoder is then tasked with reconstructing the missing pixels based on the latent representation of the visible patches.

### Design Parameters
- **Masking Ratio:** Set to a strict **0.75 (75%)**. By obscuring most of the image, the ViT cannot rely on local pixel interpolation. It is forced to learn holistic, high-level geometric structures (e.g., the long, linear nature of farm buildings, or the specific texture of agricultural fields).
- **Initialization & Weights:** In the grid search config (`pretrained: true`), the ViTs are initialized with official ImageNet weights before undergoing domain adaptation on the NAIP data. This allows the transformer to utilize foundational edge-detection capabilities while learning the specific spectral properties of satellite imagery.
- **Hyperparameters:** Driven by AdamW optimizer, Learning Rate of `1e-4`, weight decay of `0.05`, and utilizing `fp16` precision to maximize batch sizes (e.g., `batch_size: 64`). Deeper models leverage a `drop_path_rate: 0.1` for stochastic depth regularization.
- **Config Compatibility:** `pretrained` is supported in the SSL grid config but now defaults to `[false]` when absent, preserving older minimal configs while keeping explicit pretrained experiments available.

## 2. Sparse Masked Modeling (SparK)
**Target Architectures:** Hierarchical CNNs (ResNet50, YOLO11l, YOLOv26l)
**Configuration Module:** `configs/spark_grid_search.yaml`
**Implementation:** `scripts/spark_grid_train.sh`

While MIM works seamlessly for Transformers (which process unmasked tokens efficiently as sets), Convolutional Neural Networks operate on dense grids. If masks are represented as zero-filled pixels, CNNs waste processing power learning the artificial, sharp edges of the mask boundaries rather than the underlying terrain.

### Mechanism
SparK (Sparse Masked Modeling) extends the MAE paradigm to CNNs using 3D sparse convolutions. The network mathematically skips masked regions, treating the unmasked patches as a sparse point cloud. This enables standard CNNs (like YOLO's CSPDarknet) to act as efficient masked autoencoders. A UNet-style hierarchical decoder then reconstructs the masked parts at multiple spatial resolutions.

### Design Parameters
- **Masking Ratio:** Matches MIM at **0.75 (75%)**.
- **Initialization & Weights:** Crucially, the `spark_grid_search.yaml` explicitly enforces `pretrained: false`. This means the CNN backbones are **randomly initialized**. They are forced to build feature representations derived *purely* from the NAIP aerial imagery, completely unburdened by the inductive bias of ImageNet's dog/cat/car dataset.
- **Hyperparameters:** AdamW optimizer, Learning Rate of `4e-4`, weight decay of `0.05`, and `fp16` precision (scaling up to `batch_size: 128` depending on hardware).
- **Optional Sparse Backend:** `spconv` is used when installed. The code also supports dense/test fallbacks and late `spconv` availability so tests and CPU-only development environments remain usable.

## 3. Stateful Grid Search Orchestration
Self-Supervised Learning requires extensive hyperparameter tuning. However, executing multiple large models in parallel via multiprocessing rapidly exhausts GPU VRAM, causing OOM kernel crashes.

### Sequential & Stateful Execution
The project employs robust grid search frameworks (`train_ssl_grid` and `spark_grid_train`).
- **Sequential Looping:** Permutations (e.g., testing `fp32` vs `bf16`, or testing `freeze_backbone` variations) are queued and processed strictly sequentially.
- **State Preservation:** Because long-running campaigns are vulnerable to OS updates or power loss, the loops are rigorously stateful. The optimizer states, epoch counts, and random seed states are continuously checkpointed. If the script is interrupted, executing the exact same CLI command will instantly resume the training at the exact epoch it failed, preventing the loss of hundreds of hours of GPU compute.

Current wrappers:

```bash
./scripts/train_ssl_grid.sh --training-name mim-search --config configs/mim_grid_config.yaml --verbose
./scripts/spark_grid_train.sh --training-name spark-search --config configs/spark_grid_search.yaml --verbose
```

Before running, set `data.image_dir` in each config to a tile folder produced by the slicer, such as `./tmp/tiles_256_png`.

## 4. Parameter-Efficient Fine-Tuning (PEFT)
To train massive models (like ViT-Large) on consumer-grade hardware, the SSL framework natively supports Low-Rank Adaptation (LoRA) and NF4 (4-bit NormalFloat) quantization. 
By activating `--use-lora` and `--freeze-backbone`, the core weights of the model are frozen, and only tiny adapter matrices are trained. This radically reduces VRAM consumption and prevents catastrophic forgetting of the SSL pretraining when shifting to downstream tasks.
