# Machine Learning & Self-Supervised Learning (SSL)

To minimize reliance on expensive human-annotated datasets, this project heavily leverages Self-Supervised Learning (SSL). SSL allows foundational models to learn rich, domain-specific visual representations directly from massive unannotated NAIP satellite imagery, resulting in superior performance when fine-tuned on the specialized task of poultry house segmentation.

## 1. Self-Supervised Learning Paradigms

The project implements two state-of-the-art SSL pipelines capable of pretraining different architectures directly on remote sensing data.

### A. Masked Image Modeling (MIM)
**Target Architectures:** Vision Transformers (ViT) - Tiny, Small, Base, Large, Huge.
**Configuration:** `configs/ssl_grid_search.yaml`

MIM is heavily inspired by Masked Autoencoders (MAE). The objective is to mask a substantial portion of an input tile and force the ViT to reconstruct the missing pixels.

- **Adapter & Backbone:** `RemoteSensingMIMModel` uses a lightweight 1x1 `BandAdapter` to project multi-band imagery (e.g., 4-channel RGB-NIR) into the embedding dimension of the `RemoteSensingViT`.
- **Masking Strategy:** The grid search enforces a high **0.75 (75%) mask ratio**. Masking such a large portion of the image forces the model to learn holistic global structures (like the linear geometry of a farm) rather than relying on local pixel interpolation.
- **Initialization:** According to the configuration (`pretrained: true`), these ViTs can be initialized from official ImageNet weights before undergoing domain adaptation on NAIP data, or they can be trained entirely from scratch.

### B. Sparse Masked Modeling (SparK)
**Target Architectures:** Hierarchical CNNs (ResNet50, YOLOv11l, YOLOv26l).
**Configuration:** `configs/spark_grid_search.yaml`

While ViTs easily drop masked tokens, Convolutional Neural Networks (CNNs) require dense grids. If masks are filled with zeros, CNNs waste capacity learning the artificial geometric edges of the masks rather than the underlying imagery. SparK resolves this.

- **Sparse Convolutions:** SparK utilizes sparse convolutions during the encoding phase. Masked regions are mathematically ignored, allowing the CNN to extract features only from the unmasked visible patches.
- **Hierarchical Decoding:** A UNet-style decoder projects the multi-scale, sparse feature maps back onto a dense grid to reconstruct the original image.
- **Initialization:** The SparK grid search configuration dictates `pretrained: [false]`. This ensures the CNN backbones are **randomly initialized**, allowing them to build representations derived *purely* from aerial imagery without the inductive bias of ImageNet's dog/cat/car dataset.

## 2. Stateful Grid Search Orchestration

Both SSL pipelines employ robust grid search frameworks (`ssl_grid_search.py`, `spark_grid_search.py`).

- **Sequential Processing:** Because GPU VRAM is highly constrained, hyperparameter permutations are evaluated sequentially, avoiding the OOM crashes inherent to parallel multiprocessing.
- **Statefulness:** The pipeline tracks optimizer states, learning rates, and schedulers incrementally. If an external event interrupts the run, re-executing the CLI command resumes the epoch immediately where it stopped.
- **Hyperparameter Sweep:** Evaluates variables including learning rates (e.g., `0.0001` to `0.0004`), precisions (`fp32`, `fp16`, `bf16`), optimizers (`adamw`, `adan`), and stochastic depth (`drop_path_rate: 0.1`).

## 3. Model Selection & Fine-Tuning

Following pretraining, models are fine-tuned for binary semantic instance segmentation to detect poultry house roofs.

### Target Frameworks
1. **Ultralytics YOLO (Baseline & Production):** YOLO variants (v8, v11, v26) are prioritized for deployment due to their rapid inference speed and excellent hardware optimization.
2. **Transformers (RT-DETR, Mask2Former):** Supported for scenarios demanding extreme accuracy at the cost of slower inference throughput.
3. **SAM (Segment Anything Model):** Used strictly as a post-processing refinement tool (`sam.py`). YOLO/DETR provides a bounding box prompt to SAM, which dynamically generates an edge-snapped, ultra-precise polygon mask of the roof.

### Fine-Tuning Configurations
Controlled by `configs/downstream_finetune.yaml`:
- **Ablation Studies:** The pipeline natively compares models initialized from standard COCO weights (e.g., `yolo11l_pretrained`) against those utilizing the custom SparK pretraining (`yolo11l_spark`).
- **Memory Optimization:** Supports advanced memory saving techniques like freezing the foundational backbone and injecting **LoRA (Low-Rank Adaptation)** layers, allowing the model to adapt to segmentation tasks without catastrophic forgetting of the SSL pretraining.
- **Standard Parameters:** Operates heavily around 640x640 image sizes across 50 to 100 epochs using augmented PyTorch dataloaders.