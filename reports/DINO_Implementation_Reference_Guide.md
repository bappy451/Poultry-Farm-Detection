# Complete DINO Implementation & Training Reference Guide
**A Reusable Architectural, Algorithmic, and Operational Manual for Self-Supervised Vision Transformer Pre-Training**

---

## Executive Summary
This document serves as an exhaustive reference guide for implementing, executing, and extending the **Domain-Adaptive DINO (self-distillation with no labels)** self-supervised learning (SSL) framework implemented in the `ag_foundation` codebase. It covers the mathematical formulation, PyTorch multi-crop architecture, composite loss function, WebDataset high-throughput streaming pipeline, and CLI execution anatomy—designed so that any engineer or researcher can port this implementation to a new project or domain.

---

## 1. Architectural Blueprint: Student–Teacher Self-Distillation

### 1.1 Core Design Concept
DINO learns self-supervised visual representations by matching the output distributions of two neural networks—a **Student** network ($\theta_s$) and a **Teacher** network ($\theta_t$)—across different spatial views of the same image.

```
       Input Image x
             │
     ┌───────┴────────┐
     ▼                ▼
Global Crops     All Crops (Global + Local)
(2 @ 224x224)    (2 @ 224x224 + 8 @ 96x96)
     │                │
     ▼                ▼
Teacher Network  Student Network
  ViT-Base/16      ViT-Base/16
(Stop-Gradient)  (Trainable)
     │                │
     ▼                ▼
Centering & Sinkhorn  │
     │                ▼
     └────────► DINO Cross-Entropy + iBOT + KoLeo Loss ──► Backprop onto Student (θ_s)
                      │
                      └──────────────────────────────────► EMA Update onto Teacher (θ_t)
```

### 1.2 Network Anatomy (`src/ag_foundation/models/dino.py`)
- **Backbone**: Vision Transformer (`ViT-Base/16`, 12 transformer blocks, hidden dimension $D = 768$, 12 attention heads, 86M parameters). Initialized from official `timm` / `dinov3` weights for Domain-Adaptive Pre-Training (DAPT) or randomly for scratch training.
- **DINO Projection Head (`DINOHead`)**:
  - A 3-layer Multi-Layer Perceptron (MLP) attached to the `[CLS]` token:
    1. Linear ($768 \to 2048$) + GELU + LayerNorm
    2. Linear ($2048 \to 2048$) + GELU + LayerNorm
    3. Linear Bottleneck ($2048 \to 256$, L2-normalized)
    4. Weight-Normalized Output Projection ($256 \to K = 65,536$ prototype logits)

---

## 2. Multi-Crop Augmentation Pipeline (`src/ag_foundation/training/dino_trainer.py`)

To enforce **local-to-global semantic consistency**, each input image is transformed into a set of $V$ cropped views:
1. **2 Global Views ($x_g$)**:
   - Resolution: $224 \times 224$ pixels.
   - Scale Range: `[0.32, 1.0]` of image area.
   - Augmentations: Random horizontal flip (50%), color jitter (brightness 0.4, contrast 0.4, saturation 0.2, hue 0.1), Gaussian blur (1.0 on 1st crop, 0.1 on 2nd), solarization (20% on 2nd crop), ImageNet normalization.
2. **8 Local Views ($x_l$)**:
   - Resolution: $96 \times 96$ pixels.
   - Scale Range: `[0.05, 0.32]` of image area.
   - Augmentations: Random horizontal flip (50%), color jitter, Gaussian blur (50%), ImageNet normalization.

**Feeding Rule**:
- The **Student** receives all $V = 10$ crops ($2\text{ global} + 8\text{ local}$).
- The **Teacher** receives **only** the 2 global crops ($x_g$).

---

## 3. Mathematical Formulation of the Composite Objective

The total loss minimized during training combines three distinct signals:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{dino}} \mathcal{L}_{\text{DINO}} + \lambda_{\text{ibot}} \mathcal{L}_{\text{iBOT}} + \lambda_{\text{koleo}} \mathcal{L}_{\text{KoLeo}}$$

In our reference configuration (`dino_dapt_full_scale.yaml`): $\lambda_{\text{dino}} = 1.0, \; \lambda_{\text{ibot}} = 1.0, \; \lambda_{\text{koleo}} = 0.1$.

### 3.1 DINO Cross-Entropy with Centering & Sharpening
For each student crop $x_i$ and teacher crop $x_j$ (where $x_j$ is a global crop and $i \neq j$):

$$\mathcal{L}_{\text{DINO}} = \frac{1}{|V_s| |V_t|} \sum_{i \in V_s} \sum_{j \in V_t, j \neq i} H\left( P_t(x_j), P_s(x_i) \right)$$

- **Student Probability Distribution**:
  $$P_s(x_i)^{(k)} = \frac{\exp\left(z_s(x_i)^{(k)} / \tau_s\right)}{\sum_{m=1}^K \exp\left(z_s(x_i)^{(m)} / \tau_s\right)}, \quad \tau_s = 0.1$$
- **Teacher Probability Distribution (with Centering Vector $c$)**:
  $$P_t(x_j)^{(k)} = \frac{\exp\left( (z_t(x_j)^{(k)} - c_k) / \tau_t \right)}{\sum_{m=1}^K \exp\left( (z_t(x_j)^{(m)} - c_m) / \tau_t \right)}$$
  - **Sharpening**: Teacher temperature $\tau_t$ warms up from `0.04` to `0.07` over the first 2 epochs, keeping distributions sharp to prevent uniform collapse.
  - **Centering Vector Update**: To prevent collapse into a single prototype dimension, the centering vector $c$ is updated asynchronously via an Exponential Moving Average (EMA) over batch activations:
    $$c \leftarrow m_c \, c + (1 - m_c) \frac{1}{B} \sum_{b=1}^B z_t(x_j)_b, \quad m_c = 0.9$$

### 3.2 iBOT Masked Patch Prediction Loss
- A blockwise boolean mask $M \in \{0, 1\}^{14 \times 14}$ masks out $\approx 30\%$ of the student's input patches ($x_i \odot M$).
- The student must predict the clean teacher patch embeddings across the masked token positions, enforcing localized token-to-token semantic understanding.

### 3.3 KoLeo Regularization Loss
- Applied to the L2-normalized `[CLS]` token embeddings within each batch to encourage a uniform distribution across the unit hypersphere:
  $$\mathcal{L}_{\text{KoLeo}} = -\frac{1}{n} \sum_{i=1}^n \log\left( d(x_i, x_{\text{NN}(i)}) \right)$$
  where $d(x_i, x_{\text{NN}(i)})$ is the Euclidean distance to the nearest neighbor within the batch.

---

## 4. Parameter Optimization & Momentum Schedules

### 4.1 Stop-Gradient Teacher EMA Update
The Teacher network $\theta_t$ is detached from the computational graph (`stop_grad`). Its weights are updated strictly as a moving average of the Student weights $\theta_s$:

$$\theta_t \leftarrow \lambda \theta_t + (1 - \lambda) \theta_s$$

- **Cosine Momentum Schedule**: $\lambda$ starts at $\lambda_{\text{start}} = 0.996$ and smoothly approaches $\lambda_{\text{end}} = 1.0$ at the end of training:
  $$\lambda(k) = 1.0 - (1.0 - \lambda_{\text{start}}) \cdot \frac{1 + \cos(\pi k / K_{\text{total}})}{2}$$

### 4.2 Learning Rate & Weight Decay Cosine Schedules
- **Linear Warmup**: For the first `warmup_epochs = 2`, the learning rate scales linearly from $0 \to \eta_{\text{max}}$.
- **Cosine LR Decay**: Decays from $\eta_{\text{max}} \to \eta_{\text{min}} = 1\times 10^{-6}$ across remaining epochs.
- **Cosine Weight Decay**: AdamW weight decay scales from `weight_decay = 0.04` to `weight_decay_end = 0.40`.

---

## 5. WebDataset High-Throughput I/O Pipeline (`src/ag_foundation/data/wds_loader.py`)
To train over massive image repositories without random-access inode overhead, images are sharded into uncompressed `.tar` archives (`dataset-00000.tar` …):
- **Streaming Pipeline**: Reads byte streams directly from `.tar` archives in sequential order.
- **Worker Configuration**: `num_workers=8`, `prefetch_factor=4`, `pin_memory=True`.
- **Effective Batch Size**: In `dino_dapt_full_scale.yaml`, `batch_size=32` $\times$ `gradient_accumulation_steps=32` = **1,024 images per effective weight update step**.
- **Mixed Precision**: Autocasting in `bf16` (`bfloat16`) to eliminate underflow/overflow scaling issues while halving VRAM usage.

---

## 6. Comprehensive Anatomy of the Execution Command

```powershell
conda activate ag-foundation; $env:PYTHONPATH = "src"; python -m ag_foundation train-dino --config configs/dino_dapt_full_scale.yaml --resume
```

### 6.1 Line-by-Line Breakdown
1. **`conda activate ag-foundation`**:
   - Activates the isolated conda environment containing PyTorch 2.x, CUDA kernels, torchvision, timm, WebDataset, and PyMuPDF/PGFPlots dependencies.
2. **`$env:PYTHONPATH = "src"`**:
   - Explicitly instructs the Python interpreter to look inside the `src/` directory for top-level packages. This allows `python -m ag_foundation` to import `src/ag_foundation` dynamically without requiring a global pip package installation (`pip install -e .`).
3. **`python -m ag_foundation train-dino`**:
   - Invokes the package root dispatcher (`src/ag_foundation/__main__.py` $\to$ `src/ag_foundation/cli.py`).
   - Parses the command verb `train-dino` and delegates execution to `src/ag_foundation/training/dino_runner.main()`.
4. **`--config configs/dino_dapt_full_scale.yaml`**:
   - Loads the hierarchical YAML configuration tree, overriding default CLI argparse values.
5. **`--resume`**:
   - Instructs `_resolve_resume_checkpoint()` in `dino_runner.py` to check for `last.pt` inside `runtime.output_dir` (`runs/dinov3_dapt_full_scale/last.pt`) or load the explicit `resume_from` path.

### 6.2 What Happens Under the Hood During `--resume`
When `--resume` is triggered, `load_training_checkpoint(last.pt)` performs the following state restorations:
1. **Model Weights**:
   - `model.student.load_state_dict(checkpoint["student"])`: Restores all 12 ViT blocks + student projection head.
   - `model.teacher.load_state_dict(checkpoint["teacher"])`: Restores the exact EMA teacher weights.
2. **Optimizer & Scaler Buffers**:
   - `optimizer.load_state_dict(checkpoint["optimizer"])`: Restores AdamW first/second moment buffers ($m_t, v_t$) and learning rate group histories.
   - `grad_scaler.load_state_dict(checkpoint["scaler"])`: Restores AMP loss scaling factors.
3. **Centering Vector ($c$)**:
   - Restores the exact prototype centering vector buffer, preventing sudden entropy spikes upon resumption.
4. **Schedule Continuity**:
   - Extracts `saved_epoch = int(checkpoint["epoch"])` (e.g., `75`).
   - The training loop initializes `range(saved_epoch, total_epochs)` (e.g., `range(75, 100)`).
   - Because `global_step` and schedule generators compute progress via `float(current_epoch) / float(total_epochs)`, learning rate decay, weight decay, and teacher momentum continue seamlessly along their cosine curves without any step discontinuity.

---

## 7. Reusability Guide: How to Adapt to a New Project

To reuse this DINO framework in another domain (e.g., medical pathology, satellite remote sensing, autonomous driving):

### Step 1: Prepare Your Sharded WebDataset
Package your domain RGB images into `.tar` shards of $\approx 1,000$ to $5,000$ images each using `tar -cf dataset-00000.tar image001.jpg image002.jpg ...`:
```
/path/to/my_project_shards/
  ├── dataset-00000.tar
  ├── dataset-00001.tar
  └── ...
```

### Step 2: Create a Custom YAML Configuration (`configs/my_domain_dino.yaml`)
```yaml
data:
  data_root: "/path/to/my_project_shards/dataset-*.tar"
  crop_size: 224
  channels: 3
  batch_size: 32
  num_workers: 8
  epoch_batches: 4000
  val_fraction: 0.02

runtime:
  output_dir: "runs/my_domain_dino"
  epochs: 100
  seed: 42
  precision: "bf16"
  gradient_accumulation_steps: 8
  warmup_epochs: 2
  resume: true

model:
  model_name: "B"               # ViT-Base (or 'S' for ViT-Small)
  pretrained_backbone: true     # Set true for DAPT from DINOv3; false for scratch
  pretrained_source: "dinov3"
  dino_out_dim: 65536
  head_nlayers: 3
  num_global_crops: 2
  num_local_crops: 8
  global_crop_scale: [0.32, 1.0]
  local_crop_scale: [0.05, 0.32]
  student_temperature: 0.1
  teacher_temperature: 0.04
  teacher_momentum_start: 0.996
  teacher_momentum_end: 1.0

optimizer:
  learning_rate: 0.0005
  weight_decay: 0.04
  weight_decay_end: 0.4
  min_learning_rate: 1.0e-06
```

### Step 3: Execute Training & Monitor Health
Launch training:
```powershell
conda activate ag-foundation; $env:PYTHONPATH = "src"; python -m ag_foundation train-dino --config configs/my_domain_dino.yaml --resume
```

**Diagnosing Training Health (`metrics.csv`)**:
- **`train_dino_loss`**: Should decrease steadily from $\approx 10.0 \to 6.0$. A sudden drop to $0.00$ indicates representation collapse.
- **`prototype_entropy`**: Should remain bounded between $7.0 \text{ and } 10.5$. If entropy drops below $3.0$, the model is suffering from dimensional collapse (increase centering momentum $m_c$ or decrease teacher temperature $\tau_t$).
- **`feature_variance`**: Should remain above $0.25$. A value near $0.00$ indicates uniform token collapse.

### Step 4: Extracting Backbone for Downstream Tasks
When fine-tuning on downstream tasks (classification, segmentation, detection), load the saved checkpoint (`best.pt` or `last.pt`) and extract only the student backbone weights by stripping the `student_backbone.` prefix:
```python
import torch
from ag_foundation.models.dino import create_vit_backbone

checkpoint = torch.load("runs/my_domain_dino/best.pt", map_location="cpu")
state_dict = checkpoint["student"]

# Strip prefix to isolate ViT-Base backbone from projection head
backbone_weights = {
    k.replace("student_backbone.", ""): v
    for k, v in state_dict.items()
    if k.startswith("student_backbone.")
}

backbone = create_vit_backbone(model_name="B", num_classes=0)
backbone.load_state_dict(backbone_weights, strict=True)
print("Successfully loaded self-supervised DINO backbone for downstream fine-tuning!")
```
