"""
DINO / DINOv3 Cosine Similarity Map & Unsupervised Feature Visualization.

Extracts patch token embeddings from a trained or pretrained DINO Vision Transformer
and computes:
  1. Query-Point Cosine Similarity Heatmaps
  2. Structural Feature Salience / Attention Maps
  3. Unsupervised 3-Component PCA Feature Segmentation Maps
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")  # Use thread-safe non-interactive backend to prevent Tk/Tcl crashes
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from poultry_geo.dino import build_dino_model


def load_image_tensor(image_path: str | Path, target_size: int = 512) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load an image from disk, resize if needed, and return both the PyTorch
    [1, 3, H, W] normalized float32 tensor and the [H, W, 3] uint8 RGB array.
    """
    path_str = str(image_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Image not found at {path_str}")

    img_bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image at {path_str}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    if h != target_size or w != target_size:
        img_rgb = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor, img_rgb


def compute_cosine_similarity_maps(
    image_tensor: torch.Tensor,
    backbone: torch.nn.Module,
    patch_size: int = 16,
    query_x: int | None = None,
    query_y: int | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Extract DINO patch token embeddings and compute cosine similarity maps.

    Args:
        image_tensor: Input tensor [1, 3, H, W] in range [0, 1].
        backbone: The ViT backbone network (e.g. RemoteSensingViT).
        patch_size: ViT patch size (default 16).
        query_x, query_y: Pixel coordinate for query patch. If None, defaults to image center.
        device: Torch execution device.

    Returns:
        Dictionary containing similarity maps and metadata.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = backbone.to(device)
    backbone.eval()

    img = image_tensor.to(device)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    _, _, h, w = img.shape
    grid_h, grid_w = h // patch_size, w // patch_size

    with torch.no_grad():
        # Forward features returns patch token embeddings [B, N_patches, D]
        # or tuple depending on backbone output
        if hasattr(backbone, "forward_features"):
            features = backbone.forward_features(img)
        elif hasattr(backbone, "backbone") and hasattr(backbone.backbone, "forward_features"):
            features = backbone.backbone.forward_features(img)
        else:
            features = backbone(img)
        if isinstance(features, tuple):
            features = features[0]
        if features.dim() == 4:
            # If [B, D, H_patch, W_patch], convert to [B, N_patches, D]
            features = features.flatten(2).transpose(1, 2)

    # Ensure features are in float32 for SVD, F.interpolate, and NumPy compatibility
    # (prevents 'Got unsupported ScalarType BFloat16' when models are trained in bf16/fp16)
    features = features.float()

    # Normalize patch embeddings to unit sphere for cosine similarity
    features_norm = F.normalize(features[0], p=2, dim=-1)  # [N_patches, D]
    num_patches, feat_dim = features_norm.shape

    if num_patches != grid_h * grid_w:
        # Adjust grid size if backbone included CLS/dist tokens
        spatial_tokens = features_norm[- (grid_h * grid_w):]
    else:
        spatial_tokens = features_norm

    # Determine query patch index
    if query_x is None or query_y is None:
        qy_patch, qx_patch = grid_h // 2, grid_w // 2
    else:
        qy_patch = min(max(0, int(query_y // patch_size)), grid_h - 1)
        qx_patch = min(max(0, int(query_x // patch_size)), grid_w - 1)

    query_idx = qy_patch * grid_w + qx_patch
    query_vec = spatial_tokens[query_idx]  # [D]

    # 1. Query Cosine Similarity Map
    cosine_sim = torch.mv(spatial_tokens, query_vec)  # [N]
    cosine_sim_grid = cosine_sim.reshape(1, 1, grid_h, grid_w)
    cosine_sim_upsampled = F.interpolate(
        cosine_sim_grid, size=(h, w), mode="bilinear", align_corners=False
    ).squeeze().cpu().float().numpy()

    # 2. Mean Cosine Similarity / Structural Salience Map (similarity against global average feature)
    mean_vec = F.normalize(spatial_tokens.mean(dim=0), p=2, dim=-1)
    salience_sim = torch.mv(spatial_tokens, mean_vec)
    salience_grid = salience_sim.reshape(1, 1, grid_h, grid_w)
    salience_upsampled = F.interpolate(
        salience_grid, size=(h, w), mode="bilinear", align_corners=False
    ).squeeze().cpu().float().numpy()

    # 3. PCA Feature Segmentation Map (Top 3 Principal Components mapped to RGB)
    # Perform PCA on spatial tokens [N, D]
    tokens_centered = spatial_tokens - spatial_tokens.mean(dim=0, keepdim=True)
    try:
        _, _, V = torch.svd_lowrank(tokens_centered, q=3)
        pca_features = torch.matmul(tokens_centered, V[:, :3])  # [N, 3]
        # Normalize each component to [0, 1] for RGB visualization
        for c in range(3):
            c_min = pca_features[:, c].min()
            c_max = pca_features[:, c].max()
            pca_features[:, c] = (pca_features[:, c] - c_min) / max(1e-6, float(c_max - c_min))
        pca_grid = pca_features.reshape(1, grid_h, grid_w, 3).permute(0, 3, 1, 2)  # [1, 3, grid_h, grid_w]
        pca_upsampled = F.interpolate(
            pca_grid, size=(h, w), mode="bilinear", align_corners=False
        ).squeeze().permute(1, 2, 0).cpu().float().numpy()
    except Exception:
        pca_upsampled = np.zeros((h, w, 3), dtype=np.float32)

    return {
        "query_sim": cosine_sim_upsampled,
        "salience_sim": salience_upsampled,
        "pca_rgb": (pca_upsampled * 255.0).astype(np.uint8),
        "query_coord": (qx_patch * patch_size + patch_size // 2, qy_patch * patch_size + patch_size // 2),
        "patch_size": patch_size,
    }


def save_dino_similarity_figure(
    image_rgb: np.ndarray,
    sim_results: dict[str, Any],
    output_path: str | Path,
    title: str = "DINOv3 Cosine Similarity & Feature Analysis",
) -> Path:
    """
    Render and save a 4-panel comparison figure showing:
      1. Original Image with Query Crosshair
      2. Query-Point Cosine Similarity Map
      3. Structural Salience / Attention Map
      4. DINO Unsupervised PCA Feature Segmentation
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    qx, qy = sim_results["query_coord"]

    # 1. Original Image
    axes[0].imshow(image_rgb)
    axes[0].plot(qx, qy, "r+", markersize=14, markeredgewidth=2, label="Query Patch")
    axes[0].plot(qx, qy, "ro", markersize=6, fillstyle="none", markeredgewidth=1.5)
    axes[0].set_title("1. Original Tile & Query Point", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Query Cosine Similarity Map
    im2 = axes[1].imshow(sim_results["query_sim"], cmap="turbo", vmin=-0.2, vmax=1.0)
    axes[1].plot(qx, qy, "w+", markersize=12, markeredgewidth=2)
    axes[1].set_title("2. Query Cosine Similarity Map", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Structural Salience Map
    im3 = axes[2].imshow(sim_results["salience_sim"], cmap="viridis", vmin=-0.2, vmax=1.0)
    axes[2].set_title("3. Mean Cosine Salience / Attention", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Unsupervised PCA Segmentation
    axes[3].imshow(sim_results["pca_rgb"])
    axes[3].set_title("4. DINO PCA Feature Segmentation", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return out_path


def generate_dino_visualizations(
    image_paths: list[str | Path],
    model_name: str = "B",
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path = "runs/dino_visualizations",
    patch_size: int = 16,
    query_x: int | None = None,
    query_y: int | None = None,
    device: str | None = None,
) -> list[Path]:
    """
    Generate DINO cosine similarity visualizations for a list of image paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading DINO model ({model_name}) on {torch_device} for similarity visualization...")

    model = build_dino_model(
        model_name=model_name,
        out_dim=65536,
        crop_size=512,
        patch_size=patch_size,
        pretrained_backbone=True,
    )

    if checkpoint_path and os.path.exists(str(checkpoint_path)):
        print(f"Loading weights from checkpoint: {checkpoint_path}")
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        if "teacher" in ckpt:
            model.load_state_dict(ckpt["teacher"], strict=False)
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

    backbone = model.teacher_backbone.to(torch_device)
    backbone.eval()

    saved_figures = []
    for i, img_path in enumerate(image_paths):
        path_obj = Path(img_path)
        print(f"[{i+1}/{len(image_paths)}] Processing {path_obj.name}...")

        tensor, img_rgb = load_image_tensor(path_obj, target_size=512)
        sim_results = compute_cosine_similarity_maps(
            image_tensor=tensor,
            backbone=backbone,
            patch_size=patch_size,
            query_x=query_x,
            query_y=query_y,
            device=torch_device,
        )

        out_name = out_dir / f"{path_obj.stem}_dino_similarity.png"
        saved_path = save_dino_similarity_figure(
            image_rgb=img_rgb,
            sim_results=sim_results,
            output_path=out_name,
            title=f"DINO Cosine Similarity Analysis: {path_obj.name}",
        )
        saved_figures.append(saved_path)

    return saved_figures


def generate_dino_visualizations_from_backbone(
    backbone: torch.nn.Module,
    image_paths: list[str | Path],
    output_dir: str | Path,
    patch_size: int = 16,
    device: torch.device | None = None,
) -> list[Path]:
    """
    Generate DINO cosine similarity visualizations directly from an in-memory backbone model.
    Ideal for running after each training epoch without reloading weights from disk.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.eval()

    saved_figures = []
    for i, img_path in enumerate(image_paths):
        path_obj = Path(img_path)
        try:
            tensor, img_rgb = load_image_tensor(path_obj, target_size=512)
            sim_results = compute_cosine_similarity_maps(
                image_tensor=tensor,
                backbone=backbone,
                patch_size=patch_size,
                device=torch_device,
            )
            out_name = out_dir / f"{path_obj.stem}_dino_similarity.png"
            saved_path = save_dino_similarity_figure(
                image_rgb=img_rgb,
                sim_results=sim_results,
                output_path=out_name,
                title=f"DINO Cosine Similarity Analysis: {path_obj.name}",
            )
            saved_figures.append(saved_path)
        except Exception as e:
            print(f"Warning: Failed to visualize sample {path_obj.name}: {e}")

    return saved_figures
