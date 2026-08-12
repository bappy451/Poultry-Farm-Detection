from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import torch

from poultry_geo.dino import DINOMultiCropWrapper
from poultry_geo.dino_loss import CompositeDINOLoss
from poultry_geo.ssl_models import PRECISION_DTYPES


def train_one_step(
    model: DINOMultiCropWrapper,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    loss_fn: CompositeDINOLoss,
    batch: dict[str, Any],
    device: torch.device,
    precision: str = "fp32",
    momentum: float = 0.996,
    epoch: int = 0,
) -> dict[str, float]:
    """Execute one optimization step for DINO self-distillation."""
    optimizer.zero_grad(set_to_none=True)

    global_crops = [crop.to(device, non_blocking=True) for crop in batch["global_crops"]]
    local_crops = [crop.to(device, non_blocking=True) for crop in batch["local_crops"]]
    all_crops = global_crops + local_crops

    dtype = PRECISION_DTYPES.get(precision, torch.float32)
    use_amp = precision != "fp32"

    with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=dtype):
        student_logits = model.forward_student(all_crops)
        teacher_logits = model.forward_teacher(global_crops)

        # Optional features from the first student global crop for KoLeo loss
        with torch.no_grad():
            pass

        # We compute KoLeo using the first global crop representation
        student_features = None
        if loss_fn.lambda_koleo > 0:
            student_features = model.student_backbone(global_crops[0])

        loss_dict = loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_features=student_features,
            epoch=epoch,
        )
        loss = loss_dict["loss"]

    if not use_amp or scaler is None or not scaler.is_enabled():
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Asynchronously update Teacher EMA
    model.update_teacher_ema(momentum=momentum)

    return {
        "loss": float(loss_dict["loss"].item()),
        "dino_loss": float(loss_dict["dino_loss"].item()),
        "koleo_loss": float(loss_dict["koleo_loss"].item()),
        "prototype_entropy": float(loss_dict["prototype_entropy"]),
    }


def save_dino_checkpoint(
    model: DINOMultiCropWrapper,
    loss_fn: CompositeDINOLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    config: Any,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save full DINO training checkpoint and extracted backbone for downstream tasks.
    
    Returns:
      tuple of (unified_checkpoint_path, backbone_checkpoint_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    unified_path = output_path / f"dino_unified_ep{epoch}.pt"
    backbone_path = output_path / f"backbone_dino_ep{epoch}.pt"

    config_dict = vars(config) if hasattr(config, "__dict__") else config

    checkpoint = {
        "epoch": epoch,
        "student_backbone": model.student_backbone.state_dict(),
        "teacher_backbone": model.teacher_backbone.state_dict(),
        "student_head": model.student_head.state_dict(),
        "teacher_head": model.teacher_head.state_dict(),
        "centering_vector": loss_fn.dino_loss.center.cpu(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": config_dict,
    }
    torch.save(checkpoint, unified_path)

    # Save extracted Student backbone weights for downstream fine-tuning
    torch.save(model.student_backbone.state_dict(), backbone_path)

    # Also maintain latest convenience symlink/copy as last.pt
    last_path = output_path / "last.pt"
    torch.save(checkpoint, last_path)

    # Automatically prune old intermediate checkpoints to prevent C: drive exhaustion (~350+ GB savings)
    # Preserve milestone epochs (multiples of 50) and the last 3 epochs
    for old_ep in range(0, epoch - 2):
        if old_ep % 50 != 0:
            for prefix in ("dino_unified_ep", "backbone_dino_ep"):
                old_file = output_path / f"{prefix}{old_ep}.pt"
                if old_file.exists():
                    try:
                        old_file.unlink()
                    except OSError:
                        pass

    return unified_path, backbone_path


def load_dino_checkpoint(
    path: str | Path,
    model: DINOMultiCropWrapper,
    loss_fn: CompositeDINOLoss | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load DINO checkpoint state dicts into model, loss_fn, optimizer, and scaler."""
    checkpoint = torch.load(path, map_location=device)

    model.student_backbone.load_state_dict(checkpoint["student_backbone"])
    model.teacher_backbone.load_state_dict(checkpoint["teacher_backbone"])
    model.student_head.load_state_dict(checkpoint["student_head"])
    model.teacher_head.load_state_dict(checkpoint["teacher_head"])

    if loss_fn is not None and "centering_vector" in checkpoint:
        center_tensor = checkpoint["centering_vector"].to(loss_fn.dino_loss.center.device)
        loss_fn.dino_loss.center.copy_(center_tensor)

    if optimizer is not None and "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scaler is not None and "scaler" in checkpoint and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    return checkpoint
