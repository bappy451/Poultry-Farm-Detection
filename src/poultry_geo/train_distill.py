from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from poultry_geo.dino_checkpoint import extract_dino_backbone_state, load_state_dict_with_coverage
from poultry_geo.ssl_models import RemoteSensingViT


_VIT_SPECS = {
    "S": {"embed_dim": 384, "num_heads": 6, "depth": 12},
    "B": {"embed_dim": 768, "num_heads": 12, "depth": 12},
    "L": {"embed_dim": 1024, "num_heads": 16, "depth": 24},
}


@dataclass
class DistillConfig:
    precision: str = "bf16"
    student_model: str = "yolo11l.yaml"
    lr: float = 1e-3
    batch_size: int = 32
    vit_name: str = "B"
    image_size: int = 224

    def __post_init__(self) -> None:
        self.vit_name = self.vit_name.upper()
        if self.vit_name not in _VIT_SPECS:
            raise ValueError(f"Unsupported ViT name: {self.vit_name}. Expected one of {sorted(_VIT_SPECS)}.")
        if self.image_size <= 0 or self.image_size % 32:
            raise ValueError("image_size must be a positive multiple of 32 for YOLO P3/P4/P5 distillation.")
        if not self.student_model.lower().endswith((".yaml", ".yml")):
            raise ValueError("student_model must be an Ultralytics detection YAML, not a pretrained .pt file.")

    @property
    def vit_embed_dim(self) -> int:
        return _VIT_SPECS[self.vit_name]["embed_dim"]


def normalize_dino_rgb(images: torch.Tensor) -> torch.Tensor:
    """Apply the ImageNet normalization used by this project's DINO data pipeline."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("DINO-to-YOLO distillation requires RGB tensors [B, 3, H, W].")
    mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (images - mean) / std


class OfficialDinoTeacher(nn.Module):
    """Expose dense patch tokens from the official legacy DINO implementation."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.model.get_intermediate_layers(images, n=1)[-1]
        if tokens.ndim != 3 or tokens.shape[1] < 2:
            raise RuntimeError("Official DINO teacher did not return CLS plus patch tokens.")
        return tokens[:, 1:, :]

class OfficialDinoV2Teacher(nn.Module):
    """Expose normalized dense patch tokens from an official DINOv2 ViT-B/14."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model.forward_features(images)
        if not isinstance(output, dict) or "x_norm_patchtokens" not in output:
            raise RuntimeError("Official DINOv2 teacher did not return normalized patch tokens.")
        tokens = output["x_norm_patchtokens"]
        if tokens.ndim != 3 or tokens.shape[-1] != 768:
            raise RuntimeError(f"Expected DINOv2 ViT-B/14 patch tokens [B, N, 768], got {tuple(tokens.shape)}.")
        return tokens


def load_teacher(checkpoint_path: str, config: DistillConfig, device: torch.device) -> nn.Module:
    """Load a verified project DINO teacher or an explicit official DINO teacher."""
    print(f"Loading DINO Teacher ViT-{config.vit_name} from {checkpoint_path}...")
    if checkpoint_path.lower() == "official":
        if config.vit_name not in {"S", "B"}:
            raise ValueError("The legacy facebookresearch/dino hub exposes official ViT-S and ViT-B teachers only.")
        model_name = f"dino_vit{config.vit_name.lower()}16"
        print(f"Downloading official TorchHub weights: {model_name}")
        teacher: nn.Module = OfficialDinoTeacher(torch.hub.load("facebookresearch/dino:main", model_name))
    elif checkpoint_path.lower() == "dinov2":
        if config.vit_name != "B":
            raise ValueError("The official DINOv2 teacher currently supported here is ViT-B/14; use --vit-name B.")
        print("Downloading/loading official DINOv2 ViT-B/14 TorchHub weights...")
        teacher = OfficialDinoV2Teacher(torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True, trust_repo=True))
    else:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"DINO teacher checkpoint was not found: {path}")
        spec = _VIT_SPECS[config.vit_name]
        teacher = RemoteSensingViT(
            image_size=config.image_size,
            patch_size=16,
            embed_dim=spec["embed_dim"],
            num_heads=spec["num_heads"],
            depth=spec["depth"],
            precision=config.precision,
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = extract_dino_backbone_state(checkpoint, role="teacher")
        coverage = load_state_dict_with_coverage(teacher, state, context=f"DINO teacher checkpoint '{path}'")
        print(f"Loaded project DINO teacher with {coverage:.1%} parameter coverage.")

    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def _autocast_dtype(precision: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]


def train_one_step(student, teacher, criterion, optimizer, scaler, images, config: DistillConfig) -> float:
    """Train on raw YOLO input; normalize a separate view only for the DINO teacher."""
    if images.shape[-2:] != (config.image_size, config.image_size):
        raise ValueError(
            f"Expected {config.image_size}x{config.image_size} crops, got {tuple(images.shape[-2:])}."
        )
    student.train()
    optimizer.zero_grad(set_to_none=True)
    teacher_images = normalize_dino_rgb(images)
    autocast_enabled = config.precision != "fp32" and images.device.type in {"cuda", "cpu"}
    with torch.autocast(device_type=images.device.type, enabled=autocast_enabled, dtype=_autocast_dtype(config.precision)):
        with torch.no_grad():
            teacher_tokens = teacher.forward_features(teacher_images)
        student_tokens = student(images)  # Ultralytics training/inference expects RGB in [0, 1].
        loss = criterion(student_tokens, teacher_tokens)
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate_distillation(student, teacher, criterion, images, config: DistillConfig) -> float:
    was_training = student.training
    student.eval()
    teacher_images = normalize_dino_rgb(images)
    autocast_enabled = config.precision != "fp32" and images.device.type in {"cuda", "cpu"}
    with torch.autocast(device_type=images.device.type, enabled=autocast_enabled, dtype=_autocast_dtype(config.precision)):
        loss = criterion(student(images), teacher.forward_features(teacher_images))
    student.train(was_training)
    return float(loss)
