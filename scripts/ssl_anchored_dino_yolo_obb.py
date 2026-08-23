"""Core components for anchored DINOv2-to-YOLO11 OBB pretraining.

The student starts from the same Ultralytics OBB checkpoint used by the
supervised baseline.  Only modules 0..9 (the YOLO backbone) are optimized on
unlabeled imagery.  A frozen DINOv2 teacher supplies dense, self-supervised
patch features; a relative weight-anchor penalty limits catastrophic drift
from the original aerial OBB representation.  Disposable projectors map the
three YOLO feature scales to the DINO embedding dimension.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from poultry_geo.distill_loss import FeatureDistillationLoss
from poultry_geo.distill_models import BACKBONE_MODULE_COUNT, validate_backbone_compatibility


ARTIFACT_FORMAT = "poultry_geo_anchored_dinov2_yolo_obb/v1"
FEATURE_INDICES = {4: "p3", 6: "p4", 9: "p5"}


class AnchoredYOLOOBBStudent(nn.Module):
    """Exact pretrained YOLO11 OBB backbone plus disposable projectors."""

    def __init__(self, init_model: str | Path, teacher_dim: int = 768) -> None:
        super().__init__()
        from ultralytics import YOLO

        detector = YOLO(str(init_model))
        modules = detector.model.model
        if len(modules) < BACKBONE_MODULE_COUNT:
            raise ValueError(f"Detector exposes only {len(modules)} modules.")
        self.backbone = nn.ModuleList(list(modules[:BACKBONE_MODULE_COUNT]))
        self.init_model = str(Path(init_model).expanduser().resolve())
        # Ultralytics checkpoints are loaded inference-ready and may leave all
        # parameters frozen. SSL must explicitly re-enable backbone gradients;
        # otherwise only the disposable projectors and BN buffers can change.
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)
        trainable = sum(parameter.numel() for parameter in self.backbone.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.backbone.parameters())
        if trainable != total or total == 0:
            raise RuntimeError(
                f"Anchored SSL requires every backbone parameter trainable; got {trainable}/{total}."
            )

        was_training = self.training
        self.eval()
        with torch.no_grad():
            features = self.forward_backbone(torch.zeros(1, 3, 224, 224))
        self.train(was_training)
        expected = {"p3": 28, "p4": 14, "p5": 7}
        observed = {name: int(value.shape[-1]) for name, value in features.items()}
        if observed != expected:
            raise ValueError(
                "Anchored SSL currently supports the YOLO11 OBB graph only; "
                f"expected feature grids {expected}, found {observed}."
            )
        self.projectors = nn.ModuleDict(
            {
                name: nn.Conv2d(int(value.shape[1]), int(teacher_dim), kernel_size=1)
                for name, value in features.items()
            }
        )

    def forward_backbone(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected RGB BCHW input, got {tuple(images.shape)}.")
        result: dict[str, torch.Tensor] = {}
        current = images
        for index, module in enumerate(self.backbone):
            current = module(current)
            if index in FEATURE_INDICES:
                result[FEATURE_INDICES[index]] = current
        if set(result) != set(FEATURE_INDICES.values()):
            raise RuntimeError("YOLO backbone did not expose P3/P4/P5 features.")
        return result

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_backbone(images)
        return {
            name: self.projectors[name](value).flatten(2).transpose(1, 2).contiguous()
            for name, value in features.items()
        }

    def export_backbone_state(self) -> dict[str, torch.Tensor]:
        exported: dict[str, torch.Tensor] = {}
        for raw_key, value in self.backbone.state_dict().items():
            module_index, separator, parameter_name = raw_key.partition(".")
            if not separator or not module_index.isdigit():
                raise RuntimeError(f"Unexpected backbone state key: {raw_key}")
            exported[f"model.{module_index}.{parameter_name}"] = value.detach().cpu().clone()
        return exported


class TimmDinoV2Teacher(nn.Module):
    """Frozen timm DINOv2 model exposing dense patch tokens and normalization."""

    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
        *,
        image_size: int = 448,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        import timm

        self.model_name = str(model_name)
        self.image_size = int(image_size)
        if self.image_size <= 0 or self.image_size % 14:
            raise ValueError("DINOv2 teacher image_size must be a positive multiple of patch size 14.")
        # Supplying img_size at construction lets timm resize the pretrained
        # positional embedding once, rather than failing on non-518px inputs.
        self.model = timm.create_model(
            self.model_name, pretrained=pretrained, num_classes=0, img_size=self.image_size
        )
        self.embed_dim = int(getattr(self.model, "num_features", 0))
        if self.embed_dim <= 0:
            raise RuntimeError(f"Could not determine teacher dimension for {self.model_name}.")
        cfg = getattr(self.model, "pretrained_cfg", {}) or {}
        mean = tuple(float(value) for value in cfg.get("mean", (0.485, 0.456, 0.406)))
        std = tuple(float(value) for value in cfg.get("std", (0.229, 0.224, 0.225)))
        self.register_buffer("pixel_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        if tuple(images.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(
                f"DINOv2 teacher expects {self.image_size}x{self.image_size}, got {tuple(images.shape[-2:])}."
            )
        normalized = (images - self.pixel_mean.to(images)) / self.pixel_std.to(images)
        output = self.model.forward_features(normalized)
        if isinstance(output, dict):
            for key in ("x_norm_patchtokens", "x_prenorm"):
                if key in output:
                    output = output[key]
                    break
        if not torch.is_tensor(output) or output.ndim != 3:
            raise RuntimeError("DINOv2 teacher did not return BND patch features.")
        prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))
        patch_tokens = output[:, prefix_tokens:, :] if output.shape[1] > prefix_tokens else output
        side = int(patch_tokens.shape[1] ** 0.5)
        if side * side != int(patch_tokens.shape[1]):
            raise RuntimeError(
                f"Teacher patch-token count must be square after prefix removal; got {patch_tokens.shape[1]}."
            )
        return patch_tokens


def capture_parameter_anchors(student: AnchoredYOLOOBBStudent) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in student.backbone.named_parameters()}


def relative_anchor_loss(
    student: AnchoredYOLOOBBStudent,
    anchors: Mapping[str, torch.Tensor],
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Dimensionless mean relative drift from the pretrained backbone."""
    terms: list[torch.Tensor] = []
    for name, parameter in student.backbone.named_parameters():
        if name not in anchors:
            raise KeyError(f"Missing anchor for backbone parameter {name}.")
        anchor = anchors[name].to(parameter)
        scale = anchor.float().square().mean().clamp_min(float(epsilon))
        terms.append((parameter.float() - anchor.float()).square().mean() / scale)
    if not terms:
        raise RuntimeError("YOLO backbone exposes no trainable parameters.")
    return torch.stack(terms).mean()


def anchored_distillation_loss(
    student_tokens: Mapping[str, torch.Tensor],
    teacher_tokens: torch.Tensor,
    student: AnchoredYOLOOBBStudent,
    anchors: Mapping[str, torch.Tensor],
    *,
    anchor_weight: float,
    scale_weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if anchor_weight < 0:
        raise ValueError("anchor_weight cannot be negative.")
    feature_loss = FeatureDistillationLoss(scale_weights)(student_tokens, teacher_tokens)
    anchor_loss = relative_anchor_loss(student, anchors)
    total = feature_loss + float(anchor_weight) * anchor_loss
    return total, feature_loss, anchor_loss


def make_artifact(
    student: AnchoredYOLOOBBStudent,
    *,
    init_model: str | Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    from ultralytics import YOLO

    init_path = Path(init_model).expanduser().resolve()
    backbone = student.export_backbone_state()
    target = YOLO(str(init_path)).model.state_dict()
    validate_backbone_compatibility(target, backbone)
    return {
        "format": ARTIFACT_FORMAT,
        "init_model": str(init_path),
        "backbone_state_dict": backbone,
        "metadata": dict(metadata),
    }


def load_artifact(path: str | Path) -> dict[str, Any]:
    artifact = torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or artifact.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"Expected {ARTIFACT_FORMAT} artifact: {path}")
    if not isinstance(artifact.get("backbone_state_dict"), dict):
        raise ValueError("Anchored SSL artifact has no backbone_state_dict.")
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Anchored SSL artifact has no verified training metadata.")
    if int(metadata.get("trainable_backbone_parameters", 0)) <= 0:
        raise ValueError("Anchored SSL artifact was produced without a trainable backbone and is invalid.")
    if float(metadata.get("backbone_relative_anchor_loss", 0.0)) <= 0.0:
        raise ValueError("Anchored SSL artifact shows zero backbone parameter drift and is invalid.")
    return artifact


def load_artifact_into_detector(detector: Any, artifact: Mapping[str, Any]) -> int:
    compatible = validate_backbone_compatibility(
        detector.model.state_dict(), artifact["backbone_state_dict"]
    )
    _, unexpected = detector.model.load_state_dict(compatible, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading anchored SSL backbone: {unexpected}")
    return len(compatible)
