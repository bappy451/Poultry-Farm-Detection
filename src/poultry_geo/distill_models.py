"""YOLO student and checkpoint contracts for DINO-to-YOLO distillation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


BACKBONE_MODULE_COUNT = 10
DISTILLATION_ARTIFACT_FORMAT = "poultry_geo_dino_yolo_distill/v2"


def _require_detection_yaml(model_cfg: str) -> None:
    if not str(model_cfg).lower().endswith((".yaml", ".yml")):
        raise ValueError(
            "The distillation student must be created from an Ultralytics detection YAML, "
            "for example 'yolo11l.yaml'. A .pt model would leak pretraining and may not "
            "have the same graph as the downstream detector."
        )


def _backbone_state_keys(state: Mapping[str, torch.Tensor]) -> set[str]:
    prefixes = tuple(f"model.{index}." for index in range(BACKBONE_MODULE_COUNT))
    return {key for key in state if key.startswith(prefixes)}


class YOLODenseEncoder(nn.Module):
    """The first ten modules of an Ultralytics *detection* model.

    The module instances are taken directly from a randomly initialized detection
    graph. This makes the state dict one-to-one compatible with the detector used
    downstream and avoids silently importing classifier or ImageNet weights.
    """

    def __init__(self, model_cfg: str = "yolo11l.yaml") -> None:
        super().__init__()
        _require_detection_yaml(model_cfg)
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency error
            raise ImportError("ultralytics is required to use YOLO backbones.") from exc

        detector = YOLO(model_cfg)
        modules = detector.model.model
        if len(modules) < BACKBONE_MODULE_COUNT:
            raise ValueError(f"Detection model '{model_cfg}' has only {len(modules)} modules.")
        self.model_cfg = str(model_cfg)
        self.backbone = nn.ModuleList(list(modules[:BACKBONE_MODULE_COUNT]))
        self.feature_indices = (4, 6, 9)
        was_training = self.training
        self.eval()  # Do not contaminate freshly initialized BatchNorm statistics with the probe tensor.
        with torch.no_grad():
            features = self.forward(torch.zeros(1, 3, 224, 224))
        self.train(was_training)
        expected_sizes = {"p3": 28, "p4": 14, "p5": 7}
        if {name: value.shape[-1] for name, value in features.items()} != expected_sizes:
            raise ValueError(
                f"'{model_cfg}' does not expose the expected YOLO11 P3/P4/P5 feature hierarchy. "
                "This distillation implementation currently supports the YOLO11 detection graph."
            )
        self.feature_channels = {name: int(value.shape[1]) for name, value in features.items()}

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("YOLO distillation expects unnormalized RGB images [B, 3, H, W].")
        outputs: dict[int, torch.Tensor] = {}
        current = images
        for index, module in enumerate(self.backbone):
            current = module(current)
            if index in self.feature_indices:
                outputs[index] = current
        return {"p3": outputs[4], "p4": outputs[6], "p5": outputs[9]}


class DistillationProjector(nn.Module):
    """Maps one dense CNN feature map to dense teacher-token space."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.proj(features)
        return projected.flatten(2).transpose(1, 2).contiguous()


class DistillationStudent(nn.Module):
    """Random YOLO detection backbone plus disposable multi-scale projectors."""

    def __init__(self, model_cfg: str = "yolo11l.yaml", vit_embed_dim: int = 768) -> None:
        super().__init__()
        self.encoder = YOLODenseEncoder(model_cfg=model_cfg)
        self.model_cfg = self.encoder.model_cfg
        self.projectors = nn.ModuleDict({
            name: DistillationProjector(channels, vit_embed_dim)
            for name, channels in self.encoder.feature_channels.items()
        })

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(images)
        return {name: self.projectors[name](value) for name, value in features.items()}


def export_ultralytics_backbone_state(student_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Export every detection-backbone tensor and reject incomplete mappings."""
    exported: dict[str, torch.Tensor] = {}
    legacy_prefixes = {
        "encoder.stem.": "model.0.", "encoder.stage1.0.": "model.1.",
        "encoder.stage1.1.": "model.2.", "encoder.stage2.0.": "model.3.",
        "encoder.stage2.1.": "model.4.", "encoder.stage3.0.": "model.5.",
        "encoder.stage3.1.": "model.6.", "encoder.stage4.0.": "model.7.",
        "encoder.stage4.1.": "model.8.", "encoder.stage4.2.": "model.9.",
    }
    for raw_key, value in student_state.items():
        key = raw_key.removeprefix("_orig_mod.").removeprefix("module.")
        if key.startswith("encoder.backbone."):
            suffix = key.removeprefix("encoder.backbone.")
            module_index, separator, parameter_name = suffix.partition(".")
            if separator and module_index.isdigit() and int(module_index) < BACKBONE_MODULE_COUNT:
                exported[f"model.{module_index}.{parameter_name}"] = value.detach().cpu().clone()
                continue
        # Legacy checkpoints remain readable, but the strict target compatibility
        # check will reject them if their classifier graph differs from detection.
        for source_prefix, target_prefix in legacy_prefixes.items():
            if key.startswith(source_prefix):
                exported[target_prefix + key[len(source_prefix):]] = value.detach().cpu().clone()
                break
    if not exported:
        raise ValueError("Distillation checkpoint contains no detection-backbone parameters to export.")
    return exported


def validate_backbone_compatibility(
    target_state: Mapping[str, torch.Tensor], backbone_state: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Require an exact tensor-for-tensor backbone match before downstream use."""
    expected = _backbone_state_keys(target_state)
    supplied = _backbone_state_keys(backbone_state)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    mismatched = sorted(
        key for key in expected & supplied if tuple(target_state[key].shape) != tuple(backbone_state[key].shape)
    )
    if missing or extra or mismatched:
        raise ValueError(
            "Distilled backbone is incompatible with the requested YOLO detector: "
            f"missing={len(missing)}, extra={len(extra)}, shape_mismatches={len(mismatched)}. "
            "Use the identical --student-model/--model-cfg and Ultralytics version used for distillation."
        )
    return {key: backbone_state[key] for key in sorted(expected)}


def make_distillation_artifact(
    student_state: Mapping[str, torch.Tensor], *, model_cfg: str, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Create the portable artifact consumed by downstream YOLO fine-tuning."""
    _require_detection_yaml(model_cfg)
    return {
        "format": DISTILLATION_ARTIFACT_FORMAT,
        "student_model": str(model_cfg),
        "backbone_state_dict": export_ultralytics_backbone_state(student_state),
        "metadata": dict(metadata),
    }
