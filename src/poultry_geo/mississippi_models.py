from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from poultry_geo.ssl_models import (
    BandAdapter,
    RemoteSensingViT,
    build_nf4_quantization_config,
    inject_lora_into_vit,
)


_MODEL_SIZE_MAP = {
    "vit_s": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "vit_b": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "vit_l": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
}

_TIMM_DINOV3_MODELS = {
    "vit_s": "vit_small_patch16_dinov3",
    "vit_b": "vit_base_patch16_dinov3",
    "vit_l": "vit_large_patch16_dinov3",
}


class MississippiFMEncoder(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 256,
        in_channels: int = 3,
        patch_size: int = 16,
        model_size: str = "vit_b",
        projection_dim: int = 256,
        precision: str = "fp32",
        pretrained_anchor: bool = False,
        backbone_impl: str = "custom",
        backbone_name: str | None = None,
    ) -> None:
        super().__init__()
        if model_size not in _MODEL_SIZE_MAP:
            raise ValueError(f"model_size must be one of {sorted(_MODEL_SIZE_MAP)}.")
        if backbone_impl not in {"custom", "timm_dinov3"}:
            raise ValueError("backbone_impl must be 'custom' or 'timm_dinov3'.")
        config = _MODEL_SIZE_MAP[model_size]
        self.model_size = model_size
        self.embed_dim = int(config["embed_dim"])
        self.patch_size = int(patch_size)
        self.image_size = int(image_size)
        self.projection_dim = int(projection_dim)
        self.requested_precision = precision
        self.backbone_impl = backbone_impl
        self.backbone_name = backbone_name or _TIMM_DINOV3_MODELS[model_size]
        self.use_lora = False
        self.nf4_enabled = False
        self.quantization_config: Any | None = None

        self.adapter = BandAdapter(
            in_channels=in_channels,
            out_channels=3,
            precision=precision,
            bypass_rgb=(in_channels == 3),
        )
        self.backbone = self._build_backbone(
            image_size=image_size,
            patch_size=patch_size,
            config=config,
            precision=precision,
            pretrained=pretrained_anchor,
        )
        self.resolved_precision = getattr(self.backbone, "resolved_precision", precision)

        self.global_head = _ProjectionHead(self.embed_dim, projection_dim, precision=self.resolved_precision)
        self.patch_head = _ProjectionHead(self.embed_dim, projection_dim, precision=self.resolved_precision)
        self.overlap_head = _ProjectionHead(self.embed_dim, projection_dim, precision=self.resolved_precision)

        self.anchor_adapter = deepcopy(self.adapter)
        self.anchor_backbone = deepcopy(self.backbone)
        self.anchor_adapter.requires_grad_(False)
        self.anchor_backbone.requires_grad_(False)
        self.anchor_adapter.eval()
        self.anchor_backbone.eval()

    def forward_views(
        self,
        *,
        global_views: torch.Tensor,
        local_views: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        global_patches = self._encode_patches(global_views)
        global_cls = global_patches.mean(dim=1)
        outputs = {
            "global_cls": global_cls,
            "global_patches": global_patches,
            "global_distillation": F.normalize(self.global_head(global_cls), dim=-1),
            "patch_predictions": self.patch_head(global_patches),
            "overlap_projection": F.normalize(self.overlap_head(global_patches), dim=-1),
        }
        if local_views is not None:
            local_patches = self._encode_patches(local_views)
            local_cls = local_patches.mean(dim=1)
            outputs.update(
                {
                    "local_cls": local_cls,
                    "local_patches": local_patches,
                    "local_distillation": F.normalize(self.global_head(local_cls), dim=-1),
                    "local_patch_predictions": self.patch_head(local_patches),
                    "local_overlap_projection": F.normalize(self.overlap_head(local_patches), dim=-1),
                }
            )
        return outputs

    def anchor_patch_embeddings(self, views: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            adapted = self.anchor_adapter(views)
            return self.anchor_backbone.forward_features(adapted)

    def gram_anchor_loss(self, views: torch.Tensor) -> torch.Tensor:
        current = self._encode_patches(views)
        anchor = self.anchor_patch_embeddings(views)
        return compute_gram_anchoring_loss(current, anchor)

    def enable_lora(self, *, rank: int = 4, alpha: float = 8.0, dropout: float = 0.0) -> None:
        inject_lora_into_vit(
            self.backbone,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            freeze_backbone=False,
        )
        self.use_lora = True

    def disable_lora(self) -> None:
        for name, parameter in self.backbone.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(False)
        self.use_lora = False

    def enable_nf4(self) -> Any:
        self.quantization_config = build_nf4_quantization_config()
        self.nf4_enabled = True
        return self.quantization_config

    def disable_nf4(self) -> None:
        self.quantization_config = None
        self.nf4_enabled = False

    def _encode_patches(self, views: torch.Tensor) -> torch.Tensor:
        adapted = self.adapter(views)
        if self.backbone_impl == "timm_dinov3":
            features = self.backbone.forward_features(adapted.float())
            patch_count = (int(adapted.shape[2]) // self.patch_size) * (int(adapted.shape[3]) // self.patch_size)
            return features[:, -patch_count:, :].to(dtype=_resolved_output_dtype(self.resolved_precision))
        return self.backbone.forward_features(adapted)

    def _build_backbone(
        self,
        *,
        image_size: int,
        patch_size: int,
        config: dict[str, int],
        precision: str,
        pretrained: bool,
    ) -> nn.Module:
        if self.backbone_impl == "timm_dinov3":
            try:
                import timm
            except ImportError as exc:
                raise RuntimeError("timm is required for backbone_impl='timm_dinov3'.") from exc
            backbone = timm.create_model(
                self.backbone_name,
                pretrained=pretrained,
                img_size=image_size,
                dynamic_img_size=True,
            )
            backbone.resolved_precision = precision
            return backbone

        backbone = RemoteSensingViT(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=self.embed_dim,
            depth=int(config["depth"]),
            num_heads=int(config["num_heads"]),
            precision=precision,
        )
        if pretrained:
            backbone.load_pretrained_weights()
        return backbone


class _ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, precision: str) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )
        self.precision = precision

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.net(inputs.float())
        if self.precision == "fp16":
            return outputs.to(dtype=torch.float16)
        if self.precision in {"bf16", "nf4"}:
            return outputs.to(dtype=torch.bfloat16)
        return outputs


def compute_overlap_alignment_loss(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    patch_indices_a: list[torch.Tensor] | torch.Tensor,
    patch_indices_b: list[torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    if isinstance(patch_indices_a, torch.Tensor):
        patch_indices_a = [row for row in patch_indices_a]
    if isinstance(patch_indices_b, torch.Tensor):
        patch_indices_b = [row for row in patch_indices_b]

    per_sample_losses: list[torch.Tensor] = []
    for sample_index, (indices_a, indices_b) in enumerate(zip(patch_indices_a, patch_indices_b, strict=True)):
        if indices_a.numel() == 0 or indices_b.numel() == 0:
            continue
        count = min(indices_a.numel(), indices_b.numel())
        selected_a = F.normalize(features_a[sample_index, indices_a[:count]], dim=-1)
        selected_b = F.normalize(features_b[sample_index, indices_b[:count]], dim=-1)
        per_sample_losses.append(1.0 - (selected_a * selected_b).sum(dim=-1).mean())
    if not per_sample_losses:
        return features_a.new_tensor(0.0)
    return torch.stack(per_sample_losses).mean()


def compute_gram_anchoring_loss(
    current_patch_embeddings: torch.Tensor,
    anchor_patch_embeddings: torch.Tensor,
) -> torch.Tensor:
    current_gram = _gram_matrix(current_patch_embeddings)
    anchor_gram = _gram_matrix(anchor_patch_embeddings)
    return F.mse_loss(current_gram, anchor_gram)


def _gram_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(embeddings.float(), dim=-1)
    return normalized.transpose(1, 2) @ normalized / max(1, normalized.shape[1])


def _resolved_output_dtype(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision in {"bf16", "nf4"}:
        return torch.bfloat16
    return torch.float32
