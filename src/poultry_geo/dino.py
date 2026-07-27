from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from poultry_geo.models import BandAdapter
from poultry_geo.ssl_models import RemoteSensingViT, VIT_CONFIGS


try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm


class DINOHead(nn.Module):
    """3-layer MLP projection head for DINO with L2 bottleneck and weight norm.
    
    Structure:
      1. Linear(in_dim -> hidden_dim) + GELU + LayerNorm
      2. Linear(hidden_dim -> hidden_dim) + GELU + LayerNorm
      3. Linear(hidden_dim -> bottleneck_dim) + L2 normalization
      4. Weight-normalized Linear(bottleneck_dim -> out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
    ) -> None:
        super().__init__()
        nlayers = max(1, int(nlayers))
        layers: list[nn.Module] = []
        if nlayers == 1:
            self.mlp = nn.Identity()
            bottleneck_dim = in_dim
        else:
            current_dim = in_dim
            for _ in range(nlayers - 1):
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(hidden_dim))
                current_dim = hidden_dim
            layers.append(nn.Linear(current_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        if hasattr(self.last_layer, "weight_g"):
            self.last_layer.weight_g.data.fill_(1)
        elif hasattr(self.last_layer, "parametrizations") and hasattr(
            self.last_layer.parametrizations, "weight"
        ):
            self.last_layer.parametrizations.weight.original0.data.fill_(1)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class AdaptedBackbone(nn.Module):
    """Wraps an optional BandAdapter with a Vision Transformer or CNN backbone."""

    def __init__(self, adapter: nn.Module | None, backbone: nn.Module) -> None:
        super().__init__()
        self.adapter = adapter
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adapter is not None:
            x = self.adapter(x)
        features = self.backbone(x)
        if features.ndim == 3:
            # Global Average Pool across patch tokens [B, N, D] -> [B, D]
            return features.mean(dim=1)
        if features.ndim == 4:
            # Global Average Pool for CNN feature maps [B, C, H, W] -> [B, C]
            return F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        return features


class DINOMultiCropWrapper(nn.Module):
    """Wraps Student and Teacher backbones and DINO projection heads.
    
    The Teacher weights are detached from the computational graph and updated
    asynchronously via Exponential Moving Average (EMA) of the Student weights.
    """

    def __init__(
        self,
        student_backbone: nn.Module,
        teacher_backbone: nn.Module,
        student_head: nn.Module,
        teacher_head: nn.Module,
    ) -> None:
        super().__init__()
        self.student_backbone = student_backbone
        self.teacher_backbone = teacher_backbone
        self.student_head = student_head
        self.teacher_head = teacher_head

        # Detach teacher weights and copy student weights
        self._init_teacher()

    def _init_teacher(self) -> None:
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())

    @torch.no_grad()
    def update_teacher_ema(self, momentum: float) -> None:
        """Update Teacher parameters with cosine momentum EMA."""
        for param_s, param_t in zip(
            self.student_backbone.parameters(), self.teacher_backbone.parameters()
        ):
            param_t.data.mul_(momentum).add_((1.0 - momentum) * param_s.detach().data)
        for param_s, param_t in zip(
            self.student_head.parameters(), self.teacher_head.parameters()
        ):
            param_t.data.mul_(momentum).add_((1.0 - momentum) * param_s.detach().data)

    def forward_student(self, crops: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Process all student crops (global + local) and return prototype logits."""
        logits: list[torch.Tensor] = []
        for crop in crops:
            features = self.student_backbone(crop)
            out = self.student_head(features)
            logits.append(out)
        return logits

    @torch.no_grad()
    def forward_teacher(self, global_crops: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Process teacher global crops and return prototype logits."""
        logits: list[torch.Tensor] = []
        for crop in global_crops:
            features = self.teacher_backbone(crop)
            out = self.teacher_head(features)
            logits.append(out)
        return logits


def cosine_momentum_schedule(
    step: int,
    total_steps: int,
    momentum_start: float = 0.996,
    momentum_end: float = 1.0,
) -> float:
    """Compute cosine momentum lambda for step k out of total_steps."""
    if total_steps <= 1:
        return momentum_end
    progress = min(1.0, max(0.0, float(step) / float(total_steps)))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return momentum_end - (momentum_end - momentum_start) * cosine_decay


def build_dino_model(
    model_name: str = "B",
    channels: int = 3,
    crop_size: int = 224,
    patch_size: int = 16,
    dino_out_dim: int = 65536,
    head_nlayers: int = 3,
    precision: str = "fp32",
    pretrained_backbone: bool = True,
) -> DINOMultiCropWrapper:
    """Factory function to build Student and Teacher networks wrapped in DINOMultiCropWrapper."""
    if model_name not in VIT_CONFIGS:
        supported = ", ".join(sorted(VIT_CONFIGS.keys()))
        raise ValueError(f"Unknown ViT model_name '{model_name}'. Must be one of: {supported}")

    cfg = VIT_CONFIGS[model_name]
    embed_dim = cfg["embed_dim"]
    depth = cfg["depth"]
    num_heads = cfg["num_heads"]

    def _create_branch() -> tuple[AdaptedBackbone, DINOHead]:
        adapter = None
        if channels != 3:
            adapter = BandAdapter(in_channels=channels, out_channels=3)

        vit = RemoteSensingViT(
            image_size=(crop_size, crop_size),
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            precision=precision,
        )
        if pretrained_backbone:
            vit.load_pretrained_weights()

        backbone = AdaptedBackbone(adapter=adapter, backbone=vit)
        head = DINOHead(
            in_dim=embed_dim,
            out_dim=dino_out_dim,
            hidden_dim=2048,
            bottleneck_dim=256,
            nlayers=head_nlayers,
        )
        return backbone, head

    student_backbone, student_head = _create_branch()
    teacher_backbone, teacher_head = _create_branch()

    return DINOMultiCropWrapper(
        student_backbone=student_backbone,
        teacher_backbone=teacher_backbone,
        student_head=student_head,
        teacher_head=teacher_head,
    )
