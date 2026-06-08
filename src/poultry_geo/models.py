from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from poultry_geo.ssl_models import LoRALinear


@dataclass(slots=True)
class TileSlice:
    """One edge-snapped tile extracted from a GeoTIFF."""

    image_array: np.ndarray
    transform: Any
    crs: Any
    dtype: str
    count: int
    position: tuple[int, int]
    offset_xy: tuple[int, int]
    # Maintained for compatibility with earlier code; it now matches the real tile size.
    valid_shape: tuple[int, int]
    source_path: Path | None = None


SUPPORTED_PRECISIONS = ("fp32", "fp16", "bf16")


def _validate_precision(precision: str) -> str:
    if precision not in SUPPORTED_PRECISIONS:
        supported = ", ".join(SUPPORTED_PRECISIONS[:-1]) + f", or {SUPPORTED_PRECISIONS[-1]}"
        raise ValueError(f"precision must be {supported}.")
    return precision


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (int(value), int(value))


def _build_resnet50_backbone(pretrained: bool = False) -> tuple[nn.Module, int]:
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    backbone = resnet50(weights=weights)
    features = nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4,
    )
    return features, 2048


def _build_timm_vit_backbone(
    backbone_name: str,
    *,
    image_size: int | tuple[int, int],
    pretrained: bool = False,
) -> tuple[nn.Module, int]:
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("ViT backbones require timm. Install the ML dependencies.") from exc

    backbone = timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="",
        img_size=_pair(image_size),
    )
    out_channels = int(getattr(backbone, "num_features", getattr(backbone, "embed_dim", 0)))
    if out_channels <= 0:
        raise ValueError(f"Could not infer ViT feature size for backbone '{backbone_name}'.")
    return backbone, out_channels


def _inject_lora_into_vit_backbone(
    backbone: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    wrapped = 0
    for module in backbone.modules():
        for attr_name in ("qkv", "proj"):
            child = getattr(module, attr_name, None)
            if child is None or isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear) or (
                hasattr(child, "in_features") and hasattr(child, "out_features")
            ):
                setattr(
                    module,
                    attr_name,
                    LoRALinear(
                        child,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        precision="fp32",
                    ),
                )
                wrapped += 1
    return wrapped


class BandAdapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 3) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")
        if out_channels <= 0:
            raise ValueError("out_channels must be a positive integer.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.proj = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("BandAdapter expects inputs with shape [B, C, H, W].")
        if inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"BandAdapter expected {self.in_channels} channels, received {inputs.shape[1]}."
            )
        return self.proj(inputs)


class _UpsampleDecoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 1) -> None:
        super().__init__()
        hidden_channels = max(hidden_channels, in_channels // 2, 16)
        mid_channels = max(hidden_channels // 2, 16)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
        )

    def forward(self, features: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        logits = self.layers(features)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class UnifiedRSModel(nn.Module):
    def __init__(
        self,
        *,
        n_channels: int,
        backbone_name: str,
        image_size: int | tuple[int, int],
        precision: str = "fp32",
        pretrained: bool = False,
        freeze_backbone: bool = False,
        use_lora: bool = False,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        if n_channels <= 0:
            raise ValueError("n_channels must be a positive integer.")
        self.image_size = _pair(image_size)
        self.requested_precision = _validate_precision(precision)
        self.resolved_precision = self.requested_precision
        self.backbone_name = backbone_name
        self.pretrained = bool(pretrained)
        self.freeze_backbone = bool(freeze_backbone)
        self.use_lora = bool(use_lora)
        self.is_vit_backbone = backbone_name.startswith("vit_")

        self.adapter = BandAdapter(n_channels, out_channels=3)
        if backbone_name == "resnet50":
            if self.use_lora:
                raise ValueError("LoRA is only supported for ViT backbones, not resnet50.")
            self.backbone, backbone_channels = _build_resnet50_backbone(pretrained=self.pretrained)
        elif self.is_vit_backbone:
            self.backbone, backbone_channels = _build_timm_vit_backbone(
                backbone_name,
                image_size=self.image_size,
                pretrained=self.pretrained,
            )
        else:
            raise ValueError(
                "Unsupported backbone. Use 'resnet50' or a timm ViT name beginning with 'vit_'."
            )

        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
        if self.use_lora:
            wrapped = _inject_lora_into_vit_backbone(
                self.backbone,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            if wrapped == 0:
                raise ValueError(f"Backbone '{backbone_name}' did not expose any ViT attention projections for LoRA.")

        self.decoder = _UpsampleDecoder(backbone_channels, hidden_channels=decoder_channels, out_channels=1)

    def _vit_patch_size(self) -> tuple[int, int]:
        patch_embed = getattr(self.backbone, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", None)
        if patch_size is None:
            raise ValueError(f"Backbone '{self.backbone_name}' does not expose patch_embed.patch_size.")
        return _pair(patch_size)

    def _reshape_vit_tokens(
        self,
        tokens: torch.Tensor,
        output_hw: tuple[int, int],
    ) -> torch.Tensor:
        if tokens.ndim == 4:
            return tokens
        if tokens.ndim != 3:
            raise ValueError("ViT forward_features must return [B, N, C] or [B, C, H, W].")

        patch_h, patch_w = self._vit_patch_size()
        grid_h = output_hw[0] // patch_h
        grid_w = output_hw[1] // patch_w
        expected_patch_tokens = grid_h * grid_w
        prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 0))
        if tokens.shape[1] - prefix_tokens == expected_patch_tokens:
            patch_tokens = tokens[:, prefix_tokens:, :]
        elif tokens.shape[1] == expected_patch_tokens:
            patch_tokens = tokens
        elif tokens.shape[1] - 1 == expected_patch_tokens:
            patch_tokens = tokens[:, 1:, :]
        elif tokens.shape[1] - 2 == expected_patch_tokens:
            patch_tokens = tokens[:, 2:, :]
        else:
            raise ValueError("Could not map ViT tokens back to a spatial grid for the requested image size.")

        return patch_tokens.transpose(1, 2).reshape(tokens.shape[0], patch_tokens.shape[-1], grid_h, grid_w)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("UnifiedRSModel expects inputs with shape [B, C, H, W].")
        adapted = self.adapter(inputs)
        if self.is_vit_backbone:
            tokens = self.backbone.forward_features(adapted)
            return self._reshape_vit_tokens(tokens, (inputs.shape[2], inputs.shape[3]))
        return self.backbone(adapted)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(inputs)
        logits = self.decoder(features, output_size=(int(inputs.shape[2]), int(inputs.shape[3])))
        return logits
