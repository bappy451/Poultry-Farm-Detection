"""Shared, validated checkpoint helpers for the project DINO workflow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


DINO_CHECKPOINT_FORMAT = "poultry_geo_dino/v1"


def unwrap_compiled_module(module: nn.Module) -> nn.Module:
    """Return the original module when ``torch.compile`` wrapped it."""
    return getattr(module, "_orig_mod", module)


def extract_dino_backbone_state(
    checkpoint: Mapping[str, Any] | Mapping[str, torch.Tensor], *, role: str = "teacher"
) -> dict[str, torch.Tensor]:
    """Extract a project DINO backbone state dict in bare ViT key space."""
    if role not in {"student", "teacher"}:
        raise ValueError("role must be 'student' or 'teacher'.")
    key = f"{role}_backbone"
    state: Any = checkpoint.get(key) if isinstance(checkpoint, Mapping) else None
    if state is None and isinstance(checkpoint, Mapping):
        state = checkpoint
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint does not contain a '{key}' DINO backbone state dict.")

    normalized: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
            continue
        name = raw_key.removeprefix("_orig_mod.").removeprefix("module.")
        name = name.removeprefix("backbone.")
        if not name.startswith("adapter."):
            normalized[name] = value
    if not normalized:
        raise ValueError("No tensor parameters were found in the DINO backbone checkpoint.")
    return normalized


def load_state_dict_with_coverage(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    minimum_parameter_coverage: float = 0.90,
    context: str = "checkpoint",
) -> float:
    """Load shape-compatible tensors and reject silent, low-coverage loads."""
    target = module.state_dict()
    compatible = {
        key: value for key, value in state_dict.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    total_parameters = sum(value.numel() for value in target.values())
    matched_parameters = sum(target[key].numel() for key in compatible)
    coverage = matched_parameters / max(1, total_parameters)
    if coverage < minimum_parameter_coverage:
        raise ValueError(
            f"Refusing to load {context}: only {coverage:.1%} of target parameters match "
            f"({len(compatible)}/{len(target)} tensors); require at least "
            f"{minimum_parameter_coverage:.1%}."
        )
    module.load_state_dict(compatible, strict=False)
    return coverage
