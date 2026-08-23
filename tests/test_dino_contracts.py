from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from poultry_geo.dino_checkpoint import extract_dino_backbone_state, load_state_dict_with_coverage
from poultry_geo.distill_models import export_ultralytics_backbone_state
from poultry_geo.train_dino_ssl import apply_dino_optimizer_schedule
from poultry_geo.train_distill import normalize_dino_rgb


def test_project_dino_checkpoint_extracts_inner_backbone_and_rejects_adapter() -> None:
    checkpoint = {
        "teacher_backbone": {
            "backbone.weight": torch.ones(2, 3),
            "adapter.proj.weight": torch.ones(3, 3, 1, 1),
        }
    }
    state = extract_dino_backbone_state(checkpoint, role="teacher")
    assert set(state) == {"weight"}


def test_coverage_loader_rejects_silent_low_coverage_load() -> None:
    module = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    with pytest.raises(ValueError, match="only"):
        load_state_dict_with_coverage(module, {"0.weight": torch.ones_like(module[0].weight)})


def test_coverage_loader_accepts_complete_load() -> None:
    module = nn.Linear(3, 2)
    state = {key: value.clone() for key, value in module.state_dict().items()}
    assert load_state_dict_with_coverage(module, state) == 1.0


def test_distilled_encoder_export_uses_ultralytics_module_keys() -> None:
    state = {
        "encoder.stem.conv.weight": torch.ones(4, 3, 3, 3),
        "encoder.stage3.1.cv1.conv.weight": torch.ones(4, 4, 1, 1),
        "projector.proj.weight": torch.ones(8, 4, 1, 1),
    }
    exported = export_ultralytics_backbone_state(state)
    assert set(exported) == {"model.0.conv.weight", "model.6.cv1.conv.weight"}


def test_distillation_normalization_matches_imagenet_rgb_contract() -> None:
    image = torch.tensor([[[[0.485]], [[0.456]], [[0.406]]]])
    assert torch.allclose(normalize_dino_rgb(image), torch.zeros_like(image))


def test_dino_optimizer_schedule_warms_up_then_reaches_weight_decay_end() -> None:
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0, weight_decay=0.04)
    lr0, wd0 = apply_dino_optimizer_schedule(
        optimizer,
        step=0,
        total_steps=10,
        learning_rate=1.0,
        weight_decay=0.04,
        weight_decay_end=0.4,
        warmup_steps=2,
    )
    lr_last, wd_last = apply_dino_optimizer_schedule(
        optimizer,
        step=9,
        total_steps=10,
        learning_rate=1.0,
        weight_decay=0.04,
        weight_decay_end=0.4,
        warmup_steps=2,
    )
    assert lr0 == pytest.approx(0.5)
    assert wd0 == pytest.approx(0.04)
    assert lr_last == pytest.approx(0.0)
    assert wd_last == pytest.approx(0.4)
