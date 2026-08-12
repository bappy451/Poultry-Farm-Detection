from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
import torch.nn as nn
import yaml

from poultry_geo.ssl_models import RemoteSensingSegmentationModel
from downstream_finetune.dino_vit_compare import (
    load_dino_backbone_into_vit,
    resolve_dino_checkpoint,
    run_suite,
)


def test_dino_vit_compare_config_schema() -> None:
    config_path = Path("configs/dino_vit_downstream_compare.yaml")
    assert config_path.exists(), "configs/dino_vit_downstream_compare.yaml must exist."

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "paths" in config
    assert "training" in config
    assert "datasets" in config
    assert "models" in config

    datasets = config["datasets"]
    models = config["models"]

    assert len(datasets) == 3, f"Expected 3 datasets, found {len(datasets)}"
    assert len(models) == 3, f"Expected 3 models, found {len(models)}"

    dataset_names = [d["name"] for d in datasets]
    assert "poultry_houses" in dataset_names
    assert "grsd_greenhouse" in dataset_names
    assert "dota_aerial" in dataset_names

    model_strategies = {m["init_strategy"] for m in models}
    assert model_strategies == {"dino", "imagenet", "random"}


def test_resolve_dino_checkpoint(tmp_path: Path) -> None:
    # 1. Test direct file path
    dummy_pt = tmp_path / "custom_ckpt.pt"
    dummy_pt.touch()
    assert resolve_dino_checkpoint(dummy_pt) == dummy_pt

    # 2. Test directory with backbone_dino_ep*.pt files
    dir_path = tmp_path / "ckpts"
    dir_path.mkdir()
    (dir_path / "backbone_dino_ep50.pt").touch()
    latest_pt = dir_path / "backbone_dino_ep100.pt"
    latest_pt.touch()

    resolved = resolve_dino_checkpoint(dir_path)
    assert resolved == latest_pt


def test_load_dino_backbone_into_vit(tmp_path: Path) -> None:
    model = RemoteSensingSegmentationModel(
        in_channels=3,
        image_size=64,
        patch_size=16,
        embed_dim=768,
        depth=2,
        num_heads=12,
        precision="fp32",
    )

    # Save a synthetic DINO student_backbone checkpoint
    synth_state = {
        "pos_embed": torch.randn(1, 16, 768),
        "patch_embed.proj.weight": torch.randn(768, 3, 16, 16),
        "patch_embed.proj.bias": torch.randn(768),
        "blocks.0.norm1.weight": torch.ones(768) * 5.0,
        "blocks.0.norm1.bias": torch.zeros(768),
        "cls_token": torch.randn(1, 1, 768), # should be ignored cleanly
    }

    ckpt_file = tmp_path / "dino_synth.pt"
    torch.save({"student_backbone": synth_state}, ckpt_file)

    model = load_dino_backbone_into_vit(model, ckpt_file)
    assert torch.allclose(model.backbone.blocks[0].norm1.weight, torch.ones(768) * 5.0)


def test_dino_vit_compare_dry_run_with_dummy_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that run_suite completes sequentially across datasets and models in dry-run mode."""
    # Create a minimal synthetic config pointing to a dummy dataset class
    config_file = tmp_path / "test_config.yaml"
    out_dir = tmp_path / "runs_test"
    dino_dir = tmp_path / "dino_test"
    dino_dir.mkdir()
    (dino_dir / "last.pt").touch()

    # Save a minimal valid checkpoint to last.pt
    synth_ckpt = {"student_backbone": {"pos_embed": torch.zeros(1, 1, 768)}}
    torch.save(synth_ckpt, dino_dir / "last.pt")

    dummy_a = tmp_path / "dummy_a.yaml"
    dummy_a.touch()
    dummy_b = tmp_path / "dummy_b.yaml"
    dummy_b.touch()

    config_data = {
        "paths": {
            "dino_checkpoint_dir": str(dino_dir),
            "output_dir": str(out_dir),
        },
        "training": {
            "epochs": 1,
            "batch": 2,
            "imgsz": 32,
            "device": "cpu",
            "num_workers": 0,
        },
        "datasets": [
            {"name": "ds_a", "yaml": str(dummy_a)},
            {"name": "ds_b", "yaml": str(dummy_b)},
        ],
        "models": [
            {"name": "dino_pretrained", "init_strategy": "dino"},
            {"name": "random_init", "init_strategy": "random"},
        ],
    }
    config_file.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    # Mock dataset and dataloader so dry run doesn't need disk images
    class DummySegDataset:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __len__(self) -> int:
            return 2

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return {
                "image": torch.randn(3, 32, 32),
                "mask": torch.zeros(32, 32, dtype=torch.long),
            }

    import downstream_finetune.dino_vit_compare as dvc
    monkeypatch.setattr(dvc, "YoloFormatSegmentationDataset", DummySegDataset)

    # Run the suite in dry-run mode
    results, df = run_suite(config_file, dry_run=True)

    # We defined 2 datasets x 2 models = 4 experiments
    assert len(results) == 4
    assert len(df) == 4
    assert (out_dir / "comparison_results.csv").exists()

    expected_names = [
        ("ds_a", "dino_pretrained"),
        ("ds_a", "random_init"),
        ("ds_b", "dino_pretrained"),
        ("ds_b", "random_init"),
    ]
    actual_names = [(r["dataset"], r["model"]) for r in results]
    assert actual_names == expected_names
    for idx, (expected_ds, expected_model) in enumerate(expected_names):
        assert results[idx]["dataset"] == expected_ds
        assert results[idx]["model"] == expected_model


def test_dino_vit_compare_smart_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that smart resume skips experiments where last.pt indicates epoch >= epochs."""
    out_dir = tmp_path / "runs_resume"
    out_dir.mkdir()
    exp_dir = out_dir / "ds_a__random_init" / "weights"
    exp_dir.mkdir(parents=True)
    
    # Save a synthetic checkpoint at epoch 5 when total epochs is 5
    synth_ckpt = {
        "epoch": 5,
        "epochs": 5,
        "best_f1": 0.85,
        "best_iou": 0.74,
        "best_precision": 0.88,
        "best_recall": 0.82,
        "best_val_loss": 0.25,
    }
    torch.save(synth_ckpt, exp_dir / "last.pt")

    config_file = tmp_path / "test_resume_config.yaml"
    dummy_yaml = tmp_path / "dummy.yaml"
    dummy_yaml.touch()

    config_data = {
        "paths": {
            "dino_checkpoint_dir": "runs/dino_pretrain",
            "output_dir": str(out_dir),
        },
        "training": {
            "epochs": 5,
            "batch": 2,
            "imgsz": 32,
            "device": "cpu",
        },
        "datasets": [{"name": "ds_a", "yaml": str(dummy_yaml)}],
        "models": [{"name": "random_init", "init_strategy": "random"}],
    }
    config_file.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    from downstream_finetune.dino_vit_compare import run_suite
    results, df = run_suite(config_file, dry_run=False)

    assert len(results) == 1
    # Check that it loaded the metrics from the skipped checkpoint
    assert results[0]["f1_score"] == 0.85
    assert results[0]["iou_score"] == 0.74
