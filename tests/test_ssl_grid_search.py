from __future__ import annotations

import yaml
from pathlib import Path
import pytest
import torch

from poultry_geo.ssl_grid_search import (
    load_ssl_grid_search_config,
    expand_ssl_grid_search_configs,
    run_stateful_ssl_grid_search,
)

def test_load_ssl_grid_search_config(tmp_path: Path) -> None:
    config_dict = {
        "data": {
            "image_dir": "some_dir",
            "image_size": 224,
            "n_channels": 3,
            "batch_size": 16,
            "num_workers": 4,
        },
        "runtime": {
            "device": "cpu",
            "seed": 27,
            "output_dir": str(tmp_path / "outputs"),
            "epochs": 1,
        },
        "grid_search": {
            "model_name": ["S"],
            "use_lora": [False],
            "freeze_backbone": [False],
            "mask_ratio": [0.75],
            "optimizer": ["adamw"],
            "precision": ["fp32"],
            "learning_rate": [0.0001],
            "weight_decay": [0.05],
        }
    }
    config_path = tmp_path / "ssl_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)
    
    loaded = load_ssl_grid_search_config(config_path)
    assert loaded["data"]["image_size"] == 224
    assert loaded["grid_search"]["use_lora"] == [False]

def test_expand_ssl_grid_search_configs() -> None:
    config = {
        "data": {"image_dir": "dir", "image_size": 224, "n_channels": 3, "batch_size": 16, "num_workers": 4},
        "runtime": {"device": "cpu", "seed": 27, "output_dir": "out", "epochs": 1},
        "grid_search": {
            "model_name": ["S", "B"],
            "use_lora": [False, True],
            "freeze_backbone": [False],
            "mask_ratio": [0.75],
            "optimizer": ["adamw"],
            "precision": ["fp32"],
            "learning_rate": [0.0001],
            "weight_decay": [0.05],
        }
    }
    experiments = expand_ssl_grid_search_configs(config)
    # 2 (model_name) * 2 (use_lora) = 4 experiments
    assert len(experiments) == 4
    
    # Check if use_lora is correctly set in one of the experiments
    assert any(exp["grid"]["use_lora"] is True for exp in experiments)
    assert any(exp["grid"]["use_lora"] is False for exp in experiments)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Full grid search might be too slow on CPU for smoke test")
def test_run_stateful_ssl_grid_search_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # This is a very heavy test, maybe just mock _train_one_epoch
    pass
