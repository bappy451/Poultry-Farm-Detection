import pytest
import yaml
import torch
from pathlib import Path
from downstream_finetune.finetune import finetune_with_config

def test_finetune_with_config_parses_yaml(tmp_path, monkeypatch):
    # Create a dummy config
    config = {
        "models": [
            {
                "path": "yolo11l-seg.pt",
                "name": "test_run"
            }
        ],
        "training": {
            "epochs": 1,
            "batch": 16,
            "project": str(tmp_path / "runs"),
        },
        "data": {
            "path": "data.yaml"
        }
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Mock YOLO
    class MockYOLO:
        def __init__(self, path):
            self.path = path
            self.train_args = None
            self.val_args = None

        def train(self, **kwargs):
            self.train_args = kwargs
            return "results"

        def val(self, **kwargs):
            self.val_args = kwargs
            return "metrics"

    monkeypatch.setattr("downstream_finetune.finetune.YOLO", MockYOLO)

    # Run finetune
    results_all, metrics_all = finetune_with_config(config_path)

    assert results_all[0] == "results"
    assert metrics_all[0] == "metrics"

def test_load_spark_backbone_into_yolo_rejects_missing_or_unverified_artifacts(tmp_path):
    from downstream_finetune.finetune import load_spark_backbone_into_yolo

    with pytest.raises(FileNotFoundError):
        load_spark_backbone_into_yolo(object(), tmp_path / "missing.pt")

    artifact = tmp_path / "spark.pt"
    torch.save({"backbone_type": "resnet50"}, artifact)
    with pytest.raises(ValueError, match="no verified SparK-to-YOLO adapter"):
        load_spark_backbone_into_yolo(object(), artifact)
