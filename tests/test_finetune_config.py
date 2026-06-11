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

def test_load_spark_backbone_into_yolo(monkeypatch):
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 3)
            self.bn = torch.nn.BatchNorm2d(64)
            
    class MockYOLO:
        def __init__(self):
            self.model = torch.nn.Module()
            self.model.model = torch.nn.Sequential(
                torch.nn.Module(), # model.0
            )
            self.model.model[0].conv = torch.nn.Conv2d(3, 64, 3)
            self.model.model[0].bn = torch.nn.BatchNorm2d(64)
            # Add state_dict keys that match mapping
            self.model.state_dict = lambda: {
                "model.0.conv.weight": torch.zeros(64, 3, 3, 3),
                "model.0.bn.weight": torch.zeros(64),
                "model.0.bn.bias": torch.zeros(64),
            }
            self.model.load_state_dict = lambda d, strict: torch.nn.modules.module._IncompatibleKeys([], [])

    from downstream_finetune.finetune import load_spark_backbone_into_yolo
    
    mock_yolo = MockYOLO()
    
    # Mock torch.load
    spark_state = {
        "model_state_dict": {
            "encoder.stem.weight": torch.ones(64, 3, 3, 3),
            "encoder.stem.ln.weight": torch.ones(64),
            "encoder.stem.ln.bias": torch.ones(64),
        }
    }
    monkeypatch.setattr("torch.load", lambda p, map_location: spark_state)
    
    loaded_model = load_spark_backbone_into_yolo(mock_yolo, "fake_path.pt")
    assert loaded_model == mock_yolo
