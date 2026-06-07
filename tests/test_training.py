from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from poultry_geo.training import train_yolo_models


class _FakeModel:
    def __init__(self) -> None:
        self.train_kwargs = None
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        return {"status": "ok"}


def test_train_yolo_models_uses_runtime_loader(monkeypatch, tmp_path: Path) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_model = _FakeModel()
    monkeypatch.setattr(
        "poultry_geo.training.load_ultralytics_model",
        lambda model_name, family="auto": fake_model,
    )

    runs = train_yolo_models(
        models=["rtdetr-l.pt"],
        data_path="data.yaml",
        project_dir=tmp_path / "runs",
        epochs=2,
        batch_size=4,
        image_size=256,
        family="rtdetr",
    )

    assert len(runs) == 1
    assert fake_model.device == "cpu"
    assert fake_model.train_kwargs["data"] == "data.yaml"
    assert fake_model.train_kwargs["epochs"] == 2
    assert fake_model.train_kwargs["batch"] == 4

