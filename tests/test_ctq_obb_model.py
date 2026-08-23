from __future__ import annotations


def test_ctq_trainer_accepts_quality_alpha(monkeypatch):
    from scripts.ctq_obb_model import CTQOBBTrainer
    from ultralytics.models.yolo.obb.train import OBBTrainer

    received = None

    def fake_init(self, *args, **kwargs):
        nonlocal received
        received = kwargs.get("overrides")

    monkeypatch.setattr(OBBTrainer, "__init__", fake_init)
    trainer = CTQOBBTrainer(overrides={"data": "data.yaml", "quality_alpha": 2.0, "pseudo_obb_weight": 0.025})
    assert trainer._opt("quality_alpha", None) == 2.0
    assert trainer._opt("pseudo_obb_weight", None) == 0.025
    assert received == {"data": "data.yaml"}
