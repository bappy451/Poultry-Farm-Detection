from __future__ import annotations

import torch

def test_custom_trainer_options_survive_parent_initialization(monkeypatch):
    from scripts.ema_pseudo_obb import PseudoLabelEMADomainAdaptOBBTrainer
    from ultralytics.models.yolo.obb.train import OBBTrainer

    received_overrides = None

    def fake_init(self, *args, **kwargs):
        nonlocal received_overrides
        received_overrides = kwargs.get("overrides")

    monkeypatch.setattr(OBBTrainer, "__init__", fake_init)
    overrides = {
        "data": "labeled.yaml",
        "unlabeled_root": r"E:\NAIP_Dataset_PH\MS_tiles_512_256_jpg",
        "unlabeled_batch": 16,
        "pseudo_obb_weight": 0.05,
        "pseudo_thresholds": "0.90,0.90",
    }
    trainer = PseudoLabelEMADomainAdaptOBBTrainer(overrides=overrides)

    assert trainer._opt("unlabeled_root", None) == r"E:\NAIP_Dataset_PH\MS_tiles_512_256_jpg"
    assert trainer._opt("unlabeled_batch", None) == 16
    assert trainer._opt("pseudo_obb_weight", None) == 0.05
    assert trainer._opt("pseudo_thresholds", None) == "0.90,0.90"
    assert received_overrides == {"data": "labeled.yaml"}


def test_rotated_teacher_predictions_become_native_obb_targets():
    from scripts.ema_pseudo_obb import build_pseudo_batch

    # Ultralytics decoded OBB layout is [x, y, w, h, cls..., angle].
    inference = torch.tensor(
        [[[128.0, 256.0], [128.0, 256.0], [64.0, 64.0], [32.0, 32.0],
          [0.0, 0.0], [0.10, 0.95], [0.05, 0.05]]]
    )
    pseudo = build_pseudo_batch(
        inference,
        image_size=(512, 512),
        nc=2,
        class_thresholds=[0.90, 0.90],
    )
    assert pseudo["cls"].tolist() == [1]
    assert torch.allclose(pseudo["bboxes"], torch.tensor([[0.5, 0.5, 0.125, 0.0625, 0.05]]))
    assert torch.allclose(pseudo["pseudo_conf"], torch.tensor([0.95]), atol=1e-5)


def test_foreground_feature_loss_is_finite_and_backpropagates():
    from scripts.ema_pseudo_obb import foreground_feature_cosine_loss

    student = [torch.randn(1, 4, 8, 8, requires_grad=True)]
    teacher = [torch.randn(1, 4, 8, 8)]
    pseudo = {
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([1]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25, 0.0]]),
        "pseudo_conf": torch.tensor([0.95]),
    }
    loss = foreground_feature_cosine_loss(student, teacher, pseudo, foreground_gain=3.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert student[0].grad is not None
    assert torch.isfinite(student[0].grad).all()
    assert teacher[0].grad is None


def test_custom_loss_keeps_ultralytics_vector_contract():
    from ultralytics.cfg import get_cfg
    from scripts.ema_pseudo_obb import PseudoLabelEMADomainAdaptOBBModel

    model = PseudoLabelEMADomainAdaptOBBModel("yolo11n-obb.yaml", nc=2, pseudo_obb_weight=0.01)
    model.args = get_cfg()
    model.train()
    model._pseudo_targets = lambda weak: (
        {
            "batch_idx": torch.tensor([0]),
            "cls": torch.tensor([1]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25, 0.0]]),
            "pseudo_conf": torch.tensor([0.95]),
        },
        1,
    )
    batch = {
        "img": torch.zeros(1, 3, 64, 64),
        "batch_idx": torch.zeros(0, dtype=torch.long),
        "cls": torch.zeros(0, 1),
        "bboxes": torch.zeros(0, 5),
        "unlabeled_img_weak": torch.zeros(1, 3, 64, 64),
        "unlabeled_img_strong": torch.zeros(1, 3, 64, 64),
    }
    loss, loss_items = model.loss(batch)
    assert loss.shape == (4,)
    assert torch.isfinite(loss).all()
    assert loss_items.shape == (7,)