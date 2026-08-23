from __future__ import annotations

import torch

from scripts.ctq_obb_geometry import GeometryQualityAwareOBBLoss


def make_criterion(kind: str):
    criterion = object.__new__(GeometryQualityAwareOBBLoss)
    criterion.geometry_kind = kind
    criterion.angle_kappa = 1.0
    return criterion


def test_geometry_auxiliary_is_finite_for_each_supported_kind():
    pred = torch.tensor([[0.5, 0.5, 0.4, 0.1, 0.1]], requires_grad=True)
    target = torch.tensor([[0.5, 0.5, 0.4, 0.1, 0.0]])
    losses = [make_criterion(kind).auxiliary(pred, target) for kind in ("symmetric_kld", "cyclic_corner", "periodic_angle", "none")]
    total = sum(losses)
    total.backward()
    assert torch.isfinite(total)
    assert torch.isfinite(pred.grad).all()
