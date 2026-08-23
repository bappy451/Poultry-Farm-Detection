from __future__ import annotations

import math

import torch

from scripts.ctq_obb_losses import asymmetric_focal_loss, cyclic_corner_loss, periodic_angle_loss, symmetric_gaussian_kld


def test_periodic_angle_is_zero_for_equivalent_rectangles():
    loss = periodic_angle_loss(torch.tensor([0.0]), torch.tensor([math.pi]), torch.tensor([10.0]), torch.tensor([2.0]))
    assert loss.item() < 1e-10


def test_corner_loss_is_invariant_to_pi_angle_equivalence():
    prediction = torch.tensor([[0.5, 0.5, 0.4, 0.1, 0.0]], requires_grad=True)
    target = torch.tensor([[0.5, 0.5, 0.4, 0.1, math.pi]])
    loss = cyclic_corner_loss(prediction, target)
    assert loss.item() < 1e-6
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_symmetric_kld_is_nonnegative_and_backpropagates():
    prediction = torch.tensor([[0.5, 0.5, 0.4, 0.1, 0.1]], requires_grad=True)
    target = torch.tensor([[0.5, 0.5, 0.4, 0.1, 0.0]])
    loss = symmetric_gaussian_kld(prediction, target)
    assert loss.item() >= 0.0
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_asymmetric_focal_is_finite():
    loss = asymmetric_focal_loss(torch.tensor([[2.0, -2.0]]), torch.tensor([[1.0, 0.0]]), gamma_pos=1.0, gamma_neg=4.0)
    assert torch.isfinite(loss)
