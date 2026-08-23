"""Mathematically stable auxiliary losses for CTQ-OBB experiments.

The functions in this module are deliberately framework-independent.  The
training adapter consumes them only where it has differentiable, correctly
matched OBB predictions; no loss is approximated through NMS output.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def periodic_angle_loss(pred_angle: torch.Tensor, target_angle: torch.Tensor, width: torch.Tensor, height: torch.Tensor, *, kappa: float = 1.0) -> torch.Tensor:
    """Aspect-ratio-aware, pi-periodic rectangle angle loss."""
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    ratio = torch.abs(torch.log((width.clamp_min(1e-6)) / height.clamp_min(1e-6)))
    weight = ratio / (ratio + kappa)
    delta = pred_angle - target_angle
    return (weight * torch.sin(delta).square()).mean()


def obb_corners(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h, theta] OBBs to four corners in cyclic order."""
    if boxes.ndim != 2 or boxes.shape[-1] != 5:
        raise ValueError("boxes must have shape [N, 5]")
    cx, cy, width, height, angle = boxes.unbind(-1)
    half = torch.stack((width, height), dim=-1).unsqueeze(1) * 0.5
    signs = boxes.new_tensor(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
    local = half * signs
    cos_a, sin_a = angle.cos(), angle.sin()
    rotation = torch.stack((torch.stack((cos_a, -sin_a), -1), torch.stack((sin_a, cos_a), -1)), -2)
    return local @ rotation.transpose(-1, -2) + torch.stack((cx, cy), -1).unsqueeze(1)


def cyclic_corner_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, *, beta: float = 1.0) -> torch.Tensor:
    """Smooth-L1 corner loss invariant to a cyclic change in rectangle start corner."""
    if pred_boxes.shape != target_boxes.shape:
        raise ValueError("pred_boxes and target_boxes must have the same shape")
    if beta <= 0:
        raise ValueError("beta must be positive")
    pred = obb_corners(pred_boxes)
    target = obb_corners(target_boxes)
    losses = [F.smooth_l1_loss(pred, target.roll(shifts=shift, dims=1), beta=beta, reduction="none").mean(dim=(1, 2)) for shift in range(4)]
    return torch.stack(losses, dim=0).min(dim=0).values.mean()


def obb_to_gaussian(boxes: torch.Tensor, *, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Map OBBs to 2D Gaussian means and positive-definite covariances."""
    if boxes.ndim != 2 or boxes.shape[-1] != 5:
        raise ValueError("boxes must have shape [N, 5]")
    mean = boxes[:, :2]
    width, height, angle = boxes[:, 2].clamp_min(eps), boxes[:, 3].clamp_min(eps), boxes[:, 4]
    cos_a, sin_a = angle.cos(), angle.sin()
    rotation = torch.stack((torch.stack((cos_a, -sin_a), -1), torch.stack((sin_a, cos_a), -1)), -2)
    diagonal = torch.diag_embed(torch.stack((width.square(), height.square()), -1) / 12.0)
    return mean, rotation @ diagonal @ rotation.transpose(-1, -2)


def symmetric_gaussian_kld(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Symmetric KL divergence between OBB-derived Gaussians."""
    if pred_boxes.shape != target_boxes.shape:
        raise ValueError("pred_boxes and target_boxes must have the same shape")
    mean_p, cov_p = obb_to_gaussian(pred_boxes, eps=eps)
    mean_t, cov_t = obb_to_gaussian(target_boxes, eps=eps)
    inv_p, inv_t = torch.linalg.inv(cov_p), torch.linalg.inv(cov_t)
    delta = (mean_t - mean_p).unsqueeze(-1)
    dim = 2.0
    kl_pt = 0.5 * (torch.diagonal(inv_t @ cov_p, dim1=-2, dim2=-1).sum(-1) + (delta.transpose(-1, -2) @ inv_t @ delta).squeeze((-1, -2)) - dim + torch.logdet(cov_t) - torch.logdet(cov_p))
    kl_tp = 0.5 * (torch.diagonal(inv_p @ cov_t, dim1=-2, dim2=-1).sum(-1) + (delta.transpose(-1, -2) @ inv_p @ delta).squeeze((-1, -2)) - dim + torch.logdet(cov_p) - torch.logdet(cov_t))
    return (0.5 * (kl_pt + kl_tp)).mean()


def asymmetric_focal_loss(logits: torch.Tensor, targets: torch.Tensor, *, gamma_pos: float, gamma_neg: float) -> torch.Tensor:
    """Binary asymmetric focal loss with stable logit-space BCE."""
    if gamma_pos < 0 or gamma_neg < 0:
        raise ValueError("focal gammas must be non-negative")
    targets = targets.to(dtype=logits.dtype)
    probability = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    focusing = torch.where(targets > 0, (1.0 - probability).pow(gamma_pos), probability.pow(gamma_neg))
    return (focusing * bce).mean()
