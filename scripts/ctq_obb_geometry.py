"""Geometry-auxiliary OBB loss built on the quality-aware assignment path."""

from __future__ import annotations

import torch

from scripts.ctq_obb_losses import cyclic_corner_loss, periodic_angle_loss, symmetric_gaussian_kld
from scripts.ctq_obb_quality_safe import SafeQualityAwareOBBLoss


class GeometryQualityAwareOBBLoss(SafeQualityAwareOBBLoss):
    """Add one selected differentiable geometry term to native OBB loss.

    The parent criterion owns assignment. This class intentionally does not
    combine KLD, corner, and angle terms in one run: ``geometry_kind`` selects
    exactly one auxiliary, preserving causal attribution in the grid search.
    """

    def __init__(self, model, *, geometry_kind: str = "none", geometry_weight: float = 0.0, angle_kappa: float = 1.0):
        super().__init__(model)
        if geometry_kind not in {"none", "symmetric_kld", "cyclic_corner", "periodic_angle"}:
            raise ValueError("Unsupported geometry_kind")
        if geometry_weight < 0 or angle_kappa <= 0:
            raise ValueError("Invalid geometry parameters")
        self.geometry_kind = geometry_kind
        self.geometry_weight = float(geometry_weight)
        self.angle_kappa = float(angle_kappa)

    def auxiliary(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        if self.geometry_kind == "symmetric_kld":
            return symmetric_gaussian_kld(pred_boxes, target_boxes)
        if self.geometry_kind == "cyclic_corner":
            return cyclic_corner_loss(pred_boxes, target_boxes)
        if self.geometry_kind == "periodic_angle":
            return periodic_angle_loss(pred_boxes[:, 4], target_boxes[:, 4], target_boxes[:, 2], target_boxes[:, 3], kappa=self.angle_kappa)
        return pred_boxes.new_zeros(())
