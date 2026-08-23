"""Safety wrapper for CTQ quality loss on empty ground-truth batches."""

from __future__ import annotations

import torch

from ultralytics.utils.loss import v8OBBLoss

from scripts.ctq_obb_quality import QualityAwareOBBLoss


class SafeQualityAwareOBBLoss(QualityAwareOBBLoss):
    """Use the native empty-target path when a supervised batch has no boxes."""

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        if batch["batch_idx"].numel() == 0:
            return v8OBBLoss.loss(self, preds, batch)
        return super().loss(preds, batch)
