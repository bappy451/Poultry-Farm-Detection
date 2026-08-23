"""Standalone CTQ quality-aware EMA pseudo-OBB model and trainer."""

from __future__ import annotations

from typing import Any

import torch

from scripts.ctq_obb_quality import QualityAwareOBBLoss
from scripts.ema_pseudo_obb import PseudoLabelEMADomainAdaptOBBModel, PseudoLabelEMADomainAdaptOBBTrainer


class CTQOBBModel(PseudoLabelEMADomainAdaptOBBModel):
    """Pseudo-OBB model with native per-target confidence weighting."""

    def __init__(self, *args, quality_alpha: float = 1.0, **kwargs) -> None:
        if quality_alpha < 0:
            raise ValueError("quality_alpha must be non-negative")
        super().__init__(*args, **kwargs)
        self.quality_alpha = float(quality_alpha)

    def init_criterion(self):
        return QualityAwareOBBLoss(self)

    def loss(self, batch: dict[str, Any], preds=None):
        # This call dispatches to the native OBB model, using a unit quality
        # vector for human-labeled boxes (provided by QualityAwareOBBLoss).
        supervised_loss, loss_items = super(PseudoLabelEMADomainAdaptOBBModel, self).loss(batch, preds)
        weak = batch.get("unlabeled_img_weak")
        strong = batch.get("unlabeled_img_strong")
        pseudo_loss = torch.zeros_like(supervised_loss)
        feature_loss = supervised_loss.new_zeros(())
        pseudo_count = 0
        if weak is not None and strong is not None and (self.pseudo_obb_weight > 0 or self.feature_consistency_weight > 0):
            pseudo_batch, pseudo_count = self._pseudo_targets(weak)
            if self.pseudo_obb_weight > 0 and pseudo_count > 0:
                pseudo_batch = dict(pseudo_batch)
                pseudo_batch["quality"] = pseudo_batch["pseudo_conf"].pow(self.quality_alpha).detach()
                student_preds = self.forward(strong)
                pseudo_loss, _ = self.criterion(student_preds, pseudo_batch)
            if self.feature_consistency_weight > 0:
                from scripts.ema_pseudo_obb import foreground_feature_cosine_loss
                from poultry_geo.ema_adapt import _backbone_features

                teacher = object.__getattribute__(self, "_ema_teacher")
                with torch.no_grad():
                    teacher_features = _backbone_features(teacher, weak)
                student_features = _backbone_features(self, strong)
                feature_loss = foreground_feature_cosine_loss(student_features, teacher_features, pseudo_batch, foreground_gain=self.foreground_feature_gain)
        total_loss = supervised_loss + self.pseudo_obb_weight * pseudo_loss
        if self.feature_consistency_weight > 0:
            total_loss = total_loss.clone()
            total_loss[0] += self.feature_consistency_weight * feature_loss
        if not torch.is_tensor(loss_items):
            loss_items = torch.as_tensor(loss_items, device=total_loss.device)
        extras = torch.stack((pseudo_loss.detach().sum(), feature_loss.detach(), total_loss.new_tensor(float(pseudo_count))))
        return total_loss, torch.cat((loss_items.detach().reshape(-1), extras))


class CTQOBBTrainer(PseudoLabelEMADomainAdaptOBBTrainer):
    """Existing EMA loader/teacher workflow with the CTQ quality criterion."""

    _CUSTOM_OPTIONS = PseudoLabelEMADomainAdaptOBBTrainer._CUSTOM_OPTIONS | frozenset({"quality_alpha"})

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = CTQOBBModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data.get("channels", 3),
            verbose=verbose,
            pseudo_obb_weight=float(self._opt("pseudo_obb_weight", 0.05)),
            feature_consistency_weight=float(self._opt("feature_consistency_weight", 0.0)),
            pseudo_thresholds=self._opt("pseudo_thresholds", None),
            pseudo_iou_threshold=float(self._opt("pseudo_iou_threshold", 0.5)),
            pseudo_class_margin=float(self._opt("pseudo_class_margin", 0.05)),
            pseudo_max_det=int(self._opt("pseudo_max_det", 100)),
            foreground_feature_gain=float(self._opt("foreground_feature_gain", 3.0)),
            quality_alpha=float(self._opt("quality_alpha", 1.0)),
        )
        if weights:
            model.load(weights)
        return model
