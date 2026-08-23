"""Safe standalone CTQ trainer variant used by the experiment launcher."""

from __future__ import annotations

from scripts.ctq_obb_model import CTQOBBModel, CTQOBBTrainer
from scripts.ctq_obb_quality_safe import SafeQualityAwareOBBLoss
from scripts.ema_pseudo_obb import _thresholds


class SafeCTQOBBModel(CTQOBBModel):
    def init_criterion(self):
        return SafeQualityAwareOBBLoss(self)


class SafeCTQOBBTrainer(CTQOBBTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = SafeCTQOBBModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data.get("channels", 3),
            verbose=verbose,
            pseudo_obb_weight=float(self._opt("pseudo_obb_weight", 0.05)),
            feature_consistency_weight=float(self._opt("feature_consistency_weight", 0.0)),
            pseudo_thresholds=_thresholds(self._opt("pseudo_thresholds", None), int(self.data["nc"]), 0.9),
            pseudo_iou_threshold=float(self._opt("pseudo_iou_threshold", 0.5)),
            pseudo_class_margin=float(self._opt("pseudo_class_margin", 0.05)),
            pseudo_max_det=int(self._opt("pseudo_max_det", 100)),
            foreground_feature_gain=float(self._opt("foreground_feature_gain", 3.0)),
            quality_alpha=float(self._opt("quality_alpha", 1.0)),
        )
        if weights:
            model.load(weights)
        return model
