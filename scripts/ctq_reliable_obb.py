"""Reliability-gated CTQ pseudo-label model for the next validation-only study.

Only pseudo OBBs stable under independent teacher photometric jitters are
allowed to supervise the student's OBB regression/classification criterion.
Teacher and student retain the synchronized weak/strong geometry used by CTQ.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from scripts.ctq_obb_model_safe import SafeCTQOBBModel, SafeCTQOBBTrainer
from scripts.ema_pseudo_obb import _thresholds, build_pseudo_batch
from ultralytics.utils.metrics import batch_probiou


def teacher_jitter(images: torch.Tensor) -> torch.Tensor:
    """Apply a weak geometry-preserving teacher perturbation in [0, 1]."""
    if images.ndim != 4:
        raise ValueError(f"Expected BCHW images, got {tuple(images.shape)}")
    brightness = torch.empty((images.shape[0], 1, 1, 1), device=images.device).uniform_(0.92, 1.08)
    contrast = torch.empty((images.shape[0], 1, 1, 1), device=images.device).uniform_(0.92, 1.08)
    mean = images.mean(dim=(2, 3), keepdim=True)
    noise = torch.randn_like(images) * 0.006
    return ((images * brightness - mean) * contrast + mean + noise).clamp(0.0, 1.0)


def stable_pseudo_mask(
    reference: dict[str, torch.Tensor],
    jittered: list[dict[str, torch.Tensor]],
    *,
    min_iou: float,
    min_agreement: float,
) -> torch.Tensor:
    """Keep reference pseudo boxes with same-class OBB agreement across jitters."""
    if not 0.0 < min_iou <= 1.0 or not 0.0 < min_agreement <= 1.0:
        raise ValueError("min_iou and min_agreement must be in (0, 1].")
    count = int(reference["bboxes"].shape[0])
    device = reference["bboxes"].device
    if count == 0 or not jittered:
        return torch.ones(count, device=device, dtype=torch.bool)
    agreements = torch.zeros(count, device=device, dtype=torch.int64)
    ref_boxes = reference["bboxes"]
    ref_batch = reference["batch_idx"].long()
    ref_cls = reference["cls"].long()
    for candidate in jittered:
        candidate_boxes = candidate["bboxes"]
        candidate_batch = candidate["batch_idx"].long()
        candidate_cls = candidate["cls"].long()
        for index in range(count):
            compatible = (candidate_batch == ref_batch[index]) & (candidate_cls == ref_cls[index])
            if not compatible.any():
                continue
            overlaps = batch_probiou(ref_boxes[index : index + 1], candidate_boxes[compatible])
            if torch.as_tensor(overlaps, device=device).max() >= min_iou:
                agreements[index] += 1
    required = math.ceil(len(jittered) * min_agreement)
    return agreements >= required


def filter_pseudo_batch(pseudo: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
    if mask.ndim != 1 or mask.shape[0] != pseudo["bboxes"].shape[0]:
        raise ValueError("Pseudo stability mask must align with pseudo boxes.")
    return {key: value[mask] if torch.is_tensor(value) and value.ndim and value.shape[0] == mask.shape[0] else value for key, value in pseudo.items()}


class ReliableCTQOBBModel(SafeCTQOBBModel):
    def __init__(self, *args, teacher_jitter_views: int = 2, teacher_jitter_iou: float = 0.70, teacher_jitter_agreement: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if teacher_jitter_views < 0:
            raise ValueError("teacher_jitter_views must be non-negative")
        if not 0.0 < teacher_jitter_iou <= 1.0 or not 0.0 < teacher_jitter_agreement <= 1.0:
            raise ValueError("teacher jitter IoU/agreement must be in (0, 1]")
        self.teacher_jitter_views = int(teacher_jitter_views)
        self.teacher_jitter_iou = float(teacher_jitter_iou)
        self.teacher_jitter_agreement = float(teacher_jitter_agreement)

    def _pseudo_targets(self, weak: torch.Tensor) -> tuple[dict[str, torch.Tensor], int]:
        reference, _ = super()._pseudo_targets(weak)
        if self.teacher_jitter_views == 0 or reference["bboxes"].numel() == 0:
            return reference, int(reference["bboxes"].shape[0])
        teacher = object.__getattribute__(self, "_ema_teacher")
        if teacher is None:
            raise RuntimeError("EMA teacher has not been attached by the trainer.")
        thresholds = _thresholds(self.pseudo_thresholds, int(self.nc), 0.9)
        candidates: list[dict[str, torch.Tensor]] = []
        teacher.eval()
        for _ in range(self.teacher_jitter_views):
            with torch.no_grad():
                inference = teacher.predict(teacher_jitter(weak))[0]
            candidates.append(build_pseudo_batch(
                inference, image_size=(int(weak.shape[-2]), int(weak.shape[-1])), nc=int(self.nc),
                class_thresholds=thresholds, iou_threshold=self.pseudo_iou_threshold,
                max_det=self.pseudo_max_det, class_margin=self.pseudo_class_margin,
            ))
        filtered = filter_pseudo_batch(reference, stable_pseudo_mask(
            reference, candidates, min_iou=self.teacher_jitter_iou, min_agreement=self.teacher_jitter_agreement,
        ))
        return filtered, int(filtered["bboxes"].shape[0])


class ReliableCTQOBBTrainer(SafeCTQOBBTrainer):
    _CUSTOM_OPTIONS = SafeCTQOBBTrainer._CUSTOM_OPTIONS | frozenset({"teacher_jitter_views", "teacher_jitter_iou", "teacher_jitter_agreement"})

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = ReliableCTQOBBModel(
            cfg, nc=self.data["nc"], ch=self.data.get("channels", 3), verbose=verbose,
            pseudo_obb_weight=float(self._opt("pseudo_obb_weight", 0.05)),
            feature_consistency_weight=float(self._opt("feature_consistency_weight", 0.0)),
            pseudo_thresholds=_thresholds(self._opt("pseudo_thresholds", None), int(self.data["nc"]), 0.9),
            pseudo_iou_threshold=float(self._opt("pseudo_iou_threshold", 0.5)),
            pseudo_class_margin=float(self._opt("pseudo_class_margin", 0.05)),
            pseudo_max_det=int(self._opt("pseudo_max_det", 100)),
            foreground_feature_gain=float(self._opt("foreground_feature_gain", 3.0)),
            quality_alpha=float(self._opt("quality_alpha", 1.0)),
            teacher_jitter_views=int(self._opt("teacher_jitter_views", 2)),
            teacher_jitter_iou=float(self._opt("teacher_jitter_iou", 0.70)),
            teacher_jitter_agreement=float(self._opt("teacher_jitter_agreement", 1.0)),
        )
        if weights:
            model.load(weights)
        return model
