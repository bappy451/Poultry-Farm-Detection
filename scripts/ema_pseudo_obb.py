"""Detector-level EMA pseudo-label adaptation for YOLO11-OBB.

This module is intentionally separate from ``poultry_geo.ema_adapt``.  The
existing trainer performs only backbone feature consistency.  This trainer
adds a high-confidence teacher pseudo-OBB loss while retaining the optional
feature-consistency term for controlled ablations.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.checkpoint_retry import CheckpointSaveRetryMixin
from poultry_geo.ema_adapt import (
    FEATURE_INDICES,
    EMADomainAdaptOBBTrainer,
    PairedLoader,
    PairedNAIPDataset,
    _backbone_features,
)

try:
    from ultralytics.models.yolo.obb.train import OBBModel, OBBTrainer
    from ultralytics.utils import LOGGER, RANK
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.torch_utils import unwrap_model
except ImportError as exc:  # pragma: no cover
    raise ImportError("Ultralytics is required for EMA pseudo-OBB adaptation.") from exc


def _raw_obb_predictions(output: Any) -> dict[str, torch.Tensor]:
    """Extract the train-style raw OBB dictionary from train/eval model output."""
    if isinstance(output, dict):
        raw = output
    elif isinstance(output, (tuple, list)) and len(output) > 1 and isinstance(output[1], dict):
        raw = output[1]
    else:
        raise TypeError("Expected YOLO-OBB output containing a raw prediction dictionary.")
    required = {"boxes", "scores", "angle", "feats"}
    missing = required.difference(raw)
    if missing:
        raise KeyError(f"Raw OBB predictions are missing fields: {sorted(missing)}")
    return raw


def _thresholds(value: str | Iterable[float] | None, nc: int, default: float) -> list[float]:
    if value is None:
        values = [float(default)]
    elif isinstance(value, str):
        values = [float(token.strip()) for token in value.split(",") if token.strip()]
    else:
        values = [float(token) for token in value]
    if not values:
        values = [float(default)]
    if len(values) == 1:
        values *= nc
    if len(values) != nc:
        raise ValueError(f"Expected one pseudo threshold or {nc} class thresholds, got {len(values)}.")
    if any(not 0.0 < token < 1.0 for token in values):
        raise ValueError("Pseudo-label thresholds must be in (0, 1).")
    return values


def build_pseudo_batch(
    inference: torch.Tensor,
    *,
    image_size: tuple[int, int],
    nc: int,
    class_thresholds: Iterable[float],
    iou_threshold: float = 0.5,
    max_det: int = 100,
    class_margin: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Convert decoded teacher OBB predictions to the native loss batch format.

    ``inference`` is the decoded Ultralytics tensor with ``xywhr`` followed by
    class scores.  The returned boxes are normalized ``xywhr`` in radians,
    exactly as required by ``v8OBBLoss``.
    """
    height, width = image_size
    thresholds = list(class_thresholds)
    if len(thresholds) != nc:
        raise ValueError(f"class_thresholds must have length {nc}.")
    minimum = min(thresholds)
    detections, kept_indices = non_max_suppression(
        inference,
        conf_thres=minimum,
        iou_thres=float(iou_threshold),
        max_det=int(max_det),
        nc=int(nc),
        rotated=True,
        return_idxs=True,
    )
    device = inference.device
    batch_indices: list[torch.Tensor] = []
    classes: list[torch.Tensor] = []
    boxes: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    scale = inference.new_tensor([width, height, width, height, 1.0])
    for batch_index, det in enumerate(detections):
        if det.numel() == 0:
            continue
        # Rotated NMS returns xywh, confidence, class, angle.
        cls = det[:, 5].long()
        conf = det[:, 4]
        keep = torch.zeros_like(conf, dtype=torch.bool)
        for class_index, threshold in enumerate(thresholds):
            keep |= (cls == class_index) & (conf >= threshold)
        if not keep.any():
            continue
        if class_margin > 0.0:
            source = kept_indices[batch_index].to(device=inference.device, dtype=torch.long)
            class_scores = inference[batch_index, 4 : 4 + nc, source]
            if nc > 1:
                margin = class_scores.topk(k=2, dim=0).values
                keep &= (margin[0] - margin[1]) >= float(class_margin)
        if not keep.any():
            continue
        selected = det[keep]
        selected_cls = cls[keep]
        selected_conf = conf[keep]
        selected_boxes = torch.cat((selected[:, :4], selected[:, 6:7]), dim=1) / scale
        selected_boxes[:, :4] = selected_boxes[:, :4].clamp(0.0, 1.0)
        valid_size = (selected_boxes[:, 2] * width >= 2.0) & (selected_boxes[:, 3] * height >= 2.0)
        if not valid_size.any():
            continue
        selected_boxes = selected_boxes[valid_size]
        selected_cls = selected_cls[valid_size]
        selected_conf = selected_conf[valid_size]
        batch_indices.append(torch.full((len(selected_boxes),), batch_index, device=device, dtype=torch.long))
        classes.append(selected_cls.to(device=device, dtype=torch.long))
        boxes.append(selected_boxes.to(device=device, dtype=torch.float32))
        confidences.append(selected_conf.to(device=device, dtype=torch.float32))
    if not boxes:
        return {
            "batch_idx": torch.zeros(0, device=device, dtype=torch.long),
            "cls": torch.zeros(0, device=device, dtype=torch.long),
            "bboxes": torch.zeros((0, 5), device=device, dtype=torch.float32),
            "pseudo_conf": torch.zeros(0, device=device, dtype=torch.float32),
        }
    return {
        "batch_idx": torch.cat(batch_indices),
        "cls": torch.cat(classes),
        "bboxes": torch.cat(boxes),
        "pseudo_conf": torch.cat(confidences),
    }


def _box_foreground_masks(
    pseudo_batch: dict[str, torch.Tensor],
    *,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Build soft foreground masks for feature-map spatial weighting."""
    masks = torch.zeros((batch_size, 1, height, width), device=device, dtype=torch.float32)
    if pseudo_batch["bboxes"].numel() == 0:
        return masks
    bboxes = pseudo_batch["bboxes"]
    batch_idx = pseudo_batch["batch_idx"]
    for index in range(len(bboxes)):
        cx, cy, bw, bh = bboxes[index, :4]
        x0 = max(0, min(width, int(torch.floor((cx - bw / 2) * width).item())))
        x1 = max(0, min(width, int(torch.ceil((cx + bw / 2) * width).item()) + 1))
        y0 = max(0, min(height, int(torch.floor((cy - bh / 2) * height).item())))
        y1 = max(0, min(height, int(torch.ceil((cy + bh / 2) * height).item()) + 1))
        if x1 > x0 and y1 > y0:
            masks[batch_idx[index], :, y0:y1, x0:x1] = 1.0
    return masks


def foreground_feature_cosine_loss(
    student_features: list[torch.Tensor],
    teacher_features: list[torch.Tensor],
    pseudo_batch: dict[str, torch.Tensor],
    *,
    foreground_gain: float = 3.0,
) -> torch.Tensor:
    """Cosine consistency with extra weight on teacher-selected foreground."""
    if len(student_features) != len(teacher_features):
        raise ValueError("Teacher/student feature lists must have the same length.")
    losses = []
    for student, teacher in zip(student_features, teacher_features):
        if student.shape != teacher.shape:
            raise RuntimeError(f"Teacher/student feature shapes differ: {student.shape} vs {teacher.shape}")
        student_n = F.normalize(student.float(), dim=1, eps=1e-6)
        teacher_n = F.normalize(teacher.float(), dim=1, eps=1e-6)
        error = 1.0 - (student_n * teacher_n).sum(dim=1, keepdim=True)
        mask = _box_foreground_masks(
            pseudo_batch,
            batch_size=student.shape[0],
            height=student.shape[-2],
            width=student.shape[-1],
            device=student.device,
        )
        weights = 1.0 + max(float(foreground_gain), 0.0) * mask
        losses.append((error * weights).sum() / weights.sum().clamp_min(1.0))
    return torch.stack(losses).mean() if losses else student_features[0].new_zeros(())


class PseudoLabelEMADomainAdaptOBBModel(OBBModel):
    """YOLO-OBB with teacher pseudo-label and optional feature losses."""

    def __init__(
        self,
        *args,
        pseudo_obb_weight: float = 0.05,
        feature_consistency_weight: float = 0.0,
        pseudo_thresholds: Iterable[float] | None = None,
        pseudo_iou_threshold: float = 0.5,
        pseudo_class_margin: float = 0.05,
        pseudo_max_det: int = 100,
        foreground_feature_gain: float = 3.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pseudo_obb_weight = float(pseudo_obb_weight)
        self.feature_consistency_weight = float(feature_consistency_weight)
        self.pseudo_thresholds = list(pseudo_thresholds or [0.9])
        self.pseudo_iou_threshold = float(pseudo_iou_threshold)
        self.pseudo_class_margin = float(pseudo_class_margin)
        if not 0.0 <= self.pseudo_class_margin < 1.0:
            raise ValueError("pseudo_class_margin must be in [0, 1).")
        self.pseudo_max_det = int(pseudo_max_det)
        self.foreground_feature_gain = float(foreground_feature_gain)
        object.__setattr__(self, "_ema_teacher", None)

    def _teacher_output(self, weak: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        teacher = object.__getattribute__(self, "_ema_teacher")
        if teacher is None:
            raise RuntimeError("EMA teacher has not been attached by the trainer.")
        teacher.eval()
        with torch.no_grad():
            output = teacher(weak)
        raw = _raw_obb_predictions(output)
        if isinstance(output, (tuple, list)):
            inference = output[0]
        else:
            # Training-mode raw outputs need the model's inference path for NMS.
            was_training = teacher.training
            teacher.eval()
            with torch.no_grad():
                inference = teacher.predict(weak)[0]
            if was_training:
                teacher.eval()
        return raw, inference

    def _pseudo_targets(self, weak: torch.Tensor) -> tuple[dict[str, torch.Tensor], int]:
        raw, inference = self._teacher_output(weak)
        del raw  # raw features are recomputed only when feature consistency is enabled.
        thresholds = _thresholds(self.pseudo_thresholds, int(self.nc), 0.9)
        pseudo = build_pseudo_batch(
            inference,
            image_size=(int(weak.shape[-2]), int(weak.shape[-1])),
            nc=int(self.nc),
            class_thresholds=thresholds,
            iou_threshold=self.pseudo_iou_threshold,
            max_det=self.pseudo_max_det,
            class_margin=self.pseudo_class_margin,
        )
        return pseudo, int(pseudo["bboxes"].shape[0])

    def loss(self, batch: dict[str, Any], preds=None):
        supervised_loss, loss_items = super().loss(batch, preds)
        weak = batch.get("unlabeled_img_weak")
        strong = batch.get("unlabeled_img_strong")
        pseudo_loss = torch.zeros_like(supervised_loss)
        feature_loss = supervised_loss.new_zeros(())
        pseudo_count = 0
        if weak is not None and strong is not None and (self.pseudo_obb_weight > 0 or self.feature_consistency_weight > 0):
            pseudo_batch, pseudo_count = self._pseudo_targets(weak)
            if self.pseudo_obb_weight > 0 and pseudo_count > 0:
                student_preds = self.forward(strong)
                pseudo_loss, _ = self.criterion(student_preds, pseudo_batch)
                confidence = pseudo_batch["pseudo_conf"].mean().detach().clamp(0.0, 1.0)
                pseudo_loss = pseudo_loss * confidence
            if self.feature_consistency_weight > 0:
                teacher = object.__getattribute__(self, "_ema_teacher")
                with torch.no_grad():
                    teacher_features = _backbone_features(teacher, weak)
                student_features = _backbone_features(self, strong)
                feature_loss = foreground_feature_cosine_loss(
                    student_features,
                    teacher_features,
                    pseudo_batch,
                    foreground_gain=self.foreground_feature_gain,
                )
        # Ultralytics keeps four loss channels (box, cls, dfl, angle) and sums
        # them in the trainer. Add the scalar feature regularizer to one channel
        # exactly once rather than broadcasting it across all four channels.
        total_loss = supervised_loss + self.pseudo_obb_weight * pseudo_loss
        if self.feature_consistency_weight > 0:
            total_loss = total_loss.clone()
            total_loss[0] = total_loss[0] + self.feature_consistency_weight * feature_loss
        if not torch.is_tensor(loss_items):
            loss_items = torch.as_tensor(loss_items, device=total_loss.device)
        extras = torch.stack((pseudo_loss.detach().sum(), feature_loss.detach(), total_loss.new_tensor(float(pseudo_count))))
        return total_loss, torch.cat((loss_items.detach().reshape(-1), extras))


class PseudoLabelEMADomainAdaptOBBTrainer(CheckpointSaveRetryMixin, EMADomainAdaptOBBTrainer):
    """Trainer that pairs unlabeled NAIP crops with pseudo-OBB supervision."""

    _CUSTOM_OPTIONS = frozenset(
        {
            "unlabeled_root",
            "unlabeled_catalog",
            "unlabeled_batch",
            "unlabeled_imgsz",
            "unlabeled_workers",
            "unlabeled_limit",
            "prefetch_factor",
            "pseudo_obb_weight",
            "pseudo_thresholds",
            "pseudo_iou_threshold",
            "pseudo_class_margin",
            "pseudo_max_det",
            "pseudo_warmup_epochs",
            "pseudo_rampup_epochs",
            "feature_consistency_weight",
            "feature_rampup_epochs",
            "foreground_feature_gain",
            "ema_decay",
        }
    )

    def _opt(self, name: str, default):
        return self._adapt_options.get(name, default)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = PseudoLabelEMADomainAdaptOBBModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data.get("channels", 3),
            verbose=verbose and RANK == -1,
            pseudo_obb_weight=float(self._opt("pseudo_obb_weight", 0.05)),
            feature_consistency_weight=float(self._opt("feature_consistency_weight", 0.0)),
            pseudo_thresholds=_thresholds(self._opt("pseudo_thresholds", None), int(self.data["nc"]), 0.9),
            pseudo_iou_threshold=float(self._opt("pseudo_iou_threshold", 0.5)),
            pseudo_class_margin=float(self._opt("pseudo_class_margin", 0.05)),
            pseudo_max_det=int(self._opt("pseudo_max_det", 100)),
            foreground_feature_gain=float(self._opt("foreground_feature_gain", 3.0)),
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        # Bypass the feature-only parent validator: it adds a consistency_loss
        # column that this model does not return.
        validator = OBBTrainer.get_validator(self)
        self.loss_names = (*self.loss_names, "pseudo_obb_loss", "feature_consistency_loss", "pseudo_boxes")
        return validator

    def _setup_train(self):
        # Parent attaches the EMA teacher, configures decay, and pairs loaders.
        # Its epoch callback dispatches to this class implementation below.
        super()._setup_train()

    def _update_consistency_weight(self, trainer):
        model = unwrap_model(trainer.model)
        epoch = float(getattr(trainer, "epoch", 0))
        pseudo_warmup = max(float(self._opt("pseudo_warmup_epochs", 5.0)), 0.0)
        pseudo_ramp = max(float(self._opt("pseudo_rampup_epochs", 10.0)), 1.0)
        feature_ramp = max(float(self._opt("feature_rampup_epochs", 10.0)), 1.0)
        if epoch < pseudo_warmup:
            pseudo_factor = 0.0
        else:
            pseudo_factor = min(1.0, (epoch - pseudo_warmup + 1.0) / pseudo_ramp)
        feature_factor = min(1.0, (epoch + 1.0) / feature_ramp)
        model.pseudo_obb_weight = float(self._opt("pseudo_obb_weight", 0.05)) * pseudo_factor
        model.feature_consistency_weight = float(self._opt("feature_consistency_weight", 0.0)) * feature_factor


__all__ = [
    "PseudoLabelEMADomainAdaptOBBModel",
    "PseudoLabelEMADomainAdaptOBBTrainer",
    "build_pseudo_batch",
    "foreground_feature_cosine_loss",
]




