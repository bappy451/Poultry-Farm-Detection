"""Reliability-gated CTQ with pseudo-only class balancing and background control.

Human labels continue to use the native OBB criterion unchanged.  These
controls apply only to accepted pseudo labels: rare predicted classes are
balanced within each unlabeled minibatch and the negative/background part of
the pseudo classification loss is conservatively downweighted.
"""

from __future__ import annotations

from typing import Any

import torch

from scripts.ctq_obb_quality_safe import SafeQualityAwareOBBLoss
from scripts.ctq_reliable_obb import ReliableCTQOBBModel, ReliableCTQOBBTrainer
from scripts.ema_pseudo_obb import PseudoLabelEMADomainAdaptOBBModel
from ultralytics.utils.loss import make_anchors


def pseudo_class_weights(classes: torch.Tensor, nc: int, *, minimum: float = 0.5, maximum: float = 2.0) -> torch.Tensor:
    """Return bounded inverse-frequency weights with unit mean over active classes."""
    if classes.numel() == 0:
        return classes.new_zeros((0,), dtype=torch.float32)
    if nc < 1 or minimum <= 0 or maximum < minimum:
        raise ValueError("Invalid pseudo-class weight configuration.")
    class_ids = classes.long().view(-1)
    if (class_ids < 0).any() or (class_ids >= nc).any():
        raise ValueError("Pseudo class IDs fall outside [0, nc).")
    counts = torch.bincount(class_ids, minlength=nc).to(dtype=torch.float32)
    active = counts > 0
    per_class = torch.ones(nc, device=classes.device, dtype=torch.float32)
    per_class[active] = counts[active].sum() / (active.sum() * counts[active])
    per_class = per_class.clamp(minimum, maximum)
    return per_class[class_ids]


class BalancedPseudoOBBLoss(SafeQualityAwareOBBLoss):
    """Quality OBB loss with pseudo-only inverse-frequency and background weights."""

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        if "class_weight" not in batch and "pseudo_background_weight" not in batch:
            return super().loss(preds, batch)
        if batch["batch_idx"].numel() == 0:
            return super().loss(preds, batch)
        loss = torch.zeros(4, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        pred_angle = preds["angle"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size, dtype = pred_angle.shape[0], pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        batch_idx = batch["batch_idx"].view(-1, 1)
        quality = batch.get("quality", torch.ones_like(batch_idx, dtype=torch.float32)).view(-1, 1).to(self.device, torch.float32).clamp(0.0, 1.0)
        class_weight = batch.get("class_weight", torch.ones_like(quality)).view(-1, 1).to(self.device, torch.float32).clamp_min(0.0)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5), quality, class_weight), 1)
        valid = (targets[:, 4] * float(imgsz[1]) >= 2) & (targets[:, 5] * float(imgsz[0]) >= 2)
        targets = targets[valid].to(self.device)
        if targets.numel() == 0:
            return super().loss(preds, {key: value[:0] if torch.is_tensor(value) and value.ndim and value.shape[0] == batch_idx.shape[0] else value for key, value in batch.items()})
        packed = self.preprocess(targets[:, :-2], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = packed.split((1, 5), 2)
        gt_quality = self.preprocess_quality(torch.cat((targets[:, :-1], targets[:, -2: -1]), 1), batch_size)
        gt_class_weight = self.preprocess_quality(torch.cat((targets[:, :-1], targets[:, -1:]), 1), batch_size)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)
        detached = pred_bboxes.detach().clone()
        detached[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(pred_scores.detach().sigmoid(), detached.type(gt_bboxes.dtype), anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        reliability = gt_quality.gather(1, target_gt_idx) * gt_class_weight.gather(1, target_gt_idx)
        target_scores = target_scores * (reliability.unsqueeze(-1) * fg_mask.unsqueeze(-1)).to(target_scores.dtype)
        target_scores_sum = target_scores.sum().clamp_min(1.0)
        bce = self.bce(pred_scores, target_scores.to(dtype))
        background_weight = float(batch.get("pseudo_background_weight", 1.0))
        if not 0.0 < background_weight <= 1.0:
            raise ValueError("pseudo_background_weight must be in (0, 1].")
        bce = torch.where(fg_mask.unsqueeze(-1), bce, bce * background_weight)
        loss[1] = bce.sum() / target_scores_sum
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor)
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum)
        else:
            loss[0] += (pred_angle * 0).sum()
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        loss[3] *= self.hyp.angle
        return loss * batch_size, loss.detach()


class BalancedReliableCTQOBBModel(ReliableCTQOBBModel):
    def __init__(self, *args, pseudo_background_weight: float = 0.5, pseudo_class_weight_min: float = 0.5, pseudo_class_weight_max: float = 2.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 < pseudo_background_weight <= 1.0:
            raise ValueError("pseudo_background_weight must be in (0, 1].")
        if pseudo_class_weight_min <= 0 or pseudo_class_weight_max < pseudo_class_weight_min:
            raise ValueError("Invalid pseudo class-weight bounds.")
        self.pseudo_background_weight = float(pseudo_background_weight)
        self.pseudo_class_weight_min = float(pseudo_class_weight_min)
        self.pseudo_class_weight_max = float(pseudo_class_weight_max)

    def init_criterion(self):
        return BalancedPseudoOBBLoss(self)

    def loss(self, batch: dict[str, Any], preds=None):
        supervised_loss, loss_items = super(PseudoLabelEMADomainAdaptOBBModel, self).loss(batch, preds)
        weak, strong = batch.get("unlabeled_img_weak"), batch.get("unlabeled_img_strong")
        pseudo_loss, feature_loss = torch.zeros_like(supervised_loss), supervised_loss.new_zeros(())
        pseudo_count = 0
        if weak is not None and strong is not None and (self.pseudo_obb_weight > 0 or self.feature_consistency_weight > 0):
            pseudo_batch, pseudo_count = self._pseudo_targets(weak)
            if self.pseudo_obb_weight > 0 and pseudo_count > 0:
                pseudo_batch = dict(pseudo_batch)
                pseudo_batch["quality"] = pseudo_batch["pseudo_conf"].pow(self.quality_alpha).detach()
                pseudo_batch["class_weight"] = pseudo_class_weights(pseudo_batch["cls"], int(self.nc), minimum=self.pseudo_class_weight_min, maximum=self.pseudo_class_weight_max).detach()
                pseudo_batch["pseudo_background_weight"] = self.pseudo_background_weight
                pseudo_loss, _ = self.criterion(self.forward(strong), pseudo_batch)
            if self.feature_consistency_weight > 0:
                from scripts.ema_pseudo_obb import foreground_feature_cosine_loss
                from poultry_geo.ema_adapt import _backbone_features
                teacher = object.__getattribute__(self, "_ema_teacher")
                with torch.no_grad():
                    teacher_features = _backbone_features(teacher, weak)
                feature_loss = foreground_feature_cosine_loss(_backbone_features(self, strong), teacher_features, pseudo_batch, foreground_gain=self.foreground_feature_gain)
        total_loss = supervised_loss + self.pseudo_obb_weight * pseudo_loss
        if self.feature_consistency_weight > 0:
            total_loss = total_loss.clone()
            total_loss[0] += self.feature_consistency_weight * feature_loss
        if not torch.is_tensor(loss_items):
            loss_items = torch.as_tensor(loss_items, device=total_loss.device)
        extras = torch.stack((pseudo_loss.detach().sum(), feature_loss.detach(), total_loss.new_tensor(float(pseudo_count))))
        return total_loss, torch.cat((loss_items.detach().reshape(-1), extras))


class BalancedReliableCTQOBBTrainer(ReliableCTQOBBTrainer):
    _CUSTOM_OPTIONS = ReliableCTQOBBTrainer._CUSTOM_OPTIONS | frozenset({"pseudo_background_weight", "pseudo_class_weight_min", "pseudo_class_weight_max"})

    def get_model(self, cfg=None, weights=None, verbose=True):
        from scripts.ema_pseudo_obb import _thresholds
        model = BalancedReliableCTQOBBModel(
            cfg, nc=self.data["nc"], ch=self.data.get("channels", 3), verbose=verbose,
            pseudo_obb_weight=float(self._opt("pseudo_obb_weight", 0.05)), feature_consistency_weight=float(self._opt("feature_consistency_weight", 0.0)),
            pseudo_thresholds=_thresholds(self._opt("pseudo_thresholds", None), int(self.data["nc"]), 0.9), pseudo_iou_threshold=float(self._opt("pseudo_iou_threshold", 0.5)), pseudo_class_margin=float(self._opt("pseudo_class_margin", 0.05)), pseudo_max_det=int(self._opt("pseudo_max_det", 100)), foreground_feature_gain=float(self._opt("foreground_feature_gain", 3.0)), quality_alpha=float(self._opt("quality_alpha", 1.0)),
            teacher_jitter_views=int(self._opt("teacher_jitter_views", 2)), teacher_jitter_iou=float(self._opt("teacher_jitter_iou", 0.70)), teacher_jitter_agreement=float(self._opt("teacher_jitter_agreement", 1.0)),
            pseudo_background_weight=float(self._opt("pseudo_background_weight", 0.5)), pseudo_class_weight_min=float(self._opt("pseudo_class_weight_min", 0.5)), pseudo_class_weight_max=float(self._opt("pseudo_class_weight_max", 2.0)),
        )
        if weights:
            model.load(weights)
        return model
