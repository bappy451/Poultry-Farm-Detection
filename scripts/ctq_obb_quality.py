"""Per-box quality-aware extension of Ultralytics' native OBB loss."""

from __future__ import annotations

import torch

from ultralytics.utils.loss import make_anchors, v8OBBLoss


class QualityAwareOBBLoss(v8OBBLoss):
    """OBB criterion that applies teacher reliability before target reduction.

    ``batch['quality']`` is one reliability score per ground-truth/pseudo box,
    aligned with ``batch_idx``, ``cls`` and ``bboxes``.  It scales only
    positive assigned anchors; background anchors retain a zero target.
    """

    def preprocess_quality(self, targets: torch.Tensor, batch_size: int) -> torch.Tensor:
        if targets.numel() == 0:
            return torch.zeros(batch_size, 0, device=self.device)
        batch_idx = targets[:, 0].long()
        _, counts = batch_idx.unique(return_counts=True)
        out = torch.ones(batch_size, counts.max(), device=self.device, dtype=targets.dtype)
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within = torch.arange(len(targets), device=self.device) - offsets[batch_idx]
        out[batch_idx, within] = targets[:, -1]
        return out

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        loss = torch.zeros(4, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        pred_angle = preds["angle"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]
        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        batch_idx = batch["batch_idx"].view(-1, 1)
        quality = batch.get("quality")
        if quality is None:
            quality = torch.ones_like(batch_idx, dtype=torch.float32)
        quality = quality.view(-1, 1).to(device=self.device, dtype=torch.float32).clamp_(0.0, 1.0)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5), quality), 1)
        valid = (targets[:, 4] * float(imgsz[1]) >= 2) & (targets[:, 5] * float(imgsz[0]) >= 2)
        targets = targets[valid].to(self.device)
        # Reuse the version-pinned native OBB target layout, preserving target order.
        packed = self.preprocess(targets[:, :-1], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = packed.split((1, 5), 2)
        gt_quality = self.preprocess_quality(targets, batch_size)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)
        detached = pred_bboxes.detach().clone()
        detached[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), detached.type(gt_bboxes.dtype), anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )
        assigned_quality = gt_quality.gather(1, target_gt_idx).unsqueeze(-1) * fg_mask.unsqueeze(-1)
        target_scores = target_scores * assigned_quality.to(dtype=target_scores.dtype)
        target_scores_sum = target_scores.sum().clamp_min(1.0)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
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
