from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """Multi-crop DINO cross-entropy loss with centering vector and temperature sharpening."""

    def __init__(
        self,
        out_dim: int = 65536,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 2,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def _get_teacher_temp(self, epoch: int) -> float:
        if epoch >= self.warmup_teacher_temp_epochs or self.warmup_teacher_temp_epochs <= 0:
            return self.teacher_temp
        slope = (self.teacher_temp - self.warmup_teacher_temp) / float(self.warmup_teacher_temp_epochs)
        return self.warmup_teacher_temp + slope * float(epoch)

    def forward(
        self,
        student_logits: Sequence[torch.Tensor],
        teacher_logits: Sequence[torch.Tensor],
        epoch: int = 0,
    ) -> tuple[torch.Tensor, float]:
        """Compute DINO loss across all student crops against teacher global crops.
        
        Returns:
          loss: Scalar tensor for backprop.
          prototype_entropy: float diagnostic metric.
        """
        n_student_crops = len(student_logits)
        n_teacher_crops = len(teacher_logits)
        if n_teacher_crops == 0 or n_student_crops == 0:
            raise ValueError("Student and Teacher logits must not be empty.")

        temp_t = self._get_teacher_temp(epoch)

        # Compute teacher probability distributions with centering (in FP32)
        teacher_probs: list[torch.Tensor] = []
        for t_logit in teacher_logits:
            # Force FP32 for softmax to avoid BF16 underflow with 65536 classes
            t_logit_fp32 = t_logit.detach().float()
            center_fp32 = self.center.float()
            centered = (t_logit_fp32 - center_fp32) / temp_t
            teacher_probs.append(F.softmax(centered, dim=-1))

        # Compute student log probabilities (in FP32)
        student_log_probs: list[torch.Tensor] = []
        for s_logit in student_logits:
            s_logit_fp32 = s_logit.float()
            student_log_probs.append(F.log_softmax(s_logit_fp32 / self.student_temp, dim=-1))

        total_loss = torch.zeros(1, device=student_logits[0].device, dtype=torch.float32)
        n_loss_terms = 0

        for j, t_prob in enumerate(teacher_probs):
            for i, s_log_prob in enumerate(student_log_probs):
                if i == j:
                    continue  # Do not match the exact same view index
                loss_ij = -(t_prob * s_log_prob).sum(dim=-1).mean()
                total_loss = total_loss + loss_ij
                n_loss_terms += 1

        if n_loss_terms > 0:
            total_loss = total_loss / float(n_loss_terms)

        # Update centering vector asynchronously
        self.update_center(teacher_logits)

        # Diagnostic entropy calculation on teacher distribution
        with torch.no_grad():
            t_prob_concat = torch.cat(teacher_probs, dim=0)
            entropy = -(t_prob_concat * torch.log(t_prob_concat + 1e-8)).sum(dim=-1).mean().item()

        # Cast loss back to original dtype for autograd if necessary, or just keep as fp32 (PyTorch allows fp32 loss in AMP)
        return total_loss.squeeze().to(student_logits[0].dtype), float(entropy)

    @torch.no_grad()
    def update_center(self, teacher_logits: Sequence[torch.Tensor]) -> None:
        """EMA update of the prototype centering vector."""
        # Keep the collapse-control center in FP32 and synchronize it when DDP is active.
        batch_center = torch.cat(teacher_logits, dim=0).mean(dim=0, keepdim=True).float()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(batch_center)
            batch_center.div_(torch.distributed.get_world_size())
        self.center.mul_(self.center_momentum).add_((1.0 - self.center_momentum) * batch_center)


class KoLeoLoss(nn.Module):
    """KoLeo regularization loss applied to L2-normalized batch embeddings."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute KoLeo loss over a batch of embeddings [B, D]."""
        if embeddings.shape[0] <= 1:
            return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype).squeeze()

        normed = F.normalize(embeddings, dim=-1, p=2)
        dist = torch.cdist(normed, normed, p=2.0)
        eye = torch.eye(dist.shape[0], device=dist.device, dtype=torch.bool)
        dist = dist.masked_fill(eye, float("inf"))
        min_dist, _ = dist.min(dim=-1)
        return -(torch.log(min_dist.clamp_min(self.eps))).mean()


class iBOTPatchLoss(nn.Module):
    """iBOT Masked Patch Prediction Loss for localized token-to-token semantic distillation."""

    def __init__(
        self,
        out_dim: int = 65536,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 2,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("patch_center", torch.zeros(1, out_dim))

    def _get_teacher_temp(self, epoch: int) -> float:
        if epoch >= self.warmup_teacher_temp_epochs or self.warmup_teacher_temp_epochs <= 0:
            return self.teacher_temp
        slope = (self.teacher_temp - self.warmup_teacher_temp) / float(max(1, self.warmup_teacher_temp_epochs))
        return self.warmup_teacher_temp + slope * float(epoch)

    def forward(
        self,
        student_patch_logits: torch.Tensor,
        teacher_patch_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
        epoch: int = 0,
    ) -> torch.Tensor:
        """Compute iBOT cross-entropy across masked patch tokens [B, N, K]."""
        if student_patch_logits.ndim != 3 or teacher_patch_logits.ndim != 3:
            return torch.zeros(1, device=student_patch_logits.device, dtype=student_patch_logits.dtype).squeeze()

        temp_t = self._get_teacher_temp(epoch)
        t_logits = teacher_patch_logits.detach()

        # Update patch centering vector
        with torch.no_grad():
            batch_center = t_logits.mean(dim=(0, 1), keepdim=True).float()
            self.patch_center.data = self.patch_center.data.float()
            self.patch_center.mul_(self.center_momentum).add_((1.0 - self.center_momentum) * batch_center)
            self.patch_center.data = self.patch_center.data.to(t_logits.dtype)

        # Force FP32 for softmax stability
        t_logits_fp32 = t_logits.float()
        patch_center_fp32 = self.patch_center.float()
        
        t_probs = F.softmax((t_logits_fp32 - patch_center_fp32) / temp_t, dim=-1)
        s_log_probs = F.log_softmax(student_patch_logits.float() / self.student_temp, dim=-1)

        loss_per_token = -(t_probs * s_log_probs).sum(dim=-1)  # [B, N]
        if mask is not None and mask.any():
            loss = (loss_per_token * mask.float()).sum() / (mask.float().sum() + 1e-8)
        else:
            loss = loss_per_token.mean()

        return loss.squeeze().to(student_patch_logits.dtype)


class CompositeDINOLoss(nn.Module):
    """DINO + KoLeo objective with an explicitly optional future iBOT pathway."""

    def __init__(
        self,
        out_dim: int = 65536,
        lambda_dino: float = 1.0,
        lambda_ibot: float = 0.0,
        lambda_koleo: float = 0.1,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 2,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.lambda_dino = lambda_dino
        self.lambda_ibot = lambda_ibot
        self.lambda_koleo = lambda_koleo
        self.dino_loss = DINOLoss(
            out_dim=out_dim,
            warmup_teacher_temp=warmup_teacher_temp,
            teacher_temp=teacher_temp,
            warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
            student_temp=student_temp,
            center_momentum=center_momentum,
        )
        self.ibot_loss = iBOTPatchLoss(
            out_dim=out_dim,
            warmup_teacher_temp=warmup_teacher_temp,
            teacher_temp=teacher_temp,
            warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
            student_temp=student_temp,
            center_momentum=center_momentum,
        )
        self.koleo_loss = KoLeoLoss()

    def forward(
        self,
        student_logits: Sequence[torch.Tensor],
        teacher_logits: Sequence[torch.Tensor],
        student_features: torch.Tensor | None = None,
        student_patch_logits: torch.Tensor | None = None,
        teacher_patch_logits: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        epoch: int = 0,
    ) -> dict[str, torch.Tensor | float]:
        """Compute composite DINO, iBOT, and KoLeo loss.
        
        Args:
          student_logits: Sequence of V student prototype logits [B, K].
          teacher_logits: Sequence of V_t teacher global prototype logits [B, K].
          student_features: Optional student backbone features [B, D] for KoLeo regularizer.
          student_patch_logits: Optional student patch logits [B, N, K] for iBOT.
          teacher_patch_logits: Optional teacher patch logits [B, N, K] for iBOT.
          mask: Optional boolean patch mask for iBOT loss.
          epoch: Current epoch index for temperature warm-up.
        """
        loss_dino, proto_entropy = self.dino_loss(student_logits, teacher_logits, epoch=epoch)

        loss_ibot = torch.zeros_like(loss_dino)
        if self.lambda_ibot > 0 and student_patch_logits is not None and teacher_patch_logits is not None:
            loss_ibot = self.ibot_loss(student_patch_logits, teacher_patch_logits, mask=mask, epoch=epoch)

        loss_koleo = torch.zeros_like(loss_dino)
        if self.lambda_koleo > 0 and student_features is not None:
            loss_koleo = self.koleo_loss(student_features)

        total_loss = (
            self.lambda_dino * loss_dino
            + self.lambda_ibot * loss_ibot
            + self.lambda_koleo * loss_koleo
        )

        return {
            "loss": total_loss,
            "dino_loss": loss_dino,
            "ibot_loss": loss_ibot,
            "koleo_loss": loss_koleo,
            "prototype_entropy": proto_entropy,
        }

