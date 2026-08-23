"""Dense, multi-scale feature distillation losses."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillationLoss(nn.Module):
    """Cosine loss between spatial student features and resized DINO patch tokens."""

    def __init__(self, scale_weights: Mapping[str, float] | None = None) -> None:
        super().__init__()
        self.scale_weights = dict(scale_weights or {"p3": 1.0, "p4": 1.0, "p5": 1.0})

    @staticmethod
    def _tokens_to_grid(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Teacher tokens must be [B, N, D], got {tuple(tokens.shape)}.")
        side = int(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise ValueError(
                f"Teacher patch-token count must form a square grid, got N={tokens.shape[1]}."
            )
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)

    def forward(
        self, student_tokens: Mapping[str, torch.Tensor], teacher_tokens: torch.Tensor
    ) -> torch.Tensor:
        if not student_tokens:
            raise ValueError("Student did not produce any distillation feature maps.")
        teacher_grid = self._tokens_to_grid(teacher_tokens.float())
        weighted_loss = teacher_grid.new_zeros(())
        total_weight = 0.0
        for name, student in student_tokens.items():
            if name not in self.scale_weights:
                continue
            if student.ndim != 3 or student.shape[0] != teacher_grid.shape[0] or student.shape[2] != teacher_grid.shape[1]:
                raise ValueError(
                    f"Invalid {name} student tokens {tuple(student.shape)} for teacher {tuple(teacher_tokens.shape)}."
                )
            token_count = student.shape[1]
            side = int(token_count ** 0.5)
            if side * side != token_count:
                raise ValueError(f"Student {name} token count must form a square grid, got N={token_count}.")
            resized_teacher = F.interpolate(teacher_grid, size=(side, side), mode="bilinear", align_corners=False)
            target = resized_teacher.flatten(2).transpose(1, 2).to(dtype=student.dtype)
            scale_loss = 1.0 - F.cosine_similarity(student, target, dim=-1).mean()
            weight = float(self.scale_weights[name])
            weighted_loss = weighted_loss + weight * scale_loss
            total_weight += weight
        if total_weight <= 0:
            raise ValueError("No student feature names match the configured distillation scales.")
        return weighted_loss / total_weight
