import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDistillationLoss(nn.Module):
    """
    Computes the cosine similarity loss between the dense projected student features
    and the Vision Transformer's dense patch tokens.
    """
    def __init__(self):
        super().__init__()

    def forward(self, student_tokens: torch.Tensor, teacher_tokens: torch.Tensor) -> torch.Tensor:
        """
        student_tokens: [B, N, D]
        teacher_tokens: [B, N, D]
        """
        if student_tokens.shape != teacher_tokens.shape:
            raise ValueError(
                f"Shape mismatch: student {student_tokens.shape} vs teacher {teacher_tokens.shape}. "
                "Ensure image size is evenly divisible by patch size and dimensions align."
            )
            
        # Cosine similarity returns [B, N] values between -1 and 1
        # We want to maximize similarity (push to 1), so we minimize 1 - sim
        sim = F.cosine_similarity(student_tokens, teacher_tokens, dim=-1)
        loss = 1.0 - sim
        
        # Average across sequence length (N) and batch (B)
        return loss.mean()
