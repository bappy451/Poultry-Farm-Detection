from __future__ import annotations

import torch

from scripts.ctq_obb_quality import QualityAwareOBBLoss


def test_quality_preprocess_preserves_per_box_values_without_mean():
    criterion = object.__new__(QualityAwareOBBLoss)
    criterion.device = torch.device("cpu")
    targets = torch.tensor([[0.0, 1.0, 0.1], [0.0, 1.0, 0.8], [1.0, 1.0, 0.4]])
    quality = criterion.preprocess_quality(targets, batch_size=2)
    assert torch.allclose(quality, torch.tensor([[0.1, 0.8], [0.4, 1.0]]))
