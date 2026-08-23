import pytest
import torch
import torch.nn as nn

import poultry_geo.spark as spark
from poultry_geo.spark import SparseTransformer


def test_mask_has_exact_patch_level_ratio_without_border_leakage():
    transformer = SparseTransformer(nn.Identity(), patch_size=4, mask_ratio=0.5)
    mask, patch_mask = transformer.generate_mask(2, 16, 16, torch.device("cpu"))
    assert mask.shape == (2, 1, 16, 16)
    assert patch_mask.shape == (2, 16)
    assert patch_mask.sum(dim=1).tolist() == [8.0, 8.0]


def test_invalid_mask_ratio_is_rejected():
    with pytest.raises(ValueError, match="mask_ratio"):
        SparseTransformer(nn.Identity(), mask_ratio=1.0)


def test_forward_fails_closed_without_sparse_runtime(monkeypatch):
    monkeypatch.setattr(spark, "spconv", None)
    transformer = SparseTransformer(nn.Identity(), patch_size=4, mask_ratio=0.5)
    with pytest.raises(RuntimeError, match="requires spconv"):
        transformer(torch.randn(1, 3, 16, 16))
