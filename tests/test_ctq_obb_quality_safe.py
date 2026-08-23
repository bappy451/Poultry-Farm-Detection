from __future__ import annotations

import torch

from scripts.ctq_obb_quality_safe import SafeQualityAwareOBBLoss


def test_empty_batch_uses_native_loss_path(monkeypatch):
    criterion = object.__new__(SafeQualityAwareOBBLoss)
    called = False

    def fake_native(self, preds, batch):
        nonlocal called
        called = True
        return torch.zeros(4), torch.zeros(4)

    monkeypatch.setattr("scripts.ctq_obb_quality_safe.v8OBBLoss.loss", fake_native)
    result = criterion.loss({}, {"batch_idx": torch.zeros(0, dtype=torch.long)})
    assert called
    assert result[0].shape == (4,)
