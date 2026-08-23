from __future__ import annotations

import torch

from scripts.ctq_reliable_obb import filter_pseudo_batch, stable_pseudo_mask


def _batch(boxes: list[list[float]]) -> dict[str, torch.Tensor]:
    count = len(boxes)
    return {
        "batch_idx": torch.zeros(count, dtype=torch.long),
        "cls": torch.zeros(count, dtype=torch.long),
        "bboxes": torch.tensor(boxes, dtype=torch.float32),
        "pseudo_conf": torch.ones(count),
    }


def test_stability_gate_keeps_only_agreeing_oriented_box() -> None:
    reference = _batch([[0.5, 0.5, 0.2, 0.1, 0.0], [0.1, 0.1, 0.1, 0.1, 0.0]])
    agreeing = _batch([[0.5, 0.5, 0.2, 0.1, 0.0]])
    mask = stable_pseudo_mask(reference, [agreeing], min_iou=0.70, min_agreement=1.0)
    assert mask.tolist() == [True, False]
    filtered = filter_pseudo_batch(reference, mask)
    assert filtered["bboxes"].shape[0] == 1
