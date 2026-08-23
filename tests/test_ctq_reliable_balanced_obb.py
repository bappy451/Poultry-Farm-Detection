import torch

from scripts.ctq_reliable_balanced_obb import pseudo_class_weights


def test_pseudo_class_weights_balance_active_classes_and_are_bounded():
    weights = pseudo_class_weights(torch.tensor([[0.0], [0.0], [0.0], [1.0]]), 2, minimum=0.5, maximum=2.0)
    assert torch.allclose(weights, torch.tensor([0.5, 0.5, 0.5, 2.0]))


def test_pseudo_class_weights_rejects_invalid_class():
    try:
        pseudo_class_weights(torch.tensor([2]), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid pseudo class to fail")
