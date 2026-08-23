from __future__ import annotations

import pytest

from scripts.train_ctq_quality_obb import extract_quality_alpha


def test_extract_quality_alpha_preserves_base_arguments():
    alpha, remaining = extract_quality_alpha(["--quality-alpha", "2", "--data", "data.yaml"])
    assert alpha == 2.0
    assert remaining == ["--data", "data.yaml"]


def test_extract_quality_alpha_rejects_negative_value():
    with pytest.raises(ValueError, match="non-negative"):
        extract_quality_alpha(["--quality-alpha", "-1"])
