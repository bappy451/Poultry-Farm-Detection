from __future__ import annotations

import pytest

from scripts.preflight_ctq_full_grid import validate


def test_preflight_requires_cross_tile_manifest(tmp_path):
    data = tmp_path / "data.yaml"
    root = tmp_path / "tiles"
    model = tmp_path / "model.pt"
    data.write_text("nc: 2\n", encoding="utf-8")
    root.mkdir()
    model.write_bytes(b"model")
    config = {"data": str(data), "unlabeled_root": str(root), "model": str(model), "screen_seeds": [7], "confirmation_seeds": [7, 24, 42], "tile_manifest": str(tmp_path / "missing.csv"), "stages": [{"id": "cross", "requires_tile_manifest": True, "candidates": [{"x": 1}]}]}
    with pytest.raises(FileNotFoundError, match="manifest"):
        validate(config, require_tile_manifest=True)
    assert validate(config, require_tile_manifest=False)["candidates"] == 1
