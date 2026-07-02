from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from poultry_geo.mississippi_data import (
    OverlapAwareNAIPDataset,
    compute_patch_indices_for_overlap,
)
from poultry_geo.mississippi_adjacency import write_adjacency_index_csv


def _write_tile(path: Path, *, size: tuple[int, int] = (512, 512), value: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(value, value, value)).save(path)


def _write_demo_tiles(root: Path) -> Path:
    _write_tile(root / "raster_a__row_0__col_0.jpg", value=40)
    _write_tile(root / "raster_a__row_0__col_256.jpg", value=60)
    _write_tile(root / "raster_b__row_0__col_0.jpg", value=80)
    _write_tile(root / "raster_b__row_0__col_256.jpg", value=100)
    output_csv = root / "adjacency.csv"
    write_adjacency_index_csv(root, output_csv, overlap_ratio=0.5, tile_size=512)
    return output_csv


def test_compute_patch_indices_for_overlap_matches_patch_grid_geometry() -> None:
    indices = compute_patch_indices_for_overlap(
        overlap_box=(128, 0, 256, 256),
        image_size=(256, 256),
        patch_size=16,
    )

    assert indices[0] == 8
    assert indices[-1] == 255
    assert len(indices) == 128


def test_overlap_aware_dataset_keeps_raster_ids_split_cleanly(tmp_path: Path) -> None:
    adjacency_csv = _write_demo_tiles(tmp_path)

    train_dataset = OverlapAwareNAIPDataset(
        tmp_path,
        adjacency_csv,
        split="train",
        val_fraction=0.5,
        seed=27,
        augment=False,
        high_res=True,
        epoch_size=4,
    )
    val_dataset = OverlapAwareNAIPDataset(
        tmp_path,
        adjacency_csv,
        split="val",
        val_fraction=0.5,
        seed=27,
        augment=False,
        high_res=True,
        epoch_size=4,
    )

    assert train_dataset.active_raster_ids
    assert val_dataset.active_raster_ids
    assert train_dataset.active_raster_ids.isdisjoint(val_dataset.active_raster_ids)


def test_overlap_aware_dataset_emits_expected_pair_schedule_and_overlap_fields(tmp_path: Path) -> None:
    adjacency_csv = _write_demo_tiles(tmp_path)
    dataset = OverlapAwareNAIPDataset(
        tmp_path,
        adjacency_csv,
        split="train",
        val_fraction=0.5,
        seed=11,
        augment=False,
        high_res=True,
        epoch_size=4,
    )

    pair_types = [dataset[index]["pair_type"] for index in range(4)]
    assert pair_types == ["overlap", "overlap", "same_tile", "random"]

    overlap_sample = dataset[0]
    assert overlap_sample["image_a"].shape == (3, 512, 512)
    assert overlap_sample["image_b"].shape == (3, 512, 512)
    assert overlap_sample["pair_type"] == "overlap"
    assert overlap_sample["overlap_box_a"] == (256, 0, 512, 512)
    assert overlap_sample["patch_indices_a"].numel() == overlap_sample["patch_indices_b"].numel()
    assert overlap_sample["patch_indices_a"].numel() > 0


def test_overlap_aware_dataset_supports_low_resolution_adaptation(tmp_path: Path) -> None:
    adjacency_csv = _write_demo_tiles(tmp_path)
    dataset = OverlapAwareNAIPDataset(
        tmp_path,
        adjacency_csv,
        split="train",
        val_fraction=0.5,
        seed=5,
        augment=False,
        high_res=False,
        epoch_size=1,
    )

    sample = dataset[0]
    assert sample["image_a"].shape[1:] in {(256, 256), (320, 320)}
    assert sample["crop_origin_a"][0] >= 0
    assert sample["crop_origin_a"][1] >= 0
