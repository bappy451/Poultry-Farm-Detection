from __future__ import annotations

import csv

import numpy as np
from PIL import Image

from scripts.audit_ctq_unlabeled_leakage import audit
from scripts.create_ctq_source_disjoint_split import source_split
from scripts.diagnose_ctq_pseudo_labels import match_with_iou, score_curve


def test_oriented_diagnostic_matching_and_curve() -> None:
    polygon = np.asarray([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    records = match_with_iou([(0, 0.9, polygon)], [(0, polygon)], iou_threshold=0.5)
    curve = score_curve(records[0], target_count=1, threshold=0.8)
    assert curve["precision"] == 1.0
    assert curve["recall"] == 1.0
    assert curve["mean_matched_rotated_iou"] == 1.0


def test_source_partition_is_disjoint() -> None:
    groups = {"source_a": [__file__] * 4, "source_b": [__file__] * 3, "source_c": [__file__] * 2}
    test, val = source_split(groups, test_fraction=0.2, val_fraction=0.2, seed=7)
    assert test and val
    assert not test & val
    assert set(groups) - test - val


def test_leakage_audit_detects_exact_unlabeled_test_copy(tmp_path) -> None:
    train = tmp_path / "train" / "images"
    val = tmp_path / "val" / "images"
    test = tmp_path / "test" / "images"
    pool = tmp_path / "pool" / "source_a"
    for folder, colour in ((train, "red"), (val, "green"), (test, "blue"), (pool, "blue")):
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), colour).save(folder / "tile.jpg")
    training_yaml = tmp_path / "training.yaml"
    training_yaml.write_text(f"train: {train}\nval: {val}\n", encoding="utf-8")
    final_yaml = tmp_path / "final.yaml"
    final_yaml.write_text(f"test: {test}\n", encoding="utf-8")
    source_csv = tmp_path / "sources.csv"
    with source_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "source_raster"))
        writer.writeheader()
        writer.writerow({"image_path": str((test / "tile.jpg").resolve()), "source_raster": "source_a"})
    report = audit(
        training_data=training_yaml,
        final_test_data=final_yaml,
        final_test_split="test",
        unlabeled_root=pool.parent,
        test_source_manifest=source_csv,
    )
    assert len(report["exact_unlabeled_final_test_matches"]) == 1
    assert report["inductive_test_safe"] is False
