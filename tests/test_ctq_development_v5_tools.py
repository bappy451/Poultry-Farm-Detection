from __future__ import annotations

import csv

import torch
from PIL import Image

from scripts.audit_ctq_unlabeled_leakage_v2 import audit
from scripts.create_ctq_source_disjoint_split_v2 import destination_name
from scripts.ctq_reliable_balanced_obb import pseudo_class_weights


def test_v2_audit_detects_exact_test_copy_and_source_collision(tmp_path) -> None:
    train = tmp_path / "train" / "images"
    val = tmp_path / "val" / "images"
    test = tmp_path / "test" / "images"
    pool = tmp_path / "pool" / "source_a"
    for folder, colour in ((train, "red"), (val, "green"), (test, "blue"), (pool, "blue")):
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), colour).save(folder / "tile.jpg")
    training_yaml = tmp_path / "training.yaml"
    final_yaml = tmp_path / "final.yaml"
    training_yaml.write_text(f"train: {train}\nval: {val}\n", encoding="utf-8")
    final_yaml.write_text(f"test: {test}\n", encoding="utf-8")
    source_csv = tmp_path / "sources.csv"
    with source_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "source_raster"))
        writer.writeheader()
        writer.writerow({"image_path": str((test / "tile.jpg").resolve()), "source_raster": "source_a"})
    report = audit(training_data=training_yaml, final_test_data=final_yaml, final_test_split="test", unlabeled_root=pool.parent, test_source_manifest=source_csv)
    assert report["inductive_test_safe"] is False
    assert report["test_source_ids_present_in_unlabeled_pool"] == ["source_a"]
    assert len(report["exact_unlabeled_final_test_matches"]) == 1


def test_v2_split_destination_names_are_extension_preserving_and_path_unique(tmp_path) -> None:
    first = tmp_path / "a" / "tile.jpg"
    second = tmp_path / "b" / "tile.jpg"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()
    assert destination_name(first).endswith(".jpg")
    assert destination_name(first) != destination_name(second)


def test_balanced_pseudo_weights_are_inverse_frequency_and_bounded() -> None:
    weights = pseudo_class_weights(torch.tensor([[0.0], [0.0], [0.0], [1.0]]), 2, minimum=0.5, maximum=2.0)
    assert torch.allclose(weights, torch.tensor([2 / 3, 2 / 3, 2 / 3, 2.0]))
    assert float(weights.min()) >= 0.5
    assert float(weights.max()) <= 2.0
