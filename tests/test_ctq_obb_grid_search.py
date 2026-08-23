from __future__ import annotations

import csv

import pytest

from scripts.run_ctq_obb_grid_search import METRIC, aggregate, best_validation_map, load_config


def test_best_validation_map_uses_peak_metric(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", METRIC])
        writer.writeheader()
        writer.writerows([{"epoch": 1, METRIC: 0.51}, {"epoch": 2, METRIC: 0.63}])
    assert best_validation_map(run_dir) == pytest.approx(0.63)


def test_aggregate_ranks_highest_mean_first():
    ranking = aggregate([
        {"candidate_id": "a", "best_val_map50_95": 0.70},
        {"candidate_id": "a", "best_val_map50_95": 0.72},
        {"candidate_id": "b", "best_val_map50_95": 0.74},
        {"candidate_id": "b", "best_val_map50_95": 0.76},
    ])
    assert [row["candidate_id"] for row in ranking] == ["b", "a"]
    assert ranking[0]["mean_best_val_map50_95"] == pytest.approx(0.75)


def test_load_config_rejects_duplicate_candidate_ids(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("""data: data.yaml\nunlabeled_root: unlabeled\nmodel: model.pt\noutput_dir: runs\nseeds: [7]\nbaseline: {}\ntraining: {}\ncandidates:\n  - {id: x, loss_family: pseudo, pseudo_obb_weight: 0.01, pseudo_thresholds: '0.9', feature_consistency_weight: 0}\n  - {id: x, loss_family: pseudo, pseudo_obb_weight: 0.01, pseudo_thresholds: '0.9', feature_consistency_weight: 0}\n""", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_config(path)
