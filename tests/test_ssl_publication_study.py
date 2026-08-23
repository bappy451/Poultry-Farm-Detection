from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.run_ssl_publication_study import (
    METRIC_FIELDS,
    Study,
    best_validation_metrics,
    has_complete_test_matrix,
    select_candidate,
    write_test_reports,
)


def _result_row(epoch: int, score: float, precision: float = 0.9) -> dict[str, object]:
    return {
        "epoch": epoch,
        "metrics/precision(B)": precision,
        "metrics/recall(B)": 0.8,
        "metrics/mAP50(B)": 0.9,
        "metrics/mAP50-95(B)": score,
    }


def test_best_validation_metrics_uses_best_epoch(tmp_path: Path) -> None:
    run_dir = tmp_path / "pseudo_finetune"
    run_dir.mkdir()
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_result_row(1, 0.1)))
        writer.writeheader()
        writer.writerows([_result_row(1, 0.71), _result_row(2, 0.74), _result_row(3, 0.72)])

    result = best_validation_metrics(run_dir)

    assert result["best_epoch"] == 2
    assert result["metrics/mAP50-95(B)"] == 0.74


def test_select_candidate_excludes_diagnostic_controls() -> None:
    result = select_candidate(
        [
            {"candidate": "ema_no_pseudo", "eligible_for_selection": False, "metrics/mAP50-95(B)": 0.99},
            {"candidate": "weight_002", "eligible_for_selection": True, "metrics/mAP50-95(B)": 0.74},
            {"candidate": "weight_005", "eligible_for_selection": True, "metrics/mAP50-95(B)": 0.76},
        ]
    )

    assert result["candidate"] == "weight_005"


def test_complete_test_matrix_requires_every_method_and_seed() -> None:
    rows = [
        {"method": method, "seed": seed}
        for method in ("ultralytics_burnin", "pseudo_selected")
        for seed in (7, 24, 42)
    ]

    assert has_complete_test_matrix(
        rows, methods={"ultralytics_burnin", "pseudo_selected"}, seeds={7, 24, 42}
    )
    assert not has_complete_test_matrix(
        rows[:-1], methods={"ultralytics_burnin", "pseudo_selected"}, seeds={7, 24, 42}
    )


def test_write_test_reports_writes_aggregate_and_paired_outputs(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for seed, baseline, pseudo in ((0, 0.70, 0.72), (7, 0.73, 0.74)):
        for method, score in (("ultralytics_burnin", baseline), ("pseudo_weight_005", pseudo)):
            rows.append(
                {
                    "stage": "confirmation_10pct",
                    "fraction_percent": 10,
                    "method": method,
                    "seed": seed,
                    "checkpoint": f"{method}_{seed}.pt",
                    "test_data": "test.yaml",
                    "source": "test",
                    **{
                        metric: score if metric == "metrics/mAP50-95(B)" else 0.8
                        for metric in METRIC_FIELDS
                    },
                }
            )

    write_test_reports(rows, output_dir=tmp_path, selected_method="pseudo_weight_005")

    assert (tmp_path / "per_seed_external_test_metrics.csv").is_file()
    assert (tmp_path / "aggregate_external_test_metrics.csv").is_file()
    with (tmp_path / "paired_ssl_minus_burnin.csv").open(encoding="utf-8", newline="") as handle:
        paired = list(csv.DictReader(handle))
    assert [float(row["metrics/mAP50-95(B)"]) for row in paired] == pytest.approx([0.02, 0.01])


def test_preflight_rejects_unlabeled_crop_larger_than_declared_tiles(tmp_path: Path) -> None:
    config = {
        "data": {
            "unlabeled_min_side": 512,
            "subset_config_dir": "configs/poultry_clean_label_subsets",
            "final_test_data": "temp/poultry_houses_test/data.yaml",
            "unlabeled_root": "E:\\NAIP_Dataset_PH\\MS_tiles_512_256_jpg",
        },
        "training": {"unlabeled_imgsz": 448},
        "calibration": {},
        "execution": {},
    }
    study = Study(tmp_path / "study.yaml", config, tmp_path / "runs", plots=False)

    with pytest.raises(ValueError, match="exceeds data.unlabeled_min_side=512"):
        study.validate_unlabeled_crop_sizes(
            {"candidates": [{"name": "invalid_u640", "unlabeled_imgsz": 640}]}
        )