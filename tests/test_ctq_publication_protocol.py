from __future__ import annotations

from scripts.run_ctq_publication_protocol import METRIC, best_metric, load_config, summarize


def test_config_has_matched_publication_protocol() -> None:
    from pathlib import Path

    config = load_config(Path("configs/ctq_publication_protocol_v1.yaml"))
    assert config["seeds"] == [7, 17, 42]
    assert len(config["candidates"]) == 4
    assert config["control"]["pseudo_obb_weight"] == 0.0


def test_summary_compares_each_method_to_its_matched_baseline() -> None:
    rows = [
        {"method_id": "baseline", "seed": 7, "best_val_map50_95": 0.70},
        {"method_id": "baseline", "seed": 17, "best_val_map50_95": 0.60},
        {"method_id": "ctq", "seed": 7, "best_val_map50_95": 0.72},
        {"method_id": "ctq", "seed": 17, "best_val_map50_95": 0.63},
    ]
    summary = {row["method_id"]: row for row in summarize(rows)}
    assert summary["ctq"]["mean_delta_vs_baseline"] == 0.025
    assert METRIC == "metrics/mAP50-95(B)"
