from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from scripts import evaluate_obb_seed_ensemble_validation as protocol


def valid_config(tmp_path: Path) -> dict[str, object]:
    return {
        "study_name": "unit",
        "output_dir": str(tmp_path / "output"),
        "data": {
            "training": "training.yaml",
            "training_yaml_sha256": "a" * 64,
            "provenance_sha256": "b" * 64,
        },
        "baseline": {
            "config": "baseline.yaml",
            "config_sha256": "c" * 64,
            "protocol_dir": "baseline",
            "protocol_snapshot": "snapshot.yaml",
            "validation_rows": "rows.csv",
            "validation_rows_sha256": "d" * 64,
            "validation_marker": "marker.json",
            "validation_marker_sha256": "e" * 64,
            "checkpoints": {
                str(seed): {
                    "path": f"seed{seed}.pt",
                    "sha256": str(seed)[0] * 64,
                    "best_val_map50_95": value,
                }
                for seed, value in ((7, 0.86479), (17, 0.86012), (42, 0.86413))
            },
        },
        "evaluation": {
            "split": "val", "imgsz": 640, "batch": 16, "workers": 12,
            "device": "0", "conf": 0.01, "iou": 0.7, "max_det": 300,
            "half": False, "augment": False, "plots": False,
        },
        "selection": {"metric": protocol.METRIC, "validation_only": True},
        "runtime": {
            "python": "3.12.13", "torch": "2.12.0+cu126",
            "ultralytics": "8.4.63", "cuda": "12.6", "cudnn": 91002,
            "gpu": "NVIDIA GeForce RTX 4090", "checkpoint_ultralytics": "8.4.63",
        },
        "reference": {
            "baseline_mean_map50_95": 0.8630133333333333,
            "best_single_map50_95": 0.86479,
            "ensemble_map50_95": 0.8660672572053741,
            "absolute_tolerance": 1e-6,
        },
    }


class DummyOBB(torch.nn.Module):
    def __init__(self, value: float, anchors: int) -> None:
        super().__init__()
        self.value = value
        self.anchors = anchors

    def forward(self, x: torch.Tensor, **kwargs):
        assert "embed" not in kwargs
        return torch.full((x.shape[0], 7, self.anchors), self.value), None


def test_compatibility_wrapper_accepts_empty_embed_and_concatenates_predictions() -> None:
    ensemble = protocol.CompatibleOBBValidationEnsemble()
    ensemble.extend([DummyOBB(1.0, 2), DummyOBB(2.0, 3), DummyOBB(3.0, 1)])
    prediction, auxiliary = ensemble(torch.zeros(2, 3, 32, 32), embed=[])
    assert auxiliary is None
    assert prediction.shape == (2, 7, 6)
    assert prediction[:, :, :2].eq(1.0).all()
    assert prediction[:, :, 2:5].eq(2.0).all()
    assert prediction[:, :, 5:].eq(3.0).all()


def test_compatibility_wrapper_rejects_embedding_extraction() -> None:
    ensemble = protocol.CompatibleOBBValidationEnsemble()
    ensemble.append(DummyOBB(1.0, 2))
    with pytest.raises(ValueError, match="does not support feature embeddings"):
        ensemble(torch.zeros(1, 3, 32, 32), embed=[5])


def test_config_forbids_test_keys_and_cli_has_no_phase(tmp_path: Path) -> None:
    config = valid_config(tmp_path)
    config["data"]["test"] = "forbidden.yaml"  # type: ignore[index]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="forbids test-related keys"):
        protocol.read_config(path)
    with pytest.raises(SystemExit):
        protocol.parse_args(["--config", "config.yaml", "--phase", "test"])


def test_config_freezes_nms_and_validation_settings(tmp_path: Path) -> None:
    config = valid_config(tmp_path)
    config["evaluation"]["iou"] = 0.5  # type: ignore[index]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation.iou must be 0.7"):
        protocol.read_config(path)


def test_comparison_rows_report_improvement_over_mean_and_best_single() -> None:
    context = {
        "baseline_mean": 0.8630133333333333,
        "best_single": 0.86479,
        "checkpoints": [
            {"seed": 7, "best_val_map50_95": 0.86479},
            {"seed": 17, "best_val_map50_95": 0.86012},
            {"seed": 42, "best_val_map50_95": 0.86413},
        ],
    }
    rows = protocol.comparison_rows(context, 0.8660672572)
    ensemble = next(row for row in rows if row["entry"] == "three_seed_ensemble")
    assert ensemble["delta_vs_baseline_mean"] == pytest.approx(0.003053923866666697)
    assert ensemble["delta_vs_best_single"] == pytest.approx(0.0012772572000000482)


def test_completed_result_requires_output_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    protocol.write_json(output / "ensemble_metrics.json", {"value": 1})
    (output / "comparison.csv").write_text("entry\nensemble\n", encoding="utf-8")
    protocol.write_json(
        output / "evaluation_complete.json",
        {
            "complete": True,
            "config_sha256": "cfg",
            "ensemble_metrics_sha256": protocol.sha256(output / "ensemble_metrics.json"),
            "comparison_sha256": protocol.sha256(output / "comparison.csv"),
        },
    )
    assert protocol._completed_result(output, "cfg") == {"value": 1}
    (output / "comparison.csv").write_text("changed\n", encoding="utf-8")
    protocol.sha256.cache_clear()
    assert protocol._completed_result(output, "cfg") is None
