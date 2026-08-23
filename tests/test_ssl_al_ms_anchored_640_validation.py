from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts import run_ssl_al_ms_anchored_640_validation as protocol


def valid_config(tmp_path: Path) -> dict[str, object]:
    return {
        "study_name": "unit",
        "output_dir": str(tmp_path / "output"),
        "baseline": {"config": "baseline.yaml", "protocol_dir": "baseline"},
        "pretraining": {"config": "pretrain.yaml", "artifact": "artifact.pt"},
        "data": {"training": "training.yaml", "init_model": "model.pt"},
        "training": {
            "seeds": [7, 17, 42], "epochs": 100, "imgsz": 640,
            "batch": 16, "workers": 12, "device": "0", "lr0": 0.001,
            "weight_decay": 0.0005,
        },
        "selection": {
            "metric": protocol.METRIC, "validation_only": True,
            "minimum_mean_delta": 0.0, "report_all_seed_robustness": True,
        },
    }


def test_config_forbids_nested_final_test_key(tmp_path: Path) -> None:
    config = valid_config(tmp_path)
    config["data"]["final_test"] = "forbidden.yaml"  # type: ignore[index]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="forbids final-test keys"):
        protocol.read_config(path)


def test_cli_has_no_test_phase() -> None:
    with pytest.raises(SystemExit):
        protocol.parse_args(["--config", "config.yaml", "--phase", "test"])


def test_prepare_run_requires_exact_request_and_supports_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = valid_config(tmp_path)
    request = {"seed": 7, "epochs": 100}
    run_dir = tmp_path / "seed7"
    monkeypatch.setattr(protocol, "_ssl_complete", lambda *args, **kwargs: False)

    protocol.write_json(run_dir / "requested_run.json", {"seed": 17, "epochs": 100})
    with pytest.raises(ValueError, match="different settings"):
        protocol._prepare_run(run_dir, request, config=config, reuse=True)

    (run_dir / "requested_run.json").unlink()
    protocol.write_json(run_dir / "requested_run.json", request)
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "last.pt").write_bytes(b"partial")
    assert protocol._prepare_run(run_dir, request, config=config, reuse=True) == "resume"

    monkeypatch.setattr(protocol, "_ssl_complete", lambda *args, **kwargs: True)
    assert protocol._prepare_run(run_dir, request, config=config, reuse=True) == "reuse"
    with pytest.raises(FileExistsError, match="pass --reuse-existing"):
        protocol._prepare_run(run_dir, request, config=config, reuse=False)


def test_paired_statistics_uses_small_sample_t_interval() -> None:
    summary = protocol._paired_statistics([0.001, 0.003, 0.005])
    assert summary["n_pairs"] == 3
    assert summary["mean_delta"] == pytest.approx(0.003)
    assert summary["ci95_low"] < summary["mean_delta"] < summary["ci95_high"]


def test_summarize_writes_mean_and_all_seed_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = valid_config(tmp_path)
    output_dir = tmp_path / "output"
    ssl_values = {7: (40, 0.87), 17: (50, 0.86), 42: (60, 0.88)}
    baseline_values = {7: 0.86, 17: 0.861, 42: 0.87}
    baseline_rows = []
    for seed, value in baseline_values.items():
        checkpoint = tmp_path / f"baseline{seed}.pt"
        checkpoint.write_bytes(str(seed).encode())
        baseline_rows.append({
            "method": "baseline", "seed": seed, "best_epoch": 1,
            "best_val_map50_95": value, "checkpoint": str(checkpoint),
            "checkpoint_sha256": protocol.sha256(checkpoint),
        })
        ssl_checkpoint = output_dir / "anchored_ssl" / f"seed{seed}" / "weights" / "best.pt"
        ssl_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        ssl_checkpoint.write_bytes(f"ssl{seed}".encode())

    monkeypatch.setattr(protocol, "_ssl_complete", lambda *args, **kwargs: True)

    def metrics(run_dir: Path) -> dict[str, float | int]:
        seed = int(run_dir.name.removeprefix("seed"))
        epoch, value = ssl_values[seed]
        return {"best_epoch": epoch, protocol.METRIC: value}

    monkeypatch.setattr(protocol, "_best_csv_metrics", metrics)
    context = {
        "baseline_rows": baseline_rows,
        "training_data": tmp_path / "training.yaml",
        "init_model": tmp_path / "model.pt",
        "provenance": tmp_path / "provenance.json",
        "baseline_config_sha256": "baseline",
        "ssl_artifact_sha256": "artifact",
        "pretrain_config_sha256": "pretrain",
        "ms_ssl_manifest_sha256": "manifest",
        "runtime_versions": {"torch": "test", "ultralytics": "test"},
    }
    for path in (context["training_data"], context["init_model"], context["provenance"]):
        Path(path).write_text("x", encoding="utf-8")
    artifact = Path(config["pretraining"]["artifact"])  # type: ignore[index]
    artifact = tmp_path / artifact
    artifact.write_bytes(b"artifact")
    config["pretraining"]["artifact"] = str(artifact)  # type: ignore[index]

    frozen = protocol._summarize(config, context, output_dir, "config")
    assert frozen["anchored_ssl_beats_baseline_mean"] is True
    assert frozen["anchored_ssl_beats_baseline_all_seeds"] is False
    assert frozen["seed_wins"] == 2
    written = json.loads(
        (output_dir / "validation" / "frozen_selection.json").read_text(encoding="utf-8")
    )
    assert written["current_al_test_forbidden"] is True
    assert written["requires_new_untouched_geographic_holdout"] is True
