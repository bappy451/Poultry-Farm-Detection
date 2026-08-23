from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
import yaml

from scripts import run_yolo26x_obb_640_challenger as protocol
from scripts import train_yolo26x_obb_challenger as trainer


def load_real_config() -> dict[str, object]:
    return yaml.safe_load(
        (protocol.ROOT / "configs" / "yolo26x_obb_640_challenger_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_real_config_is_frozen_validation_only() -> None:
    config = protocol.read_config(
        protocol.ROOT / "configs" / "yolo26x_obb_640_challenger_v1.yaml"
    )
    assert config["training"]["optimizer"] == "MuSGD"
    assert config["training"]["batch"] == 8
    assert config["training"]["effective_nominal_batch"] == 64
    assert config["selection"]["validation_only"] is True
    assert protocol._forbidden_test_references(config) == []


@pytest.mark.parametrize(
    ("location", "value"),
    [
        ("data", {"final_test": "held.yaml"}),
        ("data", {"external": "temp/poultry_houses_test/data.yaml"}),
    ],
)
def test_config_rejects_test_keys_and_paths(
    tmp_path: Path, location: str, value: dict[str, str],
) -> None:
    config = load_real_config()
    config[location].update(value)  # type: ignore[union-attr]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="forbids test references"):
        protocol.read_config(path)


def test_cli_has_no_test_phase() -> None:
    with pytest.raises(SystemExit):
        protocol.parse_args(["--config", "config.yaml", "--phase", "test"])


class FakeResponse(io.BytesIO):
    status = 200

    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_official_weight_download_is_atomic_and_hash_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"pinned official bytes"
    target = tmp_path / "yolo26x-obb.pt"
    config = {
        "local_path": str(target),
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-obb.pt",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    monkeypatch.setattr(protocol.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse(payload))
    assert protocol.ensure_official_weight(config) == target
    assert target.read_bytes() == payload
    assert not list(tmp_path.glob("*.partial"))
    assert not list(tmp_path.glob("*.download.lock"))
    assert protocol.ensure_official_weight(config) == target


def test_existing_wrong_weight_is_never_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "yolo26x-obb.pt"
    target.write_bytes(b"wrong")
    config = {
        "local_path": str(target),
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-obb.pt",
        "size_bytes": 5,
        "sha256": hashlib.sha256(b"right").hexdigest(),
    }
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        protocol.ensure_official_weight(config)
    assert target.read_bytes() == b"wrong"


def test_train_args_lock_musgd_and_allow_only_local_resume_model(tmp_path: Path) -> None:
    data = tmp_path / "train.yaml"
    model = tmp_path / "yolo26x-obb.pt"
    run_dir = tmp_path / "run"
    expected = trainer.expected_train_args(
        data=data.resolve(), init_model=model.resolve(), seed=7, output_dir=run_dir.resolve()
    )
    trainer.audit_train_args(dict(expected), expected, run_dir=run_dir.resolve())
    resumed = dict(expected)
    resumed["model"] = str((run_dir / "weights" / "last.pt").resolve())
    trainer.audit_train_args(resumed, expected, run_dir=run_dir.resolve())
    wrong = dict(expected)
    wrong["optimizer"] = "AdamW"
    with pytest.raises(ValueError, match="optimizer"):
        trainer.audit_train_args(wrong, expected, run_dir=run_dir.resolve())


def test_prepare_run_requires_exact_request_and_verified_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = (tmp_path / "train.yaml").resolve()
    model = (tmp_path / "model.pt").resolve()
    data.write_text("train: x\nval: y\n", encoding="utf-8")
    model.write_bytes(b"model")
    context = {"training_data": data, "model_path": model}
    request = {"seed": 7, "contract": "one"}
    run_dir = tmp_path / "run"
    monkeypatch.setattr(protocol, "_challenger_complete", lambda *args, **kwargs: False)

    protocol.write_json(run_dir / "requested_run.json", {"seed": 17, "contract": "one"})
    with pytest.raises(ValueError, match="different settings"):
        protocol._prepare_run(run_dir, request, context=context, reuse=True)

    protocol.write_json(run_dir / "requested_run.json", request)
    expected = trainer.expected_train_args(
        data=data, init_model=model, seed=7, output_dir=run_dir.resolve()
    )
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "weights" / "last.pt").write_bytes(b"partial")
    (run_dir / "args.yaml").write_text(yaml.safe_dump(expected), encoding="utf-8")
    assert protocol._prepare_run(run_dir, request, context=context, reuse=True) == "resume"

    monkeypatch.setattr(protocol, "_challenger_complete", lambda *args, **kwargs: True)
    assert protocol._prepare_run(run_dir, request, context=context, reuse=True) == "reuse"
    with pytest.raises(FileExistsError, match="reuse-existing"):
        protocol._prepare_run(run_dir, request, context=context, reuse=False)


def test_command_uses_locked_training_entry_point(tmp_path: Path) -> None:
    config = load_real_config()
    context = {
        "training_data": tmp_path / "train.yaml", "training_data_sha256": "a" * 64,
        "model_path": tmp_path / "model.pt", "model_sha256": "b" * 64,
    }
    command = protocol._build_command(
        config, context, output_dir=tmp_path / "run", config_sha="c" * 64,
        seed=17, resume=True,
    )
    assert command[1:3] == ["-m", "scripts.train_yolo26x_obb_challenger"]
    assert command[command.index("--batch") + 1] == "8"
    assert command[command.index("--lr0") + 1] == "0.01"
    assert command[command.index("--momentum") + 1] == "0.9"
    assert command[-1] == "--resume"


def test_paired_statistics_use_small_sample_t_interval() -> None:
    summary = protocol._paired_statistics([0.001, 0.003, 0.005])
    assert summary["n_pairs"] == 3
    assert summary["mean_delta"] == pytest.approx(0.003)
    assert summary["ci95_low"] < summary["mean_delta"] < summary["ci95_high"]


def test_summarize_writes_paired_validation_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_real_config()
    config["output_dir"] = str(tmp_path / "output")
    output_dir = tmp_path / "output"
    baseline_values = {7: 0.86, 17: 0.861, 42: 0.87}
    challenger_values = {7: 0.87, 17: 0.860, 42: 0.88}
    baseline_rows = []
    for seed, value in baseline_values.items():
        checkpoint = tmp_path / f"baseline{seed}.pt"
        checkpoint.write_bytes(str(seed).encode())
        baseline_rows.append({
            "method": "yolo11l_baseline", "seed": seed, "best_epoch": 1,
            "best_val_map50_95": value, "checkpoint": str(checkpoint),
            "checkpoint_sha256": protocol.sha256(checkpoint),
        })
        challenger = output_dir / "yolo26x_musgd" / f"seed{seed}" / "weights" / "best.pt"
        challenger.parent.mkdir(parents=True, exist_ok=True)
        challenger.write_bytes(f"challenger{seed}".encode())

    monkeypatch.setattr(protocol, "_challenger_complete", lambda *args, **kwargs: True)

    def metrics(run_dir: Path) -> dict[str, float | int]:
        seed = int(run_dir.name.removeprefix("seed"))
        return {"best_epoch": seed, protocol.METRIC: challenger_values[seed]}

    monkeypatch.setattr(protocol, "_best_csv_metrics", metrics)
    context = {
        "baseline_rows": baseline_rows, "training_data": tmp_path / "train.yaml",
        "training_data_sha256": "a" * 64, "provenance_sha256": "b" * 64,
        "model_path": tmp_path / "yolo26x-obb.pt", "model_sha256": "c" * 64,
        "runtime": {"ultralytics": "8.4.63"},
        "model_identity": {"task": "obb", "scale": "x"},
    }
    Path(context["training_data"]).write_text("x", encoding="utf-8")
    Path(context["model_path"]).write_bytes(b"model")
    frozen = protocol._summarize(config, context, output_dir, "d" * 64)
    assert frozen["challenger_beats_baseline_mean"] is True
    assert frozen["challenger_beats_baseline_all_seeds"] is False
    assert frozen["seed_wins"] == 2
    written = json.loads(
        (output_dir / "validation" / "frozen_selection.json").read_text(encoding="utf-8")
    )
    assert written["test_resources_forbidden"] is True
    assert written["test_used"] is False
