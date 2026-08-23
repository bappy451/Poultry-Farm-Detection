"""Run a validation-only anchored-SSL follow-up at 640x640.

This protocol reuses, but independently verifies, the completed full-AL
YOLO11l-OBB 640 baseline and the completed v3 MS-only anchored-DINOv2
artifact. It trains only the anchored-SSL arm for seeds 7/17/42. The current
AL test split is deliberately unavailable: confirming this follow-up requires
a new untouched geographic holdout.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
METRIC = "metrics/mAP50-95(B)"
SEEDS = [7, 17, 42]
RUNTIME_FIELDS = (
    "epochs", "imgsz", "batch", "workers", "device", "lr0", "weight_decay"
)


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


@lru_cache(maxsize=None)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, Any], *, compact: bool = True) -> str:
    kwargs: dict[str, Any] = {"sort_keys": True}
    if compact:
        kwargs["separators"] = (",", ":")
    return hashlib.sha256(json.dumps(payload, **kwargs).encode("utf-8")).hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return payload


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _forbidden_final_test_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            location = f"{prefix}.{key}" if prefix else key
            normalized = key.lower().replace("-", "_").replace(" ", "_")
            if "final_test" in normalized:
                found.append(location)
            found.extend(_forbidden_final_test_keys(child, location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_final_test_keys(child, f"{prefix}[{index}]"))
    return found


def read_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    forbidden = _forbidden_final_test_keys(config)
    if forbidden:
        raise ValueError(
            "Validation-only follow-up forbids final-test keys: " + ", ".join(forbidden)
        )
    required = {"output_dir", "baseline", "pretraining", "data", "training", "selection"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Protocol config is missing {sorted(missing)}.")
    for name in ("baseline", "pretraining", "data", "training", "selection"):
        if not isinstance(config[name], dict):
            raise ValueError(f"{name} must be a mapping.")
    required_nested = {
        "baseline": {"config", "protocol_dir"},
        "pretraining": {"config", "artifact"},
        "data": {"training", "init_model"},
        "training": set(RUNTIME_FIELDS) | {"seeds"},
    }
    for name, names in required_nested.items():
        absent = names.difference(config[name])
        if absent:
            raise ValueError(f"{name} is missing {sorted(absent)}.")
    training = config["training"]
    seeds = [int(value) for value in training["seeds"]]
    if seeds != SEEDS or len(set(seeds)) != len(seeds):
        raise ValueError("The frozen comparison requires seeds [7, 17, 42].")
    if int(training["imgsz"]) != 640 or int(training["batch"]) != 16:
        raise ValueError("The matched follow-up requires imgsz=640 and batch=16.")
    if int(training["epochs"]) != 100:
        raise ValueError("The matched follow-up requires exactly 100 epochs.")
    if int(training["workers"]) < 0:
        raise ValueError("workers cannot be negative.")
    if float(training["lr0"]) <= 0 or float(training["weight_decay"]) < 0:
        raise ValueError("lr0 must be positive and weight_decay cannot be negative.")
    selection = config["selection"]
    if selection.get("validation_only") is not True:
        raise ValueError("selection.validation_only must be true.")
    if selection.get("metric") != METRIC:
        raise ValueError(f"selection.metric must be {METRIC}.")
    if float(selection.get("minimum_mean_delta", 0.0)) < 0:
        raise ValueError("selection.minimum_mean_delta cannot be negative.")
    if selection.get("report_all_seed_robustness") is not True:
        raise ValueError("selection.report_all_seed_robustness must be true.")
    return config


def _same_number(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def _validate_training_data(data_yaml: Path) -> Path:
    # Reuse the stricter full-AL audit, including exact-content and source-group
    # train/validation disjointness checks.
    from scripts.run_full_al_640_baseline import _validate_training_data as audit

    return audit(data_yaml)


def _best_csv_metrics(run_dir: Path) -> dict[str, float | int]:
    from scripts.run_ssl_publication_study import best_validation_metrics

    values = best_validation_metrics(run_dir)
    for name, value in values.items():
        if name != "best_epoch" and not math.isfinite(float(value)):
            raise ValueError(f"Non-finite validation metric {name} in {run_dir}.")
    return values


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    from ultralytics.nn.tasks import torch_safe_load

    checkpoint, _ = torch_safe_load(str(path))
    try:
        train_args = checkpoint.get("train_args")
        train_metrics = checkpoint.get("train_metrics")
        if not isinstance(train_args, dict) or not isinstance(train_metrics, dict):
            raise ValueError(f"Checkpoint lacks train_args/train_metrics: {path}")
        return {
            "version": str(checkpoint.get("version", "")),
            "train_args": dict(train_args),
            "train_metrics": dict(train_metrics),
        }
    finally:
        del checkpoint
        gc.collect()


def _audit_train_args(
    actual: dict[str, Any], *, training: dict[str, Any], data: Path,
    init_model: Path, seed: int, method: str,
) -> None:
    exact = {
        "task": "obb", "data": str(data), "model": str(init_model),
        "epochs": int(training["epochs"]), "batch": int(training["batch"]),
        "imgsz": int(training["imgsz"]), "workers": int(training["workers"]),
        "device": str(training["device"]), "seed": int(seed),
        "optimizer": "AdamW", "deterministic": True, "cos_lr": True,
        "close_mosaic": 10, "amp": True, "val": True, "fraction": 1.0,
    }
    for name, expected in exact.items():
        observed = actual.get(name)
        if isinstance(expected, bool):
            matches = observed is expected
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            matches = _same_number(observed, expected)
        elif name in {"data", "model"}:
            matches = Path(str(observed)).resolve() == Path(str(expected)).resolve()
        else:
            matches = str(observed) == str(expected)
        if not matches:
            raise ValueError(
                f"{method} seed {seed} train arg {name!r} is {observed!r}; expected {expected!r}."
            )
    numeric = {
        "lr0": float(training["lr0"]), "lrf": 0.01,
        "weight_decay": float(training["weight_decay"]), "warmup_epochs": 3.0,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 15.0,
        "flipud": 0.5, "fliplr": 0.5, "mosaic": 1.0,
    }
    for name, expected in numeric.items():
        if not _same_number(actual.get(name), expected):
            raise ValueError(
                f"{method} seed {seed} train arg {name!r} is {actual.get(name)!r}; expected {expected}."
            )
    pretrained = actual.get("pretrained")
    if method == "baseline":
        if Path(str(pretrained)).resolve() != init_model.resolve():
            raise ValueError(f"Baseline seed {seed} did not use the frozen init model.")
    elif pretrained is not True:
        raise ValueError(f"Anchored-SSL seed {seed} must preserve its injected in-memory backbone.")


def _audit_checkpoint(
    checkpoint: Path, run_dir: Path, *, training: dict[str, Any], data: Path,
    init_model: Path, seed: int, method: str,
) -> dict[str, float | int]:
    import ultralytics

    metrics = _best_csv_metrics(run_dir)
    metadata = _checkpoint_metadata(checkpoint)
    if metadata["version"] != str(ultralytics.__version__):
        raise ValueError(
            f"{method} seed {seed} used Ultralytics {metadata['version']}; "
            f"current runtime is {ultralytics.__version__}."
        )
    _audit_train_args(
        metadata["train_args"], training=training, data=data,
        init_model=init_model, seed=seed, method=method,
    )
    checkpoint_metric = metadata["train_metrics"].get(METRIC)
    if not _same_number(checkpoint_metric, metrics[METRIC], tolerance=5e-6):
        raise ValueError(
            f"{checkpoint} stores {checkpoint_metric!r}, but its best CSV {METRIC} is {metrics[METRIC]!r}."
        )
    return metrics


def _matched_training(followup: dict[str, Any], baseline: dict[str, Any]) -> None:
    left = followup["training"]
    right = baseline.get("training")
    if not isinstance(right, dict):
        raise ValueError("Baseline config has no training mapping.")
    if [int(value) for value in right.get("seeds", [])] != SEEDS:
        raise ValueError("Frozen baseline seeds do not match [7, 17, 42].")
    for name in RUNTIME_FIELDS:
        if name in {"device"}:
            matches = str(left[name]) == str(right.get(name))
        else:
            matches = _same_number(left[name], right.get(name))
        if not matches:
            raise ValueError(
                f"Follow-up training.{name}={left[name]!r} does not match baseline {right.get(name)!r}."
            )


def _expected_baseline_request(
    baseline_hash: str, baseline: dict[str, Any], data: Path,
    init_model: Path, seed: int,
) -> dict[str, Any]:
    training = baseline["training"]
    return {
        "config_sha256": baseline_hash, "method": "baseline", "seed": seed,
        "data": str(data), "init_model": str(init_model),
        "epochs": int(training["epochs"]), "imgsz": int(training["imgsz"]),
        "batch": int(training["batch"]), "workers": int(training["workers"]),
        "device": str(training["device"]), "lr0": float(training["lr0"]),
        "weight_decay": float(training["weight_decay"]),
    }


def _validate_frozen_baseline(
    config: dict[str, Any], *, data: Path, init_model: Path,
) -> tuple[list[dict[str, Any]], str]:
    source = resolve(config["baseline"]["config"])
    protocol_dir = resolve(config["baseline"]["protocol_dir"])
    _require_file(source)
    snapshot = protocol_dir / "protocol_config.yaml"
    _require_file(snapshot)
    if source.read_text(encoding="utf-8") != snapshot.read_text(encoding="utf-8"):
        raise ValueError("Frozen baseline source config differs from its protocol snapshot.")
    baseline = read_yaml(source)
    if resolve(baseline.get("output_dir", "")) != protocol_dir:
        raise ValueError("Frozen baseline config output_dir does not match baseline.protocol_dir.")
    _matched_training(config, baseline)
    baseline_data = resolve(baseline.get("data", {}).get("training", ""))
    baseline_init = resolve(baseline.get("data", {}).get("init_model", ""))
    if baseline_data != data or baseline_init != init_model:
        raise ValueError("Frozen baseline data or init model does not match the follow-up.")
    baseline_hash = canonical_hash(baseline, compact=True)
    completion = read_json(protocol_dir / "validation_complete.json")
    if (
        completion.get("config_sha256") != baseline_hash
        or completion.get("final_test_used") is not False
        or completion.get("selection_data") != "AL validation only"
        or [int(value) for value in completion.get("seeds", [])] != SEEDS
        or int(completion.get("imgsz", 0)) != 640
        or int(completion.get("batch", 0)) != 16
    ):
        raise ValueError("Frozen baseline validation marker does not match its config.")
    recorded = completion.get("checkpoints")
    if not isinstance(recorded, list) or len(recorded) != len(SEEDS):
        raise ValueError("Frozen baseline validation marker has incomplete checkpoints.")
    recorded_by_seed = {int(row["seed"]): row for row in recorded if isinstance(row, dict)}
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        run_dir = protocol_dir / "baseline" / f"seed{seed}"
        request_path = run_dir / "requested_run.json"
        marker_path = run_dir / "supervised_baseline.json"
        args_path = run_dir / "args.yaml"
        checkpoint = run_dir / "weights" / "best.pt"
        for path in (request_path, marker_path, args_path, run_dir / "results.csv", checkpoint):
            _require_file(path)
        expected = _expected_baseline_request(baseline_hash, baseline, data, init_model, seed)
        if read_json(request_path) != expected:
            raise ValueError(f"Frozen baseline seed {seed} request does not match its config.")
        marker = read_json(marker_path)
        if (
            marker.get("pipeline") != "supervised_yolo_obb_burn_in"
            or marker.get("test_used_for_selection") is not False
            or int(marker.get("seed", -1)) != seed
            or int(marker.get("epochs", -1)) != 100
            or Path(str(marker.get("best_checkpoint", ""))).resolve() != checkpoint.resolve()
            or Path(str(marker.get("data", ""))).resolve() != data
            or Path(str(marker.get("init_model", ""))).resolve() != init_model
        ):
            raise ValueError(f"Frozen baseline seed {seed} completion marker is invalid.")
        record = recorded_by_seed.get(seed)
        if not record:
            raise ValueError(f"Frozen baseline marker is missing seed {seed}.")
        checkpoint_hash = sha256(checkpoint)
        if (
            record.get("method") != "baseline"
            or Path(str(record.get("checkpoint", ""))).resolve() != checkpoint.resolve()
            or record.get("checkpoint_sha256") != checkpoint_hash
        ):
            raise ValueError(f"Frozen baseline seed {seed} checkpoint identity changed.")
        metrics = _audit_checkpoint(
            checkpoint, run_dir, training=config["training"], data=data,
            init_model=init_model, seed=seed, method="baseline",
        )
        if (
            int(record.get("best_epoch", -1)) != int(metrics["best_epoch"])
            or not _same_number(record.get("best_val_map50_95"), metrics[METRIC], 5e-6)
        ):
            raise ValueError(f"Frozen baseline seed {seed} recorded validation metric changed.")
        rows.append({
            "method": "baseline", "seed": seed,
            "best_epoch": int(metrics["best_epoch"]),
            "best_val_map50_95": float(metrics[METRIC]),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
        })
    values = [float(row["best_val_map50_95"]) for row in rows]
    if (
        not _same_number(completion.get("mean_best_val_map50_95"), mean(values), 5e-6)
        or not _same_number(completion.get("std_best_val_map50_95"), stdev(values), 5e-6)
    ):
        raise ValueError("Frozen baseline aggregate validation values changed.")
    return rows, baseline_hash


def _validate_ssl_artifact(config: dict[str, Any], *, init_model: Path) -> dict[str, str]:
    source = resolve(config["pretraining"]["config"])
    artifact = resolve(config["pretraining"]["artifact"])
    for path in (source, artifact):
        _require_file(path)
    pretraining = read_yaml(source)
    output_dir = resolve(pretraining.get("output_dir", ""))
    if artifact != output_dir / "anchored_ssl_backbone_best.pt":
        raise ValueError("SSL artifact is not the best artifact of the configured pretraining output.")
    snapshot = output_dir / "pretrain_config.yaml"
    marker_path = output_dir / "pretraining_complete.json"
    manifest = output_dir / "ms_ssl_manifest.csv"
    for path in (snapshot, marker_path, manifest):
        _require_file(path)
    if source.read_text(encoding="utf-8") != snapshot.read_text(encoding="utf-8"):
        raise ValueError("Pretraining source config differs from its output snapshot.")
    if resolve(pretraining.get("init_model", "")) != init_model:
        raise ValueError("SSL artifact and follow-up do not share the same init model.")
    pretrain_hash = canonical_hash(pretraining, compact=False)
    marker = read_json(marker_path)
    artifact_hash = sha256(artifact)
    manifest_hash = sha256(manifest)
    if (
        marker.get("complete") is not True
        or marker.get("config_sha256") != pretrain_hash
        or Path(str(marker.get("best_artifact", ""))).resolve() != artifact
        or marker.get("best_artifact_sha256") != artifact_hash
        or int(marker.get("epochs_completed", 0)) != int(pretraining.get("epochs", -1))
        or marker.get("final_test_used") is not False
        or marker.get("validation_sampling_contract") != "source_group_round_robin_v1"
        or marker.get("optimizer_contract") != "adamw_kernel_matrix_decay_bias_norm_no_decay_v1"
        or marker.get("ms_ssl_manifest_sha256") != manifest_hash
    ):
        raise ValueError("SSL completion marker does not verify the configured artifact.")
    from ultralytics import YOLO
    from scripts.ssl_anchored_dino_yolo_obb import load_artifact, load_artifact_into_detector

    payload = load_artifact(artifact)
    if Path(str(payload.get("init_model", ""))).resolve() != init_model:
        raise ValueError("SSL artifact payload references another init model.")
    detector = YOLO(str(init_model))
    loaded = load_artifact_into_detector(detector, payload)
    expected_tensors = len(payload["backbone_state_dict"])
    del detector, payload
    gc.collect()
    if loaded <= 0 or loaded != expected_tensors:
        raise ValueError(f"SSL artifact loaded {loaded}/{expected_tensors} backbone tensors.")
    return {
        "pretrain_config_sha256": pretrain_hash,
        "pretrain_config_file_sha256": sha256(source),
        "ssl_artifact_sha256": artifact_hash,
        "ms_ssl_manifest_sha256": manifest_hash,
    }


def _preflight(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    import ultralytics

    training_data = resolve(config["data"]["training"])
    init_model = resolve(config["data"]["init_model"])
    for path in (training_data, init_model):
        _require_file(path)
    data_config = read_yaml(training_data)
    if "test" in data_config or not {"train", "val"}.issubset(data_config):
        raise ValueError("Development data must define train/val and must not expose a test split.")
    provenance = _validate_training_data(training_data)
    baseline_rows, baseline_hash = _validate_frozen_baseline(
        config, data=training_data, init_model=init_model,
    )
    ssl_identity = _validate_ssl_artifact(config, init_model=init_model)
    return {
        "training_data": training_data, "init_model": init_model,
        "provenance": provenance, "baseline_rows": baseline_rows,
        "baseline_config_sha256": baseline_hash, **ssl_identity,
        "runtime_versions": {
            "torch": str(torch.__version__), "ultralytics": str(ultralytics.__version__),
        },
    }


def _ssl_request(
    config: dict[str, Any], context: dict[str, Any], *, config_sha: str, seed: int,
) -> dict[str, Any]:
    training = config["training"]
    data = context["training_data"]
    init_model = context["init_model"]
    artifact = resolve(config["pretraining"]["artifact"])
    baseline = next(row for row in context["baseline_rows"] if int(row["seed"]) == seed)
    return {
        "config_sha256": config_sha, "method": "anchored_ssl", "seed": seed,
        "data": str(data), "data_yaml_sha256": sha256(data),
        "data_provenance": str(context["provenance"]),
        "data_provenance_sha256": sha256(context["provenance"]),
        "init_model": str(init_model), "init_model_sha256": sha256(init_model),
        "ssl_artifact": str(artifact),
        "ssl_artifact_sha256": context["ssl_artifact_sha256"],
        "pretrain_config_sha256": context["pretrain_config_sha256"],
        "ms_ssl_manifest_sha256": context["ms_ssl_manifest_sha256"],
        "frozen_baseline_checkpoint": baseline["checkpoint"],
        "frozen_baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "frozen_baseline_config_sha256": context["baseline_config_sha256"],
        "epochs": int(training["epochs"]), "imgsz": int(training["imgsz"]),
        "batch": int(training["batch"]), "workers": int(training["workers"]),
        "device": str(training["device"]), "lr0": float(training["lr0"]),
        "weight_decay": float(training["weight_decay"]),
        "runtime_versions": context["runtime_versions"],
        "validation_only": True, "current_al_test_forbidden": True,
    }


def _ssl_complete(
    run_dir: Path, request: dict[str, Any], *, config: dict[str, Any], audit: bool,
) -> bool:
    required = (
        run_dir / "requested_run.json", run_dir / "ssl_finetune.json",
        run_dir / "args.yaml", run_dir / "results.csv", run_dir / "weights" / "best.pt",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        if read_json(required[0]) != request:
            return False
        marker = read_json(required[1])
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    checkpoint = required[-1]
    expected = {
        "pipeline": "anchored_dinov2_ms_ssl_then_supervised_al_obb",
        "seed": int(request["seed"]), "epochs": int(request["epochs"]),
    }
    if any(marker.get(name) != value for name, value in expected.items()):
        return False
    if (
        marker.get("test_used_for_selection") is not False
        or Path(str(marker.get("best_checkpoint", ""))).resolve() != checkpoint.resolve()
        or Path(str(marker.get("data", ""))).resolve() != Path(request["data"]).resolve()
        or Path(str(marker.get("init_model", ""))).resolve() != Path(request["init_model"]).resolve()
        or Path(str(marker.get("ssl_artifact", ""))).resolve() != Path(request["ssl_artifact"]).resolve()
    ):
        return False
    if audit:
        _audit_checkpoint(
            checkpoint, run_dir, training=config["training"],
            data=Path(request["data"]), init_model=Path(request["init_model"]),
            seed=int(request["seed"]), method="anchored_ssl",
        )
    return True


def _prepare_run(
    run_dir: Path, request: dict[str, Any], *, config: dict[str, Any], reuse: bool,
) -> str:
    if _ssl_complete(run_dir, request, config=config, audit=True):
        if reuse:
            return "reuse"
        raise FileExistsError(f"Completed run exists: {run_dir}; pass --reuse-existing.")
    request_path = run_dir / "requested_run.json"
    if request_path.is_file():
        if read_json(request_path) != request:
            raise ValueError(f"Run directory belongs to different settings: {run_dir}")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Untracked non-empty run directory: {run_dir}")
    else:
        write_json(request_path, request)
    if (run_dir / "ssl_finetune.json").exists():
        raise ValueError(f"Invalid or mismatched completion marker in {run_dir}.")
    last = run_dir / "weights" / "last.pt"
    if last.is_file():
        return "resume"
    unexpected = [run_dir / "results.csv", run_dir / "weights" / "best.pt"]
    if any(path.exists() for path in unexpected):
        raise FileExistsError(f"Partial run cannot be safely resumed without {last}: {run_dir}")
    return "start"


def _launch(command: list[str]) -> None:
    print("$ " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _paired_statistics(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("At least one paired delta is required.")
    average = mean(values)
    deviation = stdev(values) if len(values) > 1 else 0.0
    critical = {2: 12.706204736, 3: 4.30265273}.get(len(values), 1.96)
    half_width = critical * deviation / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "n_pairs": len(values), "mean_delta": average, "std_delta": deviation,
        "ci95_low": average - half_width, "ci95_high": average + half_width,
    }


def _summarize(
    config: dict[str, Any], context: dict[str, Any], output_dir: Path, config_sha: str,
) -> dict[str, Any]:
    baseline_rows = list(context["baseline_rows"])
    ssl_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        run_dir = output_dir / "anchored_ssl" / f"seed{seed}"
        request = _ssl_request(config, context, config_sha=config_sha, seed=seed)
        if not _ssl_complete(run_dir, request, config=config, audit=True):
            raise FileNotFoundError(f"Incomplete or mismatched anchored-SSL seed {seed}: {run_dir}")
        metrics = _best_csv_metrics(run_dir)
        checkpoint = run_dir / "weights" / "best.pt"
        ssl_rows.append({
            "method": "anchored_ssl", "seed": seed,
            "best_epoch": int(metrics["best_epoch"]),
            "best_val_map50_95": float(metrics[METRIC]),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
        })
    rows = baseline_rows + ssl_rows
    fields = (
        "method", "seed", "best_epoch", "best_val_map50_95",
        "checkpoint", "checkpoint_sha256",
    )
    write_csv(output_dir / "validation" / "per_seed_validation.csv", rows, fields)
    aggregates: list[dict[str, Any]] = []
    for method in ("baseline", "anchored_ssl"):
        values = [float(row["best_val_map50_95"]) for row in rows if row["method"] == method]
        aggregates.append({
            "method": method, "n_seeds": len(values),
            "mean_best_val_map50_95": mean(values),
            "std_best_val_map50_95": stdev(values),
        })
    write_csv(
        output_dir / "validation" / "aggregate_validation.csv", aggregates,
        ("method", "n_seeds", "mean_best_val_map50_95", "std_best_val_map50_95"),
    )
    paired: list[dict[str, Any]] = []
    for seed in SEEDS:
        baseline = next(float(row["best_val_map50_95"]) for row in baseline_rows if row["seed"] == seed)
        ssl = next(float(row["best_val_map50_95"]) for row in ssl_rows if row["seed"] == seed)
        paired.append({"seed": seed, "anchored_ssl_minus_baseline": ssl - baseline})
    write_csv(
        output_dir / "validation" / "paired_validation_delta.csv", paired,
        ("seed", "anchored_ssl_minus_baseline"),
    )
    deltas = [float(row["anchored_ssl_minus_baseline"]) for row in paired]
    statistics = _paired_statistics(deltas)
    minimum = float(config["selection"].get("minimum_mean_delta", 0.0))
    mean_delta = float(statistics["mean_delta"])
    primary_win = mean_delta > 0.0 if minimum == 0.0 else mean_delta >= minimum
    all_seed_win = all(delta > 0.0 if minimum == 0.0 else delta >= minimum for delta in deltas)
    frozen = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha, "selection_metric": METRIC,
        "selection_data": "AL validation only", "final_test_used": False,
        "development_only": True, "current_al_test_forbidden": True,
        "requires_new_untouched_geographic_holdout": True,
        "seeds": SEEDS, "all_runs_complete": True,
        "baseline_mean": next(row["mean_best_val_map50_95"] for row in aggregates if row["method"] == "baseline"),
        "anchored_ssl_mean": next(row["mean_best_val_map50_95"] for row in aggregates if row["method"] == "anchored_ssl"),
        "anchored_ssl_minus_baseline": statistics,
        "minimum_mean_delta": minimum,
        "primary_mean_rule_winner": "anchored_ssl" if primary_win else "baseline",
        "anchored_ssl_beats_baseline_mean": primary_win,
        "anchored_ssl_beats_baseline_all_seeds": all_seed_win,
        "seed_wins": sum(delta > 0.0 for delta in deltas),
        "frozen_checkpoints": {
            method: {
                str(row["seed"]): {
                    "path": row["checkpoint"], "sha256": row["checkpoint_sha256"],
                }
                for row in rows if row["method"] == method
            }
            for method in ("baseline", "anchored_ssl")
        },
    }
    write_json(output_dir / "validation" / "frozen_selection.json", frozen)
    print(json.dumps(frozen, indent=2), flush=True)
    return frozen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("development", "summarize"), default="development")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = resolve(args.config)
    config = read_config(config_path)
    context = _preflight(config)
    print(
        "Preflight passed: frozen 640 baseline and v3 MS-only SSL artifact verified; "
        "the existing AL test split is unavailable to this protocol.",
        flush=True,
    )
    if args.preflight_only:
        return 0
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "protocol_config.yaml"
    source_text = config_path.read_text(encoding="utf-8")
    if snapshot.is_file() and snapshot.read_text(encoding="utf-8") != source_text:
        raise ValueError("Output directory belongs to a different protocol config.")
    if not snapshot.is_file():
        write_text(snapshot, source_text)
    config_sha = canonical_hash(config, compact=True)
    if args.phase == "development":
        training = config["training"]
        artifact = resolve(config["pretraining"]["artifact"])
        for seed in SEEDS:
            run_dir = output_dir / "anchored_ssl" / f"seed{seed}"
            request = _ssl_request(config, context, config_sha=config_sha, seed=seed)
            action = _prepare_run(
                run_dir, request, config=config, reuse=args.reuse_existing,
            )
            if action == "reuse":
                print(f"Reusing completed anchored-SSL seed {seed}: {run_dir}", flush=True)
                continue
            command = [
                sys.executable, "-m", "scripts.train_ssl_initialized_obb",
                "--ssl-artifact", str(artifact),
                "--data", str(context["training_data"]),
                "--init-model", str(context["init_model"]),
                "--output-dir", str(run_dir), "--epochs", str(training["epochs"]),
                "--batch", str(training["batch"]), "--imgsz", str(training["imgsz"]),
                "--workers", str(training["workers"]), "--device", str(training["device"]),
                "--seed", str(seed), "--lr0", str(training["lr0"]),
                "--weight-decay", str(training["weight_decay"]),
            ]
            if action == "resume":
                command.append("--resume")
            _launch(command)
            if not _ssl_complete(run_dir, request, config=config, audit=True):
                raise RuntimeError(f"Anchored-SSL seed {seed} did not produce a verified completion.")
    _summarize(config, context, output_dir, config_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
