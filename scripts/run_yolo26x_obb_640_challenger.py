"""Run a validation-only YOLO26x-OBB challenger against the frozen 640 baseline.

The protocol can train or summarize seeds 7/17/42, but it deliberately has no
test phase and rejects test references in its configuration.  The official
YOLO26x-OBB initialization and every baseline checkpoint are pinned by SHA-256.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable
from urllib.parse import urlparse

import yaml

from scripts.train_yolo26x_obb_challenger import (
    METRIC,
    OPTIMIZER_CONTRACT,
    PIPELINE,
    RECIPE,
    _assert_yolo26x_obb,
    _checkpoint_metadata,
    audit_train_args,
    expected_train_args,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [7, 17, 42]
OFFICIAL_MODEL = {
    "filename": "yolo26x-obb.pt",
    "release_tag": "v8.4.0",
    "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-obb.pt",
    "sha256": "d7b1d805b0ec149890efde89119ef9d60481566ec83166052c04b5d89d9e3cb7",
    "size_bytes": 126_937_426,
}
BASELINE_TRAINING = {
    "epochs": 100, "batch": 16, "imgsz": 640, "workers": 12,
    "device": "0", "lr0": 0.001, "weight_decay": 0.0005,
}
TEST_TOKEN = re.compile(r"(^|[\\/_.-])test($|[\\/_.-])", re.IGNORECASE)


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


def _same_number(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _forbidden_test_references(value: Any, prefix: str = "") -> list[str]:
    """Return config locations that could expose or name a test resource."""

    found: list[str] = []
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            location = f"{prefix}.{key}" if prefix else key
            normalized = key.lower().replace("-", "_").replace(" ", "_")
            if re.search(r"(^|_)test($|_)", normalized):
                found.append(location)
            found.extend(_forbidden_test_references(child, location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_test_references(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and TEST_TOKEN.search(value.replace(":", "/")):
        found.append(prefix)
    return found


def _require_digest(value: Any, location: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{location} must be a lowercase SHA-256 digest.")
    return digest


def read_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Protocol config must be a YAML mapping.")
    forbidden = sorted(set(_forbidden_test_references(payload)))
    if forbidden:
        raise ValueError("Validation-only protocol forbids test references: " + ", ".join(forbidden))
    required = {"output_dir", "baseline", "challenger", "data", "training", "runtime", "selection"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Protocol config is missing {sorted(missing)}.")
    for name in required.difference({"output_dir"}):
        if not isinstance(payload[name], dict):
            raise ValueError(f"{name} must be a mapping.")

    training = payload["training"]
    if [int(value) for value in training.get("seeds", [])] != SEEDS:
        raise ValueError("Frozen comparison requires seeds [7, 17, 42].")
    for name in ("epochs", "batch", "imgsz", "workers", "device", "lr0", "lrf", "momentum",
                 "weight_decay", "nbs", "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
                 "close_mosaic", "patience"):
        if name not in training:
            raise ValueError(f"training is missing {name!r}.")
        expected = RECIPE[name]
        observed = training[name]
        matches = str(observed) == str(expected) if name == "device" else _same_number(observed, expected)
        if not matches:
            raise ValueError(f"Frozen MuSGD recipe requires training.{name}={expected!r}.")
    if training.get("optimizer") != "MuSGD" or training.get("optimizer_contract") != OPTIMIZER_CONTRACT:
        raise ValueError("Training must use the pinned MuSGD optimizer contract.")
    if int(training.get("effective_nominal_batch", 0)) != int(RECIPE["nbs"]):
        raise ValueError("effective_nominal_batch must be 64 (batch 8 with accumulation 8).")

    challenger = payload["challenger"]
    for name, expected in OFFICIAL_MODEL.items():
        observed = challenger.get(name)
        matches = int(observed) == expected if name == "size_bytes" else str(observed) == str(expected)
        if not matches:
            raise ValueError(f"challenger.{name} must pin the official {expected!r} value.")
    if Path(str(challenger.get("local_path", ""))).name != OFFICIAL_MODEL["filename"]:
        raise ValueError("challenger.local_path must end in yolo26x-obb.pt.")

    data = payload["data"]
    for name in ("training", "training_sha256", "provenance_sha256"):
        if name not in data:
            raise ValueError(f"data is missing {name!r}.")
    _require_digest(data["training_sha256"], "data.training_sha256")
    _require_digest(data["provenance_sha256"], "data.provenance_sha256")

    baseline = payload["baseline"]
    for name in ("protocol_dir", "validation_marker", "validation_marker_sha256",
                 "config_sha256", "init_model", "init_model_sha256", "checkpoints"):
        if name not in baseline:
            raise ValueError(f"baseline is missing {name!r}.")
    for name in ("validation_marker_sha256", "config_sha256", "init_model_sha256"):
        _require_digest(baseline[name], f"baseline.{name}")
    checkpoints = baseline["checkpoints"]
    if not isinstance(checkpoints, dict) or set(map(str, checkpoints)) != set(map(str, SEEDS)):
        raise ValueError("baseline.checkpoints must define exactly seeds 7, 17, and 42.")
    for seed in SEEDS:
        row = checkpoints.get(str(seed), checkpoints.get(seed))
        if not isinstance(row, dict):
            raise ValueError(f"Missing frozen baseline checkpoint for seed {seed}.")
        for name in ("path", "sha256", "best_epoch", "best_val_map50_95"):
            if name not in row:
                raise ValueError(f"Baseline seed {seed} is missing {name!r}.")
        _require_digest(row["sha256"], f"baseline.checkpoints.{seed}.sha256")

    runtime = payload["runtime"]
    for name in ("ultralytics_version", "torch_version", "cuda_device", "minimum_gpu_memory_gib"):
        if name not in runtime:
            raise ValueError(f"runtime is missing {name!r}.")
    if int(runtime["cuda_device"]) != 0 or float(runtime["minimum_gpu_memory_gib"]) < 23.0:
        raise ValueError("Runtime must pin CUDA device 0 with at least 23 GiB.")

    selection = payload["selection"]
    if selection.get("validation_only") is not True or selection.get("metric") != METRIC:
        raise ValueError("Selection must be validation-only using metrics/mAP50-95(B).")
    if selection.get("report_all_seed_robustness") is not True:
        raise ValueError("All-seed robustness reporting is mandatory.")
    if float(selection.get("minimum_mean_delta", -1.0)) < 0:
        raise ValueError("selection.minimum_mean_delta cannot be negative.")
    return payload


def _verify_file(path: Path, *, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_size = path.stat().st_size
    if observed_size != expected_size:
        raise ValueError(f"{path} is {observed_size} bytes; expected {expected_size}. Refusing to overwrite it.")
    observed_hash = sha256(path)
    if observed_hash != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}. Refusing to overwrite it.")


def ensure_official_weight(model_config: dict[str, Any]) -> Path:
    """Atomically download the pinned official asset when it is absent."""

    target = resolve(model_config["local_path"])
    expected_size = int(model_config["size_bytes"])
    expected_hash = str(model_config["sha256"])
    if target.exists():
        _verify_file(target, expected_size=expected_size, expected_sha256=expected_hash)
        return target
    url = str(model_config["url"])
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise ValueError("Official checkpoint URL must use HTTPS on github.com.")
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(target.name + ".download.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Download lock already exists: {lock}. Confirm no download is active before removing it."
        ) from exc
    temporary_path: Path | None = None
    try:
        os.write(descriptor, f"pid={os.getpid()}\nurl={url}\n".encode("utf-8"))
        os.close(descriptor)
        descriptor = -1
        request = urllib.request.Request(url, headers={"User-Agent": "Poultry-Farm-Detection/validation-preflight"})
        with urllib.request.urlopen(request, timeout=180) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise RuntimeError(f"Official weight download returned HTTP {status}.")
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) != expected_size:
                raise ValueError(f"Official server declared {declared} bytes; expected {expected_size}.")
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=target.parent, prefix=target.name + ".", suffix=".partial", delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                digest = hashlib.sha256()
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        if total != expected_size or digest.hexdigest() != expected_hash:
            raise ValueError(
                f"Downloaded official checkpoint failed verification: size={total}, sha256={digest.hexdigest()}."
            )
        if target.exists():
            _verify_file(target, expected_size=expected_size, expected_sha256=expected_hash)
            temporary_path.unlink()
            temporary_path = None
        else:
            temporary_path.replace(target)
            temporary_path = None
        _verify_file(target, expected_size=expected_size, expected_sha256=expected_hash)
        return target
    finally:
        if "descriptor" in locals() and descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        if lock.exists():
            lock.unlink()


def _runtime_preflight(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    import ultralytics
    from ultralytics.engine.trainer import MuSGD

    runtime = config["runtime"]
    if str(ultralytics.__version__) != str(runtime["ultralytics_version"]):
        raise ValueError(
            f"Ultralytics runtime is {ultralytics.__version__}; expected {runtime['ultralytics_version']}."
        )
    if str(torch.__version__) != str(runtime["torch_version"]):
        raise ValueError(f"PyTorch runtime is {torch.__version__}; expected {runtime['torch_version']}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the frozen batch-8 recipe requires the RTX 4090.")
    device = int(runtime["cuda_device"])
    if device >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device} is unavailable.")
    properties = torch.cuda.get_device_properties(device)
    memory_gib = properties.total_memory / (1024 ** 3)
    if memory_gib < float(runtime["minimum_gpu_memory_gib"]):
        raise RuntimeError(
            f"CUDA device has {memory_gib:.2f} GiB; frozen batch 8 requires "
            f"{float(runtime['minimum_gpu_memory_gib']):.2f} GiB."
        )
    if MuSGD.__name__ != "MuSGD":
        raise RuntimeError("Installed Ultralytics does not expose the required MuSGD optimizer.")
    return {
        "python": sys.version.split()[0], "torch": str(torch.__version__),
        "ultralytics": str(ultralytics.__version__), "cuda_device": device,
        "gpu_name": str(properties.name), "gpu_memory_gib": memory_gib,
        "platform": sys.platform,
    }


def _audit_official_model(path: Path) -> dict[str, Any]:
    from ultralytics import YOLO
    from ultralytics.engine.trainer import BaseTrainer

    model = YOLO(str(path))
    identity = _assert_yolo26x_obb(model, path)
    trainer_shell = object.__new__(BaseTrainer)
    optimizer = BaseTrainer.build_optimizer(
        trainer_shell, model.model, name="MuSGD", lr=RECIPE["lr0"],
        momentum=RECIPE["momentum"], decay=RECIPE["weight_decay"], iterations=6500,
    )
    muon_groups = sum(bool(group.get("use_muon")) for group in optimizer.param_groups)
    if optimizer.__class__.__name__ != "MuSGD" or muon_groups <= 0:
        raise RuntimeError("Installed Ultralytics did not build the expected hybrid MuSGD groups.")
    if not _same_number(getattr(optimizer, "muon", None), 0.2) or not _same_number(
        getattr(optimizer, "sgd", None), 1.0
    ):
        raise RuntimeError("Installed Ultralytics MuSGD coefficients do not match the pinned recipe.")
    identity.update({
        "optimizer_class": optimizer.__class__.__name__,
        "optimizer_param_groups": len(optimizer.param_groups),
        "muon_param_groups": muon_groups,
    })
    del optimizer, model
    gc.collect()
    return identity


def _best_csv_metrics(run_dir: Path) -> dict[str, float | int]:
    path = run_dir / "results.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    candidates: list[tuple[int, float]] = []
    for row in rows:
        raw = row.get(METRIC)
        if raw in (None, ""):
            continue
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {METRIC} in {path}.")
        candidates.append((int(float(row["epoch"])), value))
    if not candidates:
        raise ValueError(f"No finite {METRIC} in {path}.")
    epoch, value = max(candidates, key=lambda item: item[1])
    return {"best_epoch": epoch, METRIC: value}


def _validate_training_data(config: dict[str, Any]) -> tuple[Path, Path]:
    data = resolve(config["data"]["training"])
    if not data.is_file() or sha256(data) != str(config["data"]["training_sha256"]):
        raise ValueError(f"Frozen training YAML is missing or changed: {data}")
    data_payload = yaml.safe_load(data.read_text(encoding="utf-8"))
    if not isinstance(data_payload, dict) or not {"train", "val"}.issubset(data_payload):
        raise ValueError("Training YAML must define train and val.")
    if _forbidden_test_references(data_payload):
        raise ValueError("Training YAML exposes a forbidden test resource.")
    from scripts.run_full_al_640_baseline import _validate_training_data as audit

    provenance = audit(data)
    if sha256(provenance) != str(config["data"]["provenance_sha256"]):
        raise ValueError(f"Frozen training provenance changed: {provenance}")
    return data, provenance


def _validate_frozen_baseline(config: dict[str, Any], training_data: Path) -> list[dict[str, Any]]:
    baseline = config["baseline"]
    protocol_dir = resolve(baseline["protocol_dir"])
    marker_path = resolve(baseline["validation_marker"])
    if marker_path != protocol_dir / "validation_complete.json":
        raise ValueError("Baseline validation marker must be inside the frozen protocol directory.")
    _verify_file(
        marker_path, expected_size=marker_path.stat().st_size,
        expected_sha256=str(baseline["validation_marker_sha256"]),
    )
    init_model = resolve(baseline["init_model"])
    if not init_model.is_file() or sha256(init_model) != str(baseline["init_model_sha256"]):
        raise ValueError("Frozen YOLO11l baseline initialization changed.")
    marker = read_json(marker_path)
    if (
        marker.get("config_sha256") != baseline["config_sha256"]
        or marker.get("final_test_used") is not False
        or marker.get("selection_data") != "AL validation only"
        or [int(value) for value in marker.get("seeds", [])] != SEEDS
        or int(marker.get("imgsz", 0)) != 640
        or int(marker.get("batch", 0)) != 16
    ):
        raise ValueError("Frozen baseline validation marker is inconsistent.")
    recorded = marker.get("checkpoints")
    if not isinstance(recorded, list) or len(recorded) != 3:
        raise ValueError("Frozen baseline marker has incomplete checkpoint records.")
    by_seed = {int(row["seed"]): row for row in recorded if isinstance(row, dict)}
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        expected_record = baseline["checkpoints"].get(str(seed), baseline["checkpoints"].get(seed))
        checkpoint = resolve(expected_record["path"])
        expected_path = protocol_dir / "baseline" / f"seed{seed}" / "weights" / "best.pt"
        if checkpoint != expected_path:
            raise ValueError(f"Frozen baseline seed {seed} path changed.")
        _verify_file(
            checkpoint, expected_size=checkpoint.stat().st_size,
            expected_sha256=str(expected_record["sha256"]),
        )
        run_dir = checkpoint.parents[1]
        for required in (run_dir / "requested_run.json", run_dir / "supervised_baseline.json",
                         run_dir / "args.yaml", run_dir / "results.csv"):
            if not required.is_file():
                raise FileNotFoundError(required)
        request = read_json(run_dir / "requested_run.json")
        expected_request = {
            "config_sha256": baseline["config_sha256"], "method": "baseline", "seed": seed,
            "data": str(training_data), "init_model": str(init_model), **BASELINE_TRAINING,
        }
        if request != expected_request:
            raise ValueError(f"Frozen baseline seed {seed} request changed.")
        completion = read_json(run_dir / "supervised_baseline.json")
        if (
            completion.get("pipeline") != "supervised_yolo_obb_burn_in"
            or completion.get("test_used_for_selection") is not False
            or int(completion.get("seed", -1)) != seed
            or Path(str(completion.get("best_checkpoint", ""))).resolve() != checkpoint
        ):
            raise ValueError(f"Frozen baseline seed {seed} completion marker is invalid.")
        metrics = _best_csv_metrics(run_dir)
        recorded_row = by_seed.get(seed)
        if not recorded_row:
            raise ValueError(f"Frozen baseline marker lacks seed {seed}.")
        if (
            recorded_row.get("checkpoint_sha256") != expected_record["sha256"]
            or int(recorded_row.get("best_epoch", -1)) != int(expected_record["best_epoch"])
            or not _same_number(recorded_row.get("best_val_map50_95"), expected_record["best_val_map50_95"], 5e-6)
            or int(metrics["best_epoch"]) != int(expected_record["best_epoch"])
            or not _same_number(metrics[METRIC], expected_record["best_val_map50_95"], 5e-6)
        ):
            raise ValueError(f"Frozen baseline seed {seed} validation identity changed.")
        rows.append({
            "method": "yolo11l_baseline", "seed": seed,
            "best_epoch": int(metrics["best_epoch"]),
            "best_val_map50_95": float(metrics[METRIC]),
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        })
    return rows


def _preflight(config: dict[str, Any]) -> dict[str, Any]:
    runtime = _runtime_preflight(config)
    training_data, provenance = _validate_training_data(config)
    baseline_rows = _validate_frozen_baseline(config, training_data)
    model_path = ensure_official_weight(config["challenger"])
    model_identity = _audit_official_model(model_path)
    return {
        "runtime": runtime, "training_data": training_data, "provenance": provenance,
        "training_data_sha256": sha256(training_data), "provenance_sha256": sha256(provenance),
        "baseline_rows": baseline_rows, "model_path": model_path,
        "model_sha256": sha256(model_path), "model_identity": model_identity,
    }


def _request(config: dict[str, Any], context: dict[str, Any], config_sha: str, seed: int) -> dict[str, Any]:
    baseline = next(row for row in context["baseline_rows"] if row["seed"] == seed)
    return {
        "protocol_config_sha256": config_sha, "method": "yolo26x_obb_musgd",
        "pipeline": PIPELINE, "optimizer_contract": OPTIMIZER_CONTRACT, "seed": seed,
        "data": str(context["training_data"]), "data_sha256": context["training_data_sha256"],
        "provenance_sha256": context["provenance_sha256"],
        "init_model": str(context["model_path"]), "init_model_sha256": context["model_sha256"],
        "training": dict(config["training"]), "runtime": dict(context["runtime"]),
        "frozen_baseline_checkpoint": baseline["checkpoint"],
        "frozen_baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "frozen_baseline_best_val_map50_95": baseline["best_val_map50_95"],
    }


def _challenger_complete(
    run_dir: Path, request: dict[str, Any], *, context: dict[str, Any], audit: bool,
) -> bool:
    required = (
        run_dir / "requested_run.json", run_dir / "yolo26x_challenger.json",
        run_dir / "args.yaml", run_dir / "results.csv",
        run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        if read_json(required[0]) != request:
            return False
        marker = read_json(required[1])
        checkpoint = required[4]
        if (
            marker.get("pipeline") != PIPELINE
            or marker.get("optimizer_contract") != OPTIMIZER_CONTRACT
            or marker.get("protocol_config_sha256") != request["protocol_config_sha256"]
            or marker.get("test_used_for_selection") is not False
            or marker.get("validation_only") is not True
            or int(marker.get("seed", -1)) != int(request["seed"])
            or Path(str(marker.get("best_checkpoint", ""))).resolve() != checkpoint.resolve()
            or marker.get("best_checkpoint_sha256") != sha256(checkpoint)
            or marker.get("init_model_sha256") != request["init_model_sha256"]
            or marker.get("data_sha256") != request["data_sha256"]
        ):
            return False
        if audit:
            args_payload = yaml.safe_load(required[2].read_text(encoding="utf-8"))
            if not isinstance(args_payload, dict):
                return False
            expected = expected_train_args(
                data=context["training_data"], init_model=context["model_path"],
                seed=int(request["seed"]), output_dir=run_dir,
            )
            audit_train_args(args_payload, expected, run_dir=run_dir)
            metrics = _best_csv_metrics(run_dir)
            version, checkpoint_args, checkpoint_metrics = _checkpoint_metadata(checkpoint)
            if version != context["runtime"]["ultralytics"]:
                return False
            audit_train_args(checkpoint_args, expected, run_dir=run_dir)
            if (
                int(marker.get("best_epoch", -1)) != int(metrics["best_epoch"])
                or not _same_number(marker.get("best_val_map50_95"), metrics[METRIC], 5e-6)
                or not _same_number(checkpoint_metrics.get(METRIC), metrics[METRIC], 5e-6)
            ):
                return False
        return True
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
        return False


def _prepare_run(
    run_dir: Path, request: dict[str, Any], *, context: dict[str, Any], reuse: bool,
) -> str:
    if _challenger_complete(run_dir, request, context=context, audit=True):
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
    if (run_dir / "yolo26x_challenger.json").exists():
        raise ValueError(f"Invalid or mismatched completion marker: {run_dir}")
    last = run_dir / "weights" / "last.pt"
    if last.is_file():
        args_path = run_dir / "args.yaml"
        if not args_path.is_file():
            raise FileNotFoundError(f"Partial run has last.pt but no args.yaml: {run_dir}")
        actual = yaml.safe_load(args_path.read_text(encoding="utf-8"))
        expected = expected_train_args(
            data=context["training_data"], init_model=context["model_path"],
            seed=int(request["seed"]), output_dir=run_dir,
        )
        if not isinstance(actual, dict):
            raise ValueError(f"Invalid partial args.yaml: {args_path}")
        audit_train_args(actual, expected, run_dir=run_dir)
        return "resume"
    if any(path.exists() for path in (run_dir / "results.csv", run_dir / "weights" / "best.pt")):
        raise FileExistsError(f"Partial run cannot be safely resumed without last.pt: {run_dir}")
    return "start"


def _build_command(
    config: dict[str, Any], context: dict[str, Any], *, output_dir: Path,
    config_sha: str, seed: int, resume: bool,
) -> list[str]:
    training = config["training"]
    command = [
        sys.executable, "-m", "scripts.train_yolo26x_obb_challenger",
        "--data", str(context["training_data"]),
        "--data-sha256", context["training_data_sha256"],
        "--init-model", str(context["model_path"]),
        "--init-model-sha256", context["model_sha256"],
        "--protocol-config-sha256", config_sha,
        "--output-dir", str(output_dir), "--seed", str(seed),
        "--epochs", str(training["epochs"]), "--batch", str(training["batch"]),
        "--imgsz", str(training["imgsz"]), "--workers", str(training["workers"]),
        "--device", str(training["device"]), "--lr0", str(training["lr0"]),
        "--momentum", str(training["momentum"]),
        "--weight-decay", str(training["weight_decay"]),
    ]
    if resume:
        command.append("--resume")
    return command


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
    challenger_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        run_dir = output_dir / "yolo26x_musgd" / f"seed{seed}"
        request = _request(config, context, config_sha, seed)
        if not _challenger_complete(run_dir, request, context=context, audit=True):
            raise FileNotFoundError(f"Incomplete or mismatched challenger seed {seed}: {run_dir}")
        metrics = _best_csv_metrics(run_dir)
        checkpoint = run_dir / "weights" / "best.pt"
        challenger_rows.append({
            "method": "yolo26x_musgd", "seed": seed,
            "best_epoch": int(metrics["best_epoch"]),
            "best_val_map50_95": float(metrics[METRIC]),
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
        })
    rows = baseline_rows + challenger_rows
    fields = ("method", "seed", "best_epoch", "best_val_map50_95", "checkpoint", "checkpoint_sha256")
    write_csv(output_dir / "validation" / "per_seed_validation.csv", rows, fields)
    aggregates: list[dict[str, Any]] = []
    for method in ("yolo11l_baseline", "yolo26x_musgd"):
        values = [float(row["best_val_map50_95"]) for row in rows if row["method"] == method]
        aggregates.append({
            "method": method, "n_seeds": len(values),
            "mean_best_val_map50_95": mean(values), "std_best_val_map50_95": stdev(values),
        })
    write_csv(
        output_dir / "validation" / "aggregate_validation.csv", aggregates,
        ("method", "n_seeds", "mean_best_val_map50_95", "std_best_val_map50_95"),
    )
    paired: list[dict[str, Any]] = []
    for seed in SEEDS:
        baseline_value = next(float(row["best_val_map50_95"]) for row in baseline_rows if row["seed"] == seed)
        challenger_value = next(float(row["best_val_map50_95"]) for row in challenger_rows if row["seed"] == seed)
        paired.append({"seed": seed, "yolo26x_minus_yolo11l": challenger_value - baseline_value})
    write_csv(
        output_dir / "validation" / "paired_validation_delta.csv", paired,
        ("seed", "yolo26x_minus_yolo11l"),
    )
    deltas = [float(row["yolo26x_minus_yolo11l"]) for row in paired]
    statistics = _paired_statistics(deltas)
    threshold = float(config["selection"]["minimum_mean_delta"])
    mean_win = float(statistics["mean_delta"]) > threshold if threshold == 0 else float(statistics["mean_delta"]) >= threshold
    all_seed_win = all(delta > threshold if threshold == 0 else delta >= threshold for delta in deltas)
    frozen = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha, "selection_metric": METRIC,
        "selection_data": "AL validation only", "validation_only": True,
        "test_resources_forbidden": True, "test_used": False,
        "requires_new_untouched_geographic_holdout": True,
        "seeds": SEEDS, "all_runs_complete": True,
        "baseline_mean": next(row["mean_best_val_map50_95"] for row in aggregates if row["method"] == "yolo11l_baseline"),
        "challenger_mean": next(row["mean_best_val_map50_95"] for row in aggregates if row["method"] == "yolo26x_musgd"),
        "yolo26x_minus_yolo11l": statistics,
        "minimum_mean_delta": threshold,
        "primary_mean_rule_winner": "yolo26x_musgd" if mean_win else "yolo11l_baseline",
        "challenger_beats_baseline_mean": mean_win,
        "challenger_beats_baseline_all_seeds": all_seed_win,
        "seed_wins": sum(delta > 0 for delta in deltas),
        "official_model": {
            "path": str(context["model_path"]), "sha256": context["model_sha256"],
            "source": OFFICIAL_MODEL["url"], **context["model_identity"],
        },
        "optimizer_contract": OPTIMIZER_CONTRACT,
        "runtime": context["runtime"],
        "frozen_checkpoints": {
            method: {
                str(row["seed"]): {"path": row["checkpoint"], "sha256": row["checkpoint_sha256"]}
                for row in rows if row["method"] == method
            }
            for method in ("yolo11l_baseline", "yolo26x_musgd")
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
        "Preflight passed: official YOLO26x-OBB, MuSGD, frozen full-AL 640 baseline, "
        "dataset provenance, CUDA memory, and validation-only boundary verified.",
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
    config_sha = canonical_hash(config)
    if args.phase == "development":
        for seed in SEEDS:
            run_dir = output_dir / "yolo26x_musgd" / f"seed{seed}"
            request = _request(config, context, config_sha, seed)
            action = _prepare_run(run_dir, request, context=context, reuse=args.reuse_existing)
            if action == "reuse":
                print(f"Reusing completed YOLO26x seed {seed}: {run_dir}", flush=True)
                continue
            command = _build_command(
                config, context, output_dir=run_dir, config_sha=config_sha,
                seed=seed, resume=action == "resume",
            )
            _launch(command)
            if not _challenger_complete(run_dir, request, context=context, audit=True):
                raise RuntimeError(f"YOLO26x seed {seed} did not produce a verified completion.")
    _summarize(config, context, output_dir, config_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
