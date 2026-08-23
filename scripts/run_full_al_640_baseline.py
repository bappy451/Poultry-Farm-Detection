"""Run and evaluate a full-AL YOLO11l-OBB 640x640 baseline study.

The development phase trains ordinary Ultralytics YOLO-OBB from the same
DOTA-pretrained checkpoint for seeds 7, 17, and 42 using only the frozen,
deduplicated AL train/validation manifests. The explicit test phase evaluates
the frozen best checkpoints once on the untouched AL test split. No SSL,
pseudo-labels, or custom loss is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
METRIC = "metrics/mAP50-95(B)"
TEST_METRICS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_training_data(data_yaml: Path) -> Path:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not {"train", "val"}.issubset(payload):
        raise ValueError("Training YAML must define train and val splits.")
    if "test" in payload:
        raise ValueError("Training YAML must not expose a test split.")
    provenance_value = payload.get("provenance")
    if not provenance_value:
        raise ValueError("Training YAML must point to its provenance JSON.")
    provenance = resolve(str(provenance_value))
    if not provenance.is_file():
        raise FileNotFoundError(f"Missing training provenance: {provenance}")
    manifest = json.loads(provenance.read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Training provenance has no records.")
    split_images: dict[str, set[str]] = {"train": set(), "val": set()}
    split_groups: dict[str, set[str]] = {"train": set(), "val": set()}
    for record in records:
        split = str(record.get("split", ""))
        if split not in split_images:
            raise ValueError(f"Unexpected provenance split: {split!r}")
        image = resolve(str(record["image"]))
        label = resolve(str(record["label"]))
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Missing image/label pair: {image}, {label}")
        if sha256(image) != record.get("image_sha256") or sha256(label) != record.get("label_sha256"):
            raise ValueError(f"Training provenance hash mismatch: {image}")
        image_hash = str(record["image_sha256"])
        split_images[split].add(image_hash)
        split_groups[split].add(image.stem.split("_jpg.rf.")[0])
    if split_images["train"] & split_images["val"]:
        raise ValueError("Exact train/validation image overlap detected.")
    if split_groups["train"] & split_groups["val"]:
        raise ValueError("Source-group train/validation overlap detected.")
    return provenance


def _validate_config(config: dict[str, Any]) -> tuple[Path, Path, Path, str]:
    required = {"output_dir", "data", "training", "selection"}
    if not required.issubset(config):
        raise ValueError(f"Config is missing: {sorted(required.difference(config))}")
    data = config["data"]
    training = config["training"]
    if not isinstance(data, dict) or not isinstance(training, dict):
        raise ValueError("data and training must be mappings.")
    seeds = [int(value) for value in training.get("seeds", [])]
    if seeds != [7, 17, 42] or len(set(seeds)) != len(seeds):
        raise ValueError("This frozen comparison requires seeds [7, 17, 42].")
    if int(training.get("imgsz", 0)) != 640 or int(training.get("batch", 0)) != 16:
        raise ValueError("This baseline must use imgsz=640 and batch=16.")
    if int(training.get("epochs", 0)) <= 0 or int(training.get("workers", -1)) < 0:
        raise ValueError("epochs must be positive and workers cannot be negative.")
    training_data = resolve(str(data["training"]))
    final_test = resolve(str(data["final_test"]))
    init_model = resolve(str(data["init_model"]))
    for path in (training_data, final_test, init_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    provenance = _validate_training_data(training_data)
    if str(config["selection"].get("metric")) != METRIC:
        raise ValueError(f"selection.metric must be {METRIC!r}.")
    if config["selection"].get("validation_only") is not True:
        raise ValueError("selection.validation_only must be true.")
    return training_data, final_test, init_model, str(provenance)


def _run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _complete(run_dir: Path, requested: dict[str, Any]) -> bool:
    marker_path = run_dir / "supervised_baseline.json"
    best = run_dir / "weights" / "best.pt"
    if not marker_path.is_file() or not best.is_file() or not (run_dir / "results.csv").is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        request = json.loads((run_dir / "requested_run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("pipeline") == "supervised_yolo_obb_burn_in"
        and Path(str(marker.get("best_checkpoint", ""))).resolve() == best.resolve()
        and marker.get("test_used_for_selection") is False
        and request == requested
    )


def _validation_summary(output_dir: Path, config: dict[str, Any], cfg_hash: str) -> None:
    rows: list[dict[str, Any]] = []
    for seed in [7, 17, 42]:
        run_dir = output_dir / "baseline" / f"seed{seed}"
        with (run_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        values = [float(row[METRIC]) for row in history if row.get(METRIC) not in (None, "")]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Invalid validation history: {run_dir}")
        best_index = max(range(len(values)), key=values.__getitem__)
        rows.append({
            "method": "baseline", "seed": seed, "best_epoch": int(history[best_index]["epoch"]),
            "best_val_map50_95": values[best_index],
            "checkpoint": str((run_dir / "weights" / "best.pt").resolve()),
            "checkpoint_sha256": sha256(run_dir / "weights" / "best.pt"),
        })
    write_csv(output_dir / "validation_per_seed.csv", rows)
    values = [float(row["best_val_map50_95"]) for row in rows]
    write_csv(output_dir / "validation_aggregate.csv", [{
        "method": "baseline", "n_seeds": len(values), "mean_best_val_map50_95": mean(values),
        "std_best_val_map50_95": stdev(values),
    }])
    write_json(output_dir / "validation_complete.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": cfg_hash,
        "selection_data": "AL validation only", "final_test_used": False,
        "imgsz": config["training"]["imgsz"], "batch": config["training"]["batch"],
        "seeds": [7, 17, 42], "mean_best_val_map50_95": mean(values),
        "std_best_val_map50_95": stdev(values), "checkpoints": rows,
    })


def _test_phase(
    output_dir: Path,
    config: dict[str, Any],
    cfg_hash: str,
    final_test: Path,
    training_data: Path,
    *,
    reuse_existing: bool,
) -> None:
    validation_marker = output_dir / "validation_complete.json"
    if not validation_marker.is_file():
        raise FileNotFoundError("Run the development phase before the test phase.")
    from scripts.run_ema_benchmark import audit_test_overlap

    overlap = audit_test_overlap(training_data, final_test, final_test_split=str(config["data"]["final_test_split"]))
    if overlap["exact_sha256_matches"]:
        raise ValueError("Exact train/validation-to-test image overlap detected.")
    test_dir = output_dir / "final_test"
    if test_dir.exists() and any(test_dir.iterdir()) and not reuse_existing:
        raise FileExistsError(
            f"Refusing to reuse final-test output directory: {test_dir}. "
            "Use --reuse-existing or a new output directory."
        )
    test_dir.mkdir(parents=True, exist_ok=True)
    write_json(test_dir / "test_overlap_audit.json", overlap)
    write_json(test_dir / "frozen_test_protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": cfg_hash,
        "test_used_for_selection": False, "training_data": str(training_data),
        "final_test_data": str(final_test), "final_test_split": str(config["data"]["final_test_split"]),
        "seeds": [7, 17, 42], "checkpoints": {
            str(seed): str((output_dir / "baseline" / f"seed{seed}" / "weights" / "best.pt").resolve())
            for seed in [7, 17, 42]
        },
    })
    result_path = test_dir / "per_seed_test.csv"
    rows: list[dict[str, Any]] = []
    if result_path.is_file():
        with result_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    completed = {int(row["seed"]) for row in rows}
    from ultralytics import YOLO

    for seed in [7, 17, 42]:
        if seed in completed:
            continue
        checkpoint = output_dir / "baseline" / f"seed{seed}" / "weights" / "best.pt"
        result = YOLO(str(checkpoint)).val(
            data=str(final_test), split=str(config["data"]["final_test_split"]), imgsz=640,
            batch=32, workers=12, device=str(config["training"]["device"]), plots=False,
            project=str(test_dir / "runs"), name=f"baseline_seed{seed}", exist_ok=True, verbose=False,
        )
        values = getattr(result, "results_dict", {}) or {}
        row: dict[str, Any] = {"method": "baseline", "seed": seed, "checkpoint": str(checkpoint.resolve())}
        for metric in TEST_METRICS:
            value = values.get(metric)
            if value is None or not math.isfinite(float(value)):
                raise RuntimeError(f"Missing finite {metric} for seed {seed}.")
            row[metric] = float(value)
        rows.append(row)
        write_csv(result_path, rows)
    if len(rows) != 3:
        raise RuntimeError("Incomplete baseline test results.")
    write_csv(test_dir / "aggregate_test.csv", [{
        "method": "baseline", "n_seeds": 3,
        "mean_test_map50_95": mean(float(row["metrics/mAP50-95(B)"]) for row in rows),
        "std_test_map50_95": stdev(float(row["metrics/mAP50-95(B)"]) for row in rows),
    }])
    write_json(test_dir / "test_summary.json", {
        "test_used_for_selection": False, "config_sha256": cfg_hash,
        "aggregate": rows,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("development", "test", "all"), default="development")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping.")
    training_data, final_test, init_model, provenance = _validate_config(config)
    cfg_hash = config_hash(config)
    output_dir = resolve(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "protocol_config.yaml"
    source_text = config_path.read_text(encoding="utf-8")
    if snapshot.is_file() and snapshot.read_text(encoding="utf-8") != source_text:
        raise ValueError("Output directory belongs to a different config.")
    if not snapshot.is_file():
        snapshot.write_text(source_text, encoding="utf-8")
    print(f"Preflight passed: {training_data} ({provenance}); full AL 640x640 baseline only.", flush=True)
    if args.preflight_only:
        return 0
    if args.phase in {"development", "all"}:
        training = config["training"]
        for seed in [7, 17, 42]:
            run_dir = output_dir / "baseline" / f"seed{seed}"
            requested = {
                "config_sha256": cfg_hash, "method": "baseline", "seed": seed,
                "data": str(training_data), "init_model": str(init_model),
                "epochs": int(training["epochs"]), "imgsz": int(training["imgsz"]),
                "batch": int(training["batch"]), "workers": int(training["workers"]),
                "device": str(training["device"]), "lr0": float(training["lr0"]),
                "weight_decay": float(training["weight_decay"]),
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            request_path = run_dir / "requested_run.json"
            if request_path.is_file() and json.loads(request_path.read_text(encoding="utf-8")) != requested:
                raise ValueError(f"Run directory belongs to different settings: {run_dir}")
            if _complete(run_dir, requested):
                if not args.reuse_existing:
                    raise FileExistsError(f"Completed run exists: {run_dir}; use --reuse-existing.")
                print(f"Reusing completed seed {seed}: {run_dir}", flush=True)
                continue
            write_json(request_path, requested)
            command = [sys.executable, "-m", "scripts.train_supervised_obb",
                       "--data", str(training_data), "--init-model", str(init_model),
                       "--output-dir", str(run_dir), "--epochs", str(training["epochs"]),
                       "--batch", str(training["batch"]), "--imgsz", str(training["imgsz"]),
                       "--workers", str(training["workers"]), "--device", str(training["device"]),
                       "--seed", str(seed), "--lr0", str(training["lr0"]),
                       "--weight-decay", str(training["weight_decay"])]
            if (run_dir / "weights" / "last.pt").is_file():
                command.append("--resume")
            _run(command)
        _validation_summary(output_dir, config, cfg_hash)
    if args.phase in {"test", "all"}:
        _test_phase(
            output_dir, config, cfg_hash, final_test, training_data,
            reuse_existing=args.reuse_existing,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
