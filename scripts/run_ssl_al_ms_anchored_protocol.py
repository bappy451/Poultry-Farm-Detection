"""Run the MS self-supervised -> AL fine-tune -> frozen AL-test protocol.

Development performs MS-only anchored DINOv2 pretraining and compares the
resulting initialization with the exact Ultralytics baseline over seeds
7/17/42 using AL validation only.  Final test evaluation is a separate,
explicit phase and cannot run until the validation comparison is frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import numbers
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Protocol config must be a YAML mapping.")
    for mapping in ("data", "training", "selection"):
        if not isinstance(config.get(mapping), dict):
            raise ValueError(f"Protocol requires a '{mapping}' mapping.")
    required_data = {"training", "final_test", "final_test_split", "init_model"}
    if not required_data.issubset(config["data"]):
        raise ValueError(f"Protocol data mapping is missing {sorted(required_data - set(config['data']))}.")
    required_training = {"seeds", "epochs", "imgsz", "batch", "workers", "device", "lr0", "weight_decay"}
    if not required_training.issubset(config["training"]):
        raise ValueError(
            f"Protocol training mapping is missing {sorted(required_training - set(config['training']))}."
        )
    seeds = [int(seed) for seed in config["training"].get("seeds", [])]
    if seeds != [7, 17, 42]:
        raise ValueError("Publication protocol requires seeds [7, 17, 42] in that order.")
    if config["selection"].get("validation_only") is not True:
        raise ValueError("selection.validation_only must be true.")
    if config["selection"].get("metric") != "metrics/mAP50-95(B)":
        raise ValueError("selection.metric must be metrics/mAP50-95(B).")
    if config["selection"].get("require_all_seeds") is not True:
        raise ValueError("Publication protocol requires selection.require_all_seeds=true.")
    if float(config["selection"].get("minimum_mean_delta", 0.0)) < 0:
        raise ValueError("selection.minimum_mean_delta cannot be negative.")
    if list(config.get("methods", [])) != ["baseline", "anchored_ssl"]:
        raise ValueError("methods must be [baseline, anchored_ssl].")
    training = config["training"]
    if int(training["epochs"]) <= 0 or int(training["batch"]) <= 0 or int(training["workers"]) < 0:
        raise ValueError("training epochs/batch must be positive and workers cannot be negative.")
    if int(training["imgsz"]) <= 0 or int(training["imgsz"]) % 32:
        raise ValueError("training.imgsz must be a positive multiple of 32.")
    if float(training["lr0"]) <= 0 or float(training["weight_decay"]) < 0:
        raise ValueError("training.lr0 must be positive and weight_decay cannot be negative.")
    if str(config["data"]["final_test_split"]) != "test":
        raise ValueError("Publication protocol final_test_split must be test.")
    for name in ("output_dir", "pretrain_config", "ssl_artifact"):
        if not str(config.get(name, "")).strip():
            raise ValueError(f"Protocol requires non-empty {name}.")
    return config


def _config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


@lru_cache(maxsize=None)
def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _launch(command: list[str]) -> None:
    print("$ " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _validate_development_provenance(data_config: dict[str, Any], data_yaml: Path) -> Path:
    if "provenance" not in data_config:
        raise ValueError("Development data YAML must provide a deduplication provenance file.")

    def resolve_from_yaml(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (data_yaml.parent / path).resolve()

    train_manifest = resolve_from_yaml(data_config["train"])
    val_manifest = resolve_from_yaml(data_config["val"])
    provenance_path = resolve_from_yaml(data_config["provenance"])
    for path in (train_manifest, val_manifest, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        Path(str(provenance.get("train_manifest", ""))).resolve() != train_manifest
        or Path(str(provenance.get("val_manifest", ""))).resolve() != val_manifest
        or provenance.get("train_manifest_sha256") != _file_sha256(train_manifest)
        or provenance.get("val_manifest_sha256") != _file_sha256(val_manifest)
        or int(provenance.get("exact_train_val_overlap", -1)) != 0
    ):
        raise ValueError("Development deduplication provenance does not match its manifests.")
    train_paths = [Path(line) for line in train_manifest.read_text(encoding="utf-8-sig").splitlines() if line]
    val_paths = [Path(line) for line in val_manifest.read_text(encoding="utf-8-sig").splitlines() if line]
    if (
        len(train_paths) != int(provenance.get("retained_train_images", -1))
        or len(val_paths) != int(provenance.get("retained_val_images", -1))
        or len(set(train_paths + val_paths)) != len(train_paths) + len(val_paths)
    ):
        raise ValueError("Development manifests have incorrect counts or duplicate paths.")
    records = provenance.get("records")
    if not isinstance(records, list) or len(records) != len(train_paths) + len(val_paths):
        raise ValueError("Development provenance has an incomplete content record.")
    expected_images = {str(path) for path in [*train_paths, *val_paths]}
    recorded_images = {str(record.get("image", "")) for record in records if isinstance(record, dict)}
    if recorded_images != expected_images:
        raise ValueError("Development provenance image set differs from the manifests.")
    for record in records:
        image = Path(str(record["image"]))
        label = Path(str(record["label"]))
        if not image.is_file() or _file_sha256(image) != record.get("image_sha256"):
            raise ValueError(f"Development image changed after provenance creation: {image}")
        if not label.is_file() or _file_sha256(label) != record.get("label_sha256"):
            raise ValueError(f"Development label changed after provenance creation: {label}")
    return provenance_path


def _dataset_split_sha256(data_yaml: Path, split: str) -> str:
    """Fingerprint every image and label used by one explicit dataset split."""
    from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths

    checked = check_det_dataset(str(data_yaml))
    sources = checked.get(split)
    if sources is None:
        raise ValueError(f"Dataset does not define split {split!r}: {data_yaml}")
    if not isinstance(sources, list):
        sources = [sources]
    images: list[Path] = []
    for raw_source in sources:
        source = Path(str(raw_source))
        if source.is_dir():
            images.extend(
                path.resolve() for path in source.rglob("*")
                if path.is_file() and path.suffix.lower().lstrip(".") in IMG_FORMATS
            )
        elif source.is_file() and source.suffix.lower() == ".txt":
            for line in source.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    path = Path(line.strip())
                    images.append(path.resolve() if path.is_absolute() else (source.parent / path).resolve())
        elif source.is_file() and source.suffix.lower().lstrip(".") in IMG_FORMATS:
            images.append(source.resolve())
        else:
            raise FileNotFoundError(f"Unsupported or missing dataset split source: {source}")
    images = sorted(images)
    if not images or len(set(images)) != len(images):
        raise ValueError(f"Dataset split {split!r} is empty or contains duplicate paths.")
    labels = [Path(value) for value in img2label_paths([str(path) for path in images])]
    digest = hashlib.sha256()
    for image, label in zip(images, labels):
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Missing image/label pair in {split}: {image}, {label}")
        digest.update(str(image).encode("utf-8"))
        digest.update(_file_sha256(image).encode("ascii"))
        digest.update(str(label).encode("utf-8"))
        digest.update(_file_sha256(label).encode("ascii"))
    digest.update(str(len(images)).encode("ascii"))
    return digest.hexdigest()


def _requested_run_payload(config: dict[str, Any], method: str, seed: int, config_sha: str) -> dict[str, Any]:
    import torch
    import ultralytics

    training = config["training"]
    training_data = _resolve(config["data"]["training"])
    training_data_config = yaml.safe_load(training_data.read_text(encoding="utf-8"))
    provenance_path = _validate_development_provenance(training_data_config, training_data)
    init_model = _resolve(config["data"]["init_model"])
    ssl_artifact = _resolve(config["ssl_artifact"])
    return {
        "config_sha256": config_sha, "method": method, "seed": int(seed),
        "data": str(training_data), "data_yaml_sha256": _file_sha256(training_data),
        "data_provenance": str(provenance_path),
        "data_provenance_sha256": _file_sha256(provenance_path),
        "init_model": str(init_model), "init_model_sha256": _file_sha256(init_model),
        "ssl_artifact": str(ssl_artifact) if method == "anchored_ssl" else None,
        "ssl_artifact_sha256": _file_sha256(ssl_artifact) if method == "anchored_ssl" else None,
        "epochs": int(training["epochs"]), "batch": int(training["batch"]),
        "imgsz": int(training["imgsz"]), "workers": int(training["workers"]),
        "device": str(training["device"]), "lr0": float(training["lr0"]),
        "weight_decay": float(training["weight_decay"]),
        "runtime_versions": {"torch": str(torch.__version__), "ultralytics": ultralytics.__version__},
    }


def _complete(
    run_dir: Path, method: str, config_sha: str | None = None,
    expected_request: dict[str, Any] | None = None,
) -> bool:
    marker = run_dir / ("supervised_baseline.json" if method == "baseline" else "ssl_finetune.json")
    basic = marker.is_file() and (run_dir / "results.csv").is_file() and (run_dir / "weights" / "best.pt").is_file()
    if not basic:
        return False
    try:
        completed = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_pipeline = (
        "supervised_yolo_obb_burn_in" if method == "baseline"
        else "anchored_dinov2_ms_ssl_then_supervised_al_obb"
    )
    if completed.get("pipeline") != expected_pipeline:
        return False
    if Path(str(completed.get("best_checkpoint", ""))).resolve() != (run_dir / "weights" / "best.pt").resolve():
        return False
    if config_sha is None:
        return True
    request_path = run_dir / "requested_run.json"
    if not request_path.is_file():
        return False
    try:
        requested = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if requested.get("config_sha256") != config_sha or requested.get("method") != method:
        return False
    if expected_request is not None and requested != expected_request:
        return False
    if expected_request is not None:
        try:
            marker_fields = {
                "seed": int(completed.get("seed", -1)),
                "epochs": int(completed.get("epochs", -1)),
                "data": str(Path(str(completed.get("data", ""))).resolve()),
                "init_model": str(Path(str(completed.get("init_model", ""))).resolve()),
            }
        except (TypeError, ValueError):
            return False
        if marker_fields != {
            "seed": expected_request["seed"], "epochs": expected_request["epochs"],
            "data": str(Path(expected_request["data"]).resolve()),
            "init_model": str(Path(expected_request["init_model"]).resolve()),
        }:
            return False
        if method == "anchored_ssl" and Path(str(completed.get("ssl_artifact", ""))).resolve() != Path(
            expected_request["ssl_artifact"]
        ).resolve():
            return False
    return True


def _pretraining_complete(
    marker: Path, artifact: Path, expected_config_sha: str, expected_epochs: int | None = None,
) -> bool:
    if not marker.is_file() or not artifact.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("complete") is True
        and payload.get("config_sha256") == expected_config_sha
        and Path(str(payload.get("best_artifact", ""))).resolve() == artifact.resolve()
        and int(payload.get("epochs_completed", 0)) > 0
        and (expected_epochs is None or int(payload.get("epochs_completed", 0)) == expected_epochs)
        and payload.get("final_test_used") is False
        and payload.get("best_artifact_sha256") == _file_sha256(artifact)
        and payload.get("validation_sampling_contract") == "source_group_round_robin_v1"
        and payload.get("optimizer_contract") == "adamw_kernel_matrix_decay_bias_norm_no_decay_v1"
        and isinstance(payload.get("ms_ssl_manifest_sha256"), str)
    )


def _ssl_eligible(
    mean_delta: float, paired_deltas: list[float], minimum: float, require_all_seeds: bool,
) -> bool:
    mean_is_better = mean_delta > 0.0 if minimum == 0.0 else mean_delta >= minimum
    return mean_is_better and (
        not require_all_seeds or all(delta >= minimum for delta in paired_deltas)
    )


def _paired_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Paired summary requires at least one delta.")
    paired_mean = mean(values)
    paired_std = stdev(values) if len(values) > 1 else 0.0
    # Two-sided 95% Student-t critical values for the protocol's small n.
    critical = {2: 12.706204736, 3: 4.30265273}.get(len(values), 1.96)
    half_width = critical * paired_std / (len(values) ** 0.5) if len(values) > 1 else 0.0
    return {
        "n_pairs": len(values), "mean_delta": paired_mean, "std_delta": paired_std,
        "ci95_low": paired_mean - half_width, "ci95_high": paired_mean + half_width,
    }


def _frozen_checkpoint_map(
    rows: list[dict[str, Any]], methods: list[str],
) -> dict[str, dict[str, dict[str, str]]]:
    return {
        str(method): {
            str(row["seed"]): {
                "path": str(row["checkpoint"]),
                "sha256": str(row["checkpoint_sha256"]),
            }
            for row in rows if row["method"] == method
        }
        for method in methods
    }


def _best_validation(run_dir: Path) -> dict[str, float | int]:
    from scripts.run_ssl_publication_study import best_validation_metrics

    return best_validation_metrics(run_dir)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _summarize_validation(config: dict[str, Any], output_dir: Path, config_sha: str) -> dict[str, Any]:
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    rows: list[dict[str, Any]] = []
    for method in config["methods"]:
        for seed in seeds:
            run_dir = output_dir / "01_al_finetune" / method / f"seed{seed}"
            requested = _requested_run_payload(config, method, seed, config_sha)
            if not _complete(run_dir, method, config_sha, requested):
                raise FileNotFoundError(f"Incomplete {method} seed {seed}: {run_dir}")
            metrics = _best_validation(run_dir)
            rows.append(
                {
                    "method": method, "seed": seed, "best_epoch": metrics["best_epoch"],
                    "best_val_map50_95": metrics["metrics/mAP50-95(B)"],
                    "best_val_map50": metrics["metrics/mAP50(B)"],
                    "best_val_precision": metrics["metrics/precision(B)"],
                    "best_val_recall": metrics["metrics/recall(B)"],
                    "checkpoint": str((run_dir / "weights" / "best.pt").resolve()),
                    "checkpoint_sha256": _file_sha256((run_dir / "weights" / "best.pt").resolve()),
                }
            )
    fields = ["method", "seed", "best_epoch", "best_val_map50_95", "best_val_map50", "best_val_precision", "best_val_recall", "checkpoint", "checkpoint_sha256"]
    _write_csv(output_dir / "validation" / "per_seed_validation.csv", rows, fields)
    aggregates: list[dict[str, Any]] = []
    for method in config["methods"]:
        values = [float(row["best_val_map50_95"]) for row in rows if row["method"] == method]
        aggregates.append(
            {
                "method": method, "n_seeds": len(values), "mean_best_val_map50_95": mean(values),
                "std_best_val_map50_95": stdev(values) if len(values) > 1 else 0.0,
            }
        )
    aggregates.sort(key=lambda row: float(row["mean_best_val_map50_95"]), reverse=True)
    _write_csv(
        output_dir / "validation" / "validation_ranking.csv", aggregates,
        ["method", "n_seeds", "mean_best_val_map50_95", "std_best_val_map50_95"],
    )
    baseline = next(row for row in aggregates if row["method"] == "baseline")
    ssl = next(row for row in aggregates if row["method"] == "anchored_ssl")
    deltas = []
    for seed in seeds:
        baseline_value = next(float(row["best_val_map50_95"]) for row in rows if row["method"] == "baseline" and row["seed"] == seed)
        ssl_value = next(float(row["best_val_map50_95"]) for row in rows if row["method"] == "anchored_ssl" and row["seed"] == seed)
        deltas.append({"seed": seed, "anchored_ssl_minus_baseline": ssl_value - baseline_value})
    _write_csv(output_dir / "validation" / "paired_validation_delta.csv", deltas, ["seed", "anchored_ssl_minus_baseline"])
    minimum = float(config["selection"].get("minimum_mean_delta", 0.0))
    mean_delta = float(ssl["mean_best_val_map50_95"]) - float(baseline["mean_best_val_map50_95"])
    require_all = bool(config["selection"].get("require_all_seeds", False))
    paired_values = [float(row["anchored_ssl_minus_baseline"]) for row in deltas]
    paired_statistics = _paired_summary(paired_values)
    eligible = _ssl_eligible(mean_delta, paired_values, minimum, require_all)
    frozen_checkpoints = _frozen_checkpoint_map(rows, list(config["methods"]))
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": config_sha,
        "selection_metric": "metrics/mAP50-95(B)", "selection_data": "AL validation only",
        "final_test_used": False, "seeds": seeds, "all_runs_complete": True,
        "baseline_mean": baseline["mean_best_val_map50_95"], "anchored_ssl_mean": ssl["mean_best_val_map50_95"],
        "anchored_ssl_minus_baseline_mean": mean_delta,
        "paired_delta_statistics": paired_statistics,
        "minimum_required_delta": minimum, "require_all_seeds": require_all,
        "all_seed_deltas_meet_requirement": all(delta >= minimum for delta in paired_values),
        "winner": "anchored_ssl" if eligible else "baseline",
        "anchored_ssl_eligible_for_claim": eligible,
        "frozen_checkpoints": frozen_checkpoints,
        "frozen_for_explicit_test_phase": True,
    }
    _write_json(output_dir / "validation" / "frozen_selection.json", decision)
    print(json.dumps(decision, indent=2), flush=True)
    return decision


def _evaluate_test(config: dict[str, Any], output_dir: Path, config_sha: str, *, reuse: bool) -> None:
    frozen_path = output_dir / "validation" / "frozen_selection.json"
    if not frozen_path.is_file():
        raise FileNotFoundError("Run --phase summarize before the explicit test phase.")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("config_sha256") != config_sha or frozen.get("frozen_for_explicit_test_phase") is not True:
        raise ValueError("Frozen validation selection does not match this protocol config.")
    from ultralytics import YOLO

    training = config["training"]
    test_data = _resolve(config["data"]["final_test"])
    test_split = str(config["data"].get("final_test_split", "test"))
    test_dataset_sha256 = _dataset_split_sha256(test_data, test_split)
    rows: list[dict[str, Any]] = []
    for method in config["methods"]:
        for seed in [int(value) for value in training["seeds"]]:
            run_dir = output_dir / "01_al_finetune" / method / f"seed{seed}"
            checkpoint = run_dir / "weights" / "best.pt"
            requested = _requested_run_payload(config, method, seed, config_sha)
            if not _complete(run_dir, method, config_sha, requested):
                raise FileNotFoundError(f"Missing or protocol-mismatched final-test checkpoint: {checkpoint}")
            frozen_checkpoint = frozen.get("frozen_checkpoints", {}).get(method, {}).get(str(seed), {})
            if Path(str(frozen_checkpoint.get("path", ""))).resolve() != checkpoint.resolve():
                raise ValueError(f"Frozen checkpoint path mismatch for {method} seed {seed}.")
            if frozen_checkpoint.get("sha256") != _file_sha256(checkpoint):
                raise ValueError(f"Frozen checkpoint content changed for {method} seed {seed}.")
            result_path = output_dir / "final_test" / method / f"seed{seed}.json"
            if reuse and result_path.is_file():
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                if existing.get("config_sha256") != config_sha:
                    raise ValueError(f"Existing final-test result belongs to another protocol: {result_path}")
                if Path(str(existing.get("checkpoint", ""))).resolve() != checkpoint.resolve():
                    raise ValueError(f"Existing final-test result points to another checkpoint: {result_path}")
                if existing.get("checkpoint_sha256") != _file_sha256(checkpoint):
                    raise ValueError(f"Existing final-test checkpoint content changed: {result_path}")
                if existing.get("test_data_sha256") != _file_sha256(test_data):
                    raise ValueError(f"Existing final-test data config changed: {result_path}")
                if existing.get("test_dataset_sha256") != test_dataset_sha256:
                    raise ValueError(f"Existing final-test dataset content changed: {result_path}")
                rows.append(existing)
                continue
            if result_path.exists():
                raise FileExistsError(f"Refusing to overwrite final-test result: {result_path}")
            result_path.parent.mkdir(parents=True, exist_ok=True)
            metrics = YOLO(str(checkpoint)).val(
                data=str(test_data), split=str(config["data"].get("final_test_split", "test")),
                imgsz=int(training["imgsz"]), batch=int(training["batch"]), workers=int(training["workers"]),
                device=str(training["device"]), plots=False, save_json=False,
                project=str(output_dir / "final_test" / "ultralytics"), name=f"{method}_seed{seed}", exist_ok=True,
            )
            values = {
                str(key): float(value)
                for key, value in metrics.results_dict.items()
                if isinstance(value, numbers.Real)
            }
            row = {
                "method": method, "seed": seed, "checkpoint": str(checkpoint.resolve()),
                "test_data": str(test_data), "final_test_used_for_selection": False,
                "config_sha256": config_sha,
                "checkpoint_sha256": _file_sha256(checkpoint),
                "test_data_sha256": _file_sha256(test_data),
                "test_dataset_sha256": test_dataset_sha256,
                **values,
            }
            _write_json(result_path, row)
            rows.append(row)
    metric = "metrics/mAP50-95(B)"
    aggregates = []
    for method in config["methods"]:
        values = [float(row[metric]) for row in rows if row["method"] == method]
        aggregates.append(
            {"method": method, "n_seeds": len(values), "mean_test_map50_95": mean(values), "std_test_map50_95": stdev(values) if len(values) > 1 else 0.0}
        )
    _write_csv(output_dir / "final_test" / "aggregate_test.csv", aggregates, ["method", "n_seeds", "mean_test_map50_95", "std_test_map50_95"])
    paired_rows = []
    for seed in [int(value) for value in training["seeds"]]:
        baseline_value = next(float(row[metric]) for row in rows if row["method"] == "baseline" and row["seed"] == seed)
        ssl_value = next(float(row[metric]) for row in rows if row["method"] == "anchored_ssl" and row["seed"] == seed)
        paired_rows.append({"seed": seed, "anchored_ssl_minus_baseline": ssl_value - baseline_value})
    _write_csv(
        output_dir / "final_test" / "paired_test_delta.csv", paired_rows,
        ["seed", "anchored_ssl_minus_baseline"],
    )
    _write_json(
        output_dir / "final_test" / "paired_test_summary.json",
        {
            "metric": metric, "config_sha256": config_sha,
            "final_test_used_for_selection": False,
            **_paired_summary([float(row["anchored_ssl_minus_baseline"]) for row in paired_rows]),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("pretrain", "finetune", "summarize", "development", "test"), default="development")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = _read_config(config_path)
    output_dir = _resolve(config["output_dir"])
    config_sha = _config_hash(config)
    data = config["data"]
    paths = {
        "training": _resolve(data["training"]), "final_test": _resolve(data["final_test"]),
        "init_model": _resolve(data["init_model"]), "pretrain_config": _resolve(config["pretrain_config"]),
        "ssl_artifact": _resolve(config["ssl_artifact"]),
    }
    for name in ("training", "final_test", "init_model", "pretrain_config"):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    training_data_config = yaml.safe_load(paths["training"].read_text(encoding="utf-8"))
    if not isinstance(training_data_config, dict) or not {"train", "val"}.issubset(training_data_config):
        raise ValueError("Development data YAML must define train and val splits.")
    if "test" in training_data_config:
        raise ValueError(
            "Development data YAML must not expose a test split; use a train/validation-only YAML."
        )
    _validate_development_provenance(training_data_config, paths["training"])
    final_test_config = yaml.safe_load(paths["final_test"].read_text(encoding="utf-8"))
    if not isinstance(final_test_config, dict) or str(data.get("final_test_split", "test")) not in final_test_config:
        raise ValueError("Final-test YAML does not define the configured final_test_split.")
    pretrain_config = yaml.safe_load(paths["pretrain_config"].read_text(encoding="utf-8"))
    if not isinstance(pretrain_config, dict):
        raise ValueError("Pretraining config must be a YAML mapping.")
    pretrain_sha = _config_hash(pretrain_config)
    pretrain_output = _resolve(pretrain_config["output_dir"])
    pretrain_marker = pretrain_output / "pretraining_complete.json"
    if paths["ssl_artifact"].resolve() != (pretrain_output / "anchored_ssl_backbone_best.pt").resolve():
        raise ValueError("ssl_artifact must be the best artifact produced by pretrain_config.output_dir.")
    if _resolve(pretrain_config["init_model"]) != paths["init_model"]:
        raise ValueError("Pretraining and supervised comparison must use the same init_model.")
    if args.preflight_only:
        _launch([sys.executable, "-m", "scripts.train_ssl_anchored_dino_yolo_obb", "--config", str(paths["pretrain_config"]), "--preflight-only"])
        print("Protocol preflight passed: development is validation-only; final test requires explicit --phase test.")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "protocol_config.yaml"
    if snapshot.is_file() and snapshot.read_text(encoding="utf-8") != config_path.read_text(encoding="utf-8"):
        raise ValueError("Output directory already belongs to a different protocol config.")
    if not snapshot.is_file():
        _write_text(snapshot, config_path.read_text(encoding="utf-8"))

    if args.phase in {"pretrain", "development"}:
        pretraining_is_complete = _pretraining_complete(
            pretrain_marker, paths["ssl_artifact"], pretrain_sha, int(pretrain_config["epochs"])
        )
        if pretraining_is_complete and args.reuse_existing:
            print(f"Reusing completed SSL artifact: {paths['ssl_artifact']}")
        elif pretraining_is_complete:
            raise FileExistsError("SSL pretraining is complete; use --reuse-existing or a new output directory.")
        else:
            command = [sys.executable, "-m", "scripts.train_ssl_anchored_dino_yolo_obb", "--config", str(paths["pretrain_config"])]
            if (pretrain_output / "training_last.pt").is_file():
                command.append("--resume")
            _launch(command)

    if args.phase in {"finetune", "development"}:
        if not _pretraining_complete(
            pretrain_marker, paths["ssl_artifact"], pretrain_sha, int(pretrain_config["epochs"])
        ):
            raise FileNotFoundError(
                "SSL pretraining lacks a valid completion marker; refusing to fine-tune from a partial artifact."
            )
        training = config["training"]
        for method in config["methods"]:
            for seed in [int(value) for value in training["seeds"]]:
                run_dir = output_dir / "01_al_finetune" / method / f"seed{seed}"
                requested_run = _requested_run_payload(config, method, seed, config_sha)
                if _complete(run_dir, method, config_sha, requested_run):
                    if args.reuse_existing:
                        print(f"Reusing completed {method} seed {seed}: {run_dir}")
                        continue
                    raise FileExistsError(f"Completed run exists: {run_dir}; pass --reuse-existing.")
                common = [
                    "--data", str(paths["training"]), "--init-model", str(paths["init_model"]),
                    "--output-dir", str(run_dir), "--epochs", str(training["epochs"]),
                    "--batch", str(training["batch"]), "--imgsz", str(training["imgsz"]),
                    "--workers", str(training["workers"]), "--device", str(training["device"]),
                    "--seed", str(seed), "--lr0", str(training["lr0"]), "--weight-decay", str(training["weight_decay"]),
                ]
                if method == "baseline":
                    command = [sys.executable, "-m", "scripts.train_supervised_obb", *common]
                else:
                    command = [sys.executable, "-m", "scripts.train_ssl_initialized_obb", "--ssl-artifact", str(paths["ssl_artifact"]), *common]
                request_path = run_dir / "requested_run.json"
                if request_path.is_file():
                    existing_request = json.loads(request_path.read_text(encoding="utf-8"))
                    if existing_request != requested_run:
                        raise ValueError(f"Run directory belongs to different settings: {run_dir}")
                else:
                    _write_json(request_path, requested_run)
                if (run_dir / "weights" / "last.pt").is_file():
                    command.append("--resume")
                _launch(command)

    if args.phase in {"summarize", "development"}:
        _summarize_validation(config, output_dir, config_sha)
    elif args.phase == "test":
        _evaluate_test(config, output_dir, config_sha, reuse=args.reuse_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
