"""Evaluate a frozen three-seed YOLO-OBB ensemble on AL validation only.

The evaluator is intentionally limited to the deduplicated development
validation split.  It verifies the baseline protocol, provenance, checkpoint
contents, training settings, and file hashes before concatenating the raw OBB
predictions from seeds 7/17/42.  Ultralytics then applies its ordinary
class-aware rotated NMS and metric implementation once to the combined
predictions.

There is deliberately no test path, test phase, or arbitrary data override in
this command.  The existing AL test split has already been observed and is not
valid for selecting this exploratory ensemble.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import numbers
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch
import yaml
from ultralytics.nn.tasks import Ensemble


ROOT = Path(__file__).resolve().parents[1]
METRIC = "metrics/mAP50-95(B)"
SEEDS = (7, 17, 42)
COMPARISON_FIELDS = (
    "entry",
    "seed",
    "map50_95",
    "delta_vs_baseline_mean",
    "delta_vs_best_single",
)


class CompatibleOBBValidationEnsemble(Ensemble):
    """Ultralytics prediction-concatenation ensemble with backend compatibility.

    Ultralytics 8.4.63's :class:`PyTorchBackend` always supplies an ``embed``
    keyword, but its stock :class:`Ensemble.forward` signature does not accept
    that keyword.  Validation does not request embeddings, so accepting only
    ``None`` or an empty sequence and otherwise preserving the stock
    concatenate-then-NMS behavior is the narrow, correct compatibility fix.
    """

    def forward(
        self,
        x: torch.Tensor,
        augment: bool = False,
        profile: bool = False,
        visualize: bool = False,
        embed: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if embed not in (None, (), []):
            raise ValueError("OBB validation ensemble does not support feature embeddings.")
        if kwargs:
            raise TypeError(
                "Unexpected ensemble forward keyword(s): " + ", ".join(sorted(kwargs))
            )
        predictions: list[torch.Tensor] = []
        for module in self:
            output = module(
                x,
                augment=augment,
                profile=profile,
                visualize=visualize,
            )
            prediction = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(prediction, torch.Tensor) or prediction.ndim != 3:
                raise TypeError(
                    "Every OBB ensemble member must return a rank-3 prediction tensor."
                )
            predictions.append(prediction)
        if not predictions:
            raise RuntimeError("Cannot run an empty OBB ensemble.")
        reference_shape = predictions[0].shape[:2]
        if any(value.shape[:2] != reference_shape for value in predictions[1:]):
            raise ValueError("OBB ensemble members returned incompatible prediction shapes.")
        return torch.cat(predictions, dim=2), None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify every frozen input without loading the ensemble or running validation.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Verify and report an already completed evaluation instead of overwriting it.",
    )
    return parser.parse_args(argv)


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


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _forbidden_test_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            location = f"{prefix}.{key}" if prefix else key
            normalized = key.lower().replace("-", "_").replace(" ", "_")
            if "test" in normalized:
                found.append(location)
            found.extend(_forbidden_test_keys(child, location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_test_keys(child, f"{prefix}[{index}]"))
    return found


def _require_exact_keys(mapping: dict[str, Any], required: set[str], name: str) -> None:
    missing = required.difference(mapping)
    if missing:
        raise ValueError(f"{name} is missing {sorted(missing)}.")


def _same_number(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def read_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    forbidden = _forbidden_test_keys(config)
    if forbidden:
        raise ValueError(
            "Validation-only ensemble config forbids test-related keys: " + ", ".join(forbidden)
        )
    _require_exact_keys(
        config,
        {
            "study_name", "output_dir", "data", "baseline", "evaluation",
            "selection", "reference", "runtime",
        },
        "config",
    )
    for name in ("data", "baseline", "evaluation", "selection", "reference", "runtime"):
        if not isinstance(config[name], dict):
            raise ValueError(f"{name} must be a mapping.")
    _require_exact_keys(
        config["data"],
        {"training", "training_yaml_sha256", "provenance_sha256"},
        "data",
    )
    _require_exact_keys(
        config["baseline"],
        {
            "config", "config_sha256", "protocol_dir", "protocol_snapshot",
            "validation_rows", "validation_rows_sha256", "validation_marker",
            "validation_marker_sha256", "checkpoints",
        },
        "baseline",
    )
    _require_exact_keys(
        config["evaluation"],
        {
            "split", "imgsz", "batch", "workers", "device", "conf", "iou",
            "max_det", "half", "augment", "plots",
        },
        "evaluation",
    )
    _require_exact_keys(
        config["selection"], {"metric", "validation_only"}, "selection"
    )
    _require_exact_keys(
        config["reference"],
        {
            "baseline_mean_map50_95", "best_single_map50_95",
            "ensemble_map50_95", "absolute_tolerance",
        },
        "reference",
    )
    _require_exact_keys(
        config["runtime"],
        {
            "python", "torch", "ultralytics", "cuda", "cudnn",
            "gpu", "checkpoint_ultralytics",
        },
        "runtime",
    )

    if config["selection"]["metric"] != METRIC:
        raise ValueError(f"selection.metric must be {METRIC}.")
    if config["selection"]["validation_only"] is not True:
        raise ValueError("selection.validation_only must be true.")

    evaluation = config["evaluation"]
    expected_evaluation = {
        "split": "val", "imgsz": 640, "batch": 16, "workers": 12,
        "device": "0", "conf": 0.01, "iou": 0.7, "max_det": 300,
        "half": False, "augment": False, "plots": False,
    }
    for name, expected in expected_evaluation.items():
        observed = evaluation.get(name)
        if isinstance(expected, bool):
            matches = observed is expected
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            matches = _same_number(observed, expected)
        else:
            matches = str(observed) == str(expected)
        if not matches:
            raise ValueError(f"evaluation.{name} must be {expected!r}; found {observed!r}.")

    raw_checkpoints = config["baseline"]["checkpoints"]
    if not isinstance(raw_checkpoints, dict) or set(map(str, raw_checkpoints)) != {
        str(seed) for seed in SEEDS
    }:
        raise ValueError("baseline.checkpoints must contain exactly seeds 7, 17, and 42.")
    for seed in SEEDS:
        record = raw_checkpoints.get(str(seed), raw_checkpoints.get(seed))
        if not isinstance(record, dict):
            raise ValueError(f"baseline.checkpoints.{seed} must be a mapping.")
        _require_exact_keys(record, {"path", "sha256", "best_val_map50_95"}, f"checkpoint {seed}")
        if len(str(record["sha256"])) != 64:
            raise ValueError(f"checkpoint {seed} has an invalid SHA-256 value.")

    reference = config["reference"]
    tolerance = float(reference["absolute_tolerance"])
    if not 0.0 < tolerance <= 1e-4:
        raise ValueError("reference.absolute_tolerance must be in (0, 1e-4].")
    for name in ("baseline_mean_map50_95", "best_single_map50_95", "ensemble_map50_95"):
        value = float(reference[name])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"reference.{name} must be in [0, 1].")
    return config


def _read_validation_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(SEEDS):
        raise ValueError(f"Frozen validation table must contain exactly three rows: {path}")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            parsed.append(
                {
                    "method": str(row["method"]),
                    "seed": int(row["seed"]),
                    "best_epoch": int(row["best_epoch"]),
                    "best_val_map50_95": float(row["best_val_map50_95"]),
                    "checkpoint": str(row["checkpoint"]),
                    "checkpoint_sha256": str(row["checkpoint_sha256"]),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed frozen validation row in {path}") from exc
    parsed.sort(key=lambda item: int(item["seed"]))
    if [int(row["seed"]) for row in parsed] != list(SEEDS):
        raise ValueError("Frozen validation table must contain seeds 7, 17, and 42.")
    if any(row["method"] != "baseline" for row in parsed):
        raise ValueError("Frozen validation rows must all be baseline rows.")
    return parsed


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


def _audit_checkpoint_train_args(
    metadata: dict[str, Any], *, seed: int, training_data: Path, init_model: Path,
) -> None:
    actual = metadata["train_args"]
    expected: dict[str, Any] = {
        "task": "obb", "data": str(training_data), "model": str(init_model),
        "epochs": 100, "batch": 16, "imgsz": 640, "workers": 12,
        "device": "0", "seed": seed, "optimizer": "AdamW",
        "deterministic": True, "fraction": 1.0, "cos_lr": True,
        "close_mosaic": 10, "amp": True, "val": True,
        "lr0": 0.001, "lrf": 0.01, "weight_decay": 0.0005,
        "warmup_epochs": 3.0, "hsv_h": 0.015, "hsv_s": 0.7,
        "hsv_v": 0.4, "degrees": 15.0, "flipud": 0.5,
        "fliplr": 0.5, "mosaic": 1.0,
    }
    for name, wanted in expected.items():
        observed = actual.get(name)
        if name in {"data", "model"}:
            matches = Path(str(observed)).resolve() == Path(str(wanted)).resolve()
        elif isinstance(wanted, bool):
            matches = observed is wanted
        elif isinstance(wanted, (int, float)) and not isinstance(wanted, bool):
            matches = _same_number(observed, wanted)
        else:
            matches = str(observed) == str(wanted)
        if not matches:
            raise ValueError(
                f"Frozen seed {seed} train arg {name!r} is {observed!r}; expected {wanted!r}."
            )


def _validate_training_data(data_yaml: Path) -> Path:
    from scripts.run_full_al_640_baseline import _validate_training_data as audit

    return audit(data_yaml)


def audit_inputs(config: dict[str, Any]) -> dict[str, Any]:
    import ultralytics

    configured_runtime = config["runtime"]
    actual_runtime = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version() or 0),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for name, observed in actual_runtime.items():
        expected = configured_runtime[name]
        matches = int(observed) == int(expected) if name == "cudnn" else str(observed) == str(expected)
        if not matches:
            raise ValueError(
                f"Runtime {name!r} is {observed!r}; frozen evaluation requires {expected!r}."
            )

    data_cfg = config["data"]
    baseline_cfg = config["baseline"]
    paths = {
        "training_data": resolve(str(data_cfg["training"])),
        "baseline_config": resolve(str(baseline_cfg["config"])),
        "protocol_dir": resolve(str(baseline_cfg["protocol_dir"])),
        "protocol_snapshot": resolve(str(baseline_cfg["protocol_snapshot"])),
        "validation_rows": resolve(str(baseline_cfg["validation_rows"])),
        "validation_marker": resolve(str(baseline_cfg["validation_marker"])),
    }
    for name, path in paths.items():
        if name == "protocol_dir":
            if not path.is_dir():
                raise FileNotFoundError(path)
        elif not path.is_file():
            raise FileNotFoundError(path)

    if sha256(paths["training_data"]) != str(data_cfg["training_yaml_sha256"]):
        raise ValueError("Development training YAML content changed.")
    provenance = _validate_training_data(paths["training_data"])
    if sha256(provenance) != str(data_cfg["provenance_sha256"]):
        raise ValueError("Development provenance content changed.")

    baseline_config = read_yaml(paths["baseline_config"])
    snapshot_config = read_yaml(paths["protocol_snapshot"])
    expected_config_hash = str(baseline_cfg["config_sha256"])
    if canonical_hash(baseline_config) != expected_config_hash:
        raise ValueError("Baseline config no longer matches its frozen canonical hash.")
    if canonical_hash(snapshot_config) != expected_config_hash:
        raise ValueError("Baseline protocol snapshot does not match the frozen config.")
    if resolve(str(baseline_config.get("output_dir", ""))) != paths["protocol_dir"]:
        raise ValueError("Baseline config output_dir does not match baseline.protocol_dir.")
    if resolve(str(baseline_config.get("data", {}).get("training", ""))) != paths["training_data"]:
        raise ValueError("Baseline config does not reference the frozen development data.")

    if sha256(paths["validation_rows"]) != str(baseline_cfg["validation_rows_sha256"]):
        raise ValueError("Frozen validation table content changed.")
    if sha256(paths["validation_marker"]) != str(baseline_cfg["validation_marker_sha256"]):
        raise ValueError("Frozen validation marker content changed.")
    rows = _read_validation_rows(paths["validation_rows"])
    marker = read_json(paths["validation_marker"])
    if marker.get("config_sha256") != expected_config_hash:
        raise ValueError("Frozen validation marker has a different baseline config hash.")
    if marker.get("final_test_used") is not False:
        raise ValueError("Frozen baseline validation marker is not development-only.")
    if marker.get("seeds") != list(SEEDS):
        raise ValueError("Frozen baseline validation marker has different seeds.")
    if int(marker.get("imgsz", 0)) != 640 or int(marker.get("batch", 0)) != 16:
        raise ValueError("Frozen baseline validation marker has different image/batch settings.")

    init_model = resolve(str(baseline_config.get("data", {}).get("init_model", "")))
    if not init_model.is_file():
        raise FileNotFoundError(init_model)
    config_records = baseline_cfg["checkpoints"]
    audited_checkpoints: list[dict[str, Any]] = []
    for row in rows:
        seed = int(row["seed"])
        configured = config_records.get(str(seed), config_records.get(seed))
        checkpoint = resolve(str(configured["path"]))
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        expected_checkpoint_hash = str(configured["sha256"])
        if sha256(checkpoint) != expected_checkpoint_hash:
            raise ValueError(f"Frozen checkpoint content changed for seed {seed}.")
        if resolve(str(row["checkpoint"])) != checkpoint:
            raise ValueError(f"Frozen validation table checkpoint path mismatch for seed {seed}.")
        if row["checkpoint_sha256"] != expected_checkpoint_hash:
            raise ValueError(f"Frozen validation table checkpoint hash mismatch for seed {seed}.")
        if not _same_number(row["best_val_map50_95"], configured["best_val_map50_95"]):
            raise ValueError(f"Frozen validation metric mismatch for seed {seed}.")
        matching_marker = [
            item for item in marker.get("checkpoints", [])
            if isinstance(item, dict) and int(item.get("seed", -1)) == seed
        ]
        if len(matching_marker) != 1:
            raise ValueError(f"Frozen validation marker lacks exactly one seed {seed} record.")
        marker_row = matching_marker[0]
        if (
            resolve(str(marker_row.get("checkpoint", ""))) != checkpoint
            or marker_row.get("checkpoint_sha256") != expected_checkpoint_hash
            or not _same_number(marker_row.get("best_val_map50_95"), row["best_val_map50_95"])
        ):
            raise ValueError(f"Frozen validation marker mismatch for seed {seed}.")

        run_dir = checkpoint.parents[1]
        request = read_json(run_dir / "requested_run.json")
        completion = read_json(run_dir / "supervised_baseline.json")
        if (
            request.get("config_sha256") != expected_config_hash
            or int(request.get("seed", -1)) != seed
            or completion.get("test_used_for_selection") is not False
            or resolve(str(completion.get("best_checkpoint", ""))) != checkpoint
        ):
            raise ValueError(f"Baseline run contract mismatch for seed {seed}.")
        metadata = _checkpoint_metadata(checkpoint)
        _audit_checkpoint_train_args(
            metadata, seed=seed, training_data=paths["training_data"], init_model=init_model
        )
        checkpoint_metric = metadata["train_metrics"].get(METRIC)
        if not _same_number(checkpoint_metric, row["best_val_map50_95"], tolerance=5e-7):
            raise ValueError(f"Checkpoint metric mismatch for seed {seed}.")
        audited_checkpoints.append(
            {
                "seed": seed,
                "path": str(checkpoint),
                "sha256": expected_checkpoint_hash,
                "best_val_map50_95": float(row["best_val_map50_95"]),
                "best_epoch": int(row["best_epoch"]),
                "ultralytics_version": metadata["version"],
            }
        )

    checkpoint_versions = {str(record["ultralytics_version"]) for record in audited_checkpoints}
    expected_checkpoint_version = str(configured_runtime["checkpoint_ultralytics"])
    if checkpoint_versions != {expected_checkpoint_version}:
        raise ValueError(
            "Frozen checkpoints were not all produced by required Ultralytics version "
            f"{expected_checkpoint_version}: {sorted(checkpoint_versions)}"
        )

    values = [float(row["best_val_map50_95"]) for row in audited_checkpoints]
    baseline_mean = mean(values)
    best_single = max(values)
    reference = config["reference"]
    if not _same_number(baseline_mean, reference["baseline_mean_map50_95"]):
        raise ValueError("Configured baseline mean does not match frozen seed results.")
    if not _same_number(best_single, reference["best_single_map50_95"]):
        raise ValueError("Configured best-single value does not match frozen seed results.")
    if not _same_number(marker.get("mean_best_val_map50_95"), baseline_mean):
        raise ValueError("Frozen validation marker baseline mean is inconsistent.")

    return {
        "training_data": paths["training_data"],
        "training_data_sha256": sha256(paths["training_data"]),
        "provenance": provenance,
        "provenance_sha256": sha256(provenance),
        "baseline_config": paths["baseline_config"],
        "baseline_config_sha256": expected_config_hash,
        "validation_rows": paths["validation_rows"],
        "validation_marker": paths["validation_marker"],
        "checkpoints": audited_checkpoints,
        "baseline_mean": baseline_mean,
        "best_single": best_single,
        "init_model": init_model,
        "runtime": actual_runtime,
    }


def _assert_member_compatibility(models: list[torch.nn.Module]) -> None:
    if len(models) != len(SEEDS):
        raise ValueError("The frozen validation ensemble requires exactly three members.")
    first = models[0]
    reference_names = getattr(first, "names", None)
    reference_nc = getattr(first, "nc", None)
    reference_stride = tuple(float(value) for value in first.stride.tolist())
    reference_end2end = bool(getattr(first, "end2end", False))
    if reference_end2end:
        raise ValueError("This evaluator is frozen for non-end-to-end YOLO11 OBB checkpoints.")
    for index, model in enumerate(models[1:], start=2):
        if getattr(model, "names", None) != reference_names:
            raise ValueError(f"Ensemble member {index} has different class names.")
        if getattr(model, "nc", None) != reference_nc:
            raise ValueError(f"Ensemble member {index} has a different class count.")
        stride = tuple(float(value) for value in model.stride.tolist())
        if stride != reference_stride:
            raise ValueError(f"Ensemble member {index} has different strides.")
        if bool(getattr(model, "end2end", False)) != reference_end2end:
            raise ValueError(f"Ensemble member {index} has a different end-to-end mode.")


def build_runner(checkpoints: list[Path]) -> Any:
    """Load one YOLO wrapper and replace its model with the verified ensemble."""
    from ultralytics import YOLO
    from ultralytics.nn.tasks import load_checkpoint

    if len(checkpoints) != len(SEEDS):
        raise ValueError("Exactly three checkpoint paths are required.")
    runner = YOLO(str(checkpoints[0]))
    models: list[torch.nn.Module] = [runner.model.float().eval()]
    for checkpoint in checkpoints[1:]:
        model, _ = load_checkpoint(str(checkpoint), device="cpu", fuse=False)
        models.append(model.float().eval())
    _assert_member_compatibility(models)

    ensemble = CompatibleOBBValidationEnsemble()
    ensemble.extend(models)
    first = models[0]
    for attribute in ("names", "nc", "yaml", "args", "stride"):
        setattr(ensemble, attribute, getattr(first, attribute))
    ensemble.task = "obb"
    ensemble.end2end = False
    ensemble.pt_path = [str(path) for path in checkpoints]
    runner.model = ensemble
    runner.predictor = None
    return runner


def comparison_rows(context: dict[str, Any], ensemble_value: float) -> list[dict[str, Any]]:
    baseline_mean = float(context["baseline_mean"])
    best_single = float(context["best_single"])
    rows: list[dict[str, Any]] = []
    for record in context["checkpoints"]:
        value = float(record["best_val_map50_95"])
        rows.append(
            {
                "entry": "baseline_seed",
                "seed": int(record["seed"]),
                "map50_95": value,
                "delta_vs_baseline_mean": value - baseline_mean,
                "delta_vs_best_single": value - best_single,
            }
        )
    rows.extend(
        [
            {
                "entry": "baseline_mean", "seed": "", "map50_95": baseline_mean,
                "delta_vs_baseline_mean": 0.0,
                "delta_vs_best_single": baseline_mean - best_single,
            },
            {
                "entry": "best_single", "seed": "", "map50_95": best_single,
                "delta_vs_baseline_mean": best_single - baseline_mean,
                "delta_vs_best_single": 0.0,
            },
            {
                "entry": "three_seed_ensemble", "seed": "", "map50_95": ensemble_value,
                "delta_vs_baseline_mean": ensemble_value - baseline_mean,
                "delta_vs_best_single": ensemble_value - best_single,
            },
        ]
    )
    return rows


def _completed_result(output_dir: Path, config_sha: str) -> dict[str, Any] | None:
    marker_path = output_dir / "evaluation_complete.json"
    metrics_path = output_dir / "ensemble_metrics.json"
    comparison_path = output_dir / "comparison.csv"
    if not all(path.is_file() for path in (marker_path, metrics_path, comparison_path)):
        return None
    marker = read_json(marker_path)
    if (
        marker.get("complete") is not True
        or marker.get("config_sha256") != config_sha
        or marker.get("ensemble_metrics_sha256") != sha256(metrics_path)
        or marker.get("comparison_sha256") != sha256(comparison_path)
    ):
        return None
    return read_json(metrics_path)


def evaluate(
    config: dict[str, Any], context: dict[str, Any], config_path: Path, config_sha: str,
    *, reuse_existing: bool,
) -> dict[str, Any]:
    output_dir = resolve(str(config["output_dir"]))
    if output_dir.exists() and any(output_dir.iterdir()):
        completed = _completed_result(output_dir, config_sha)
        if reuse_existing and completed is not None:
            print(json.dumps(completed, indent=2), flush=True)
            return completed
        if completed is not None:
            raise FileExistsError(f"Completed evaluation exists: {output_dir}; use --reuse-existing.")
        raise FileExistsError(f"Refusing to reuse partial output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "protocol_config.yaml"
    snapshot.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    checkpoint_paths = [Path(record["path"]) for record in context["checkpoints"]]
    runner = build_runner(checkpoint_paths)
    settings = config["evaluation"]
    metrics = runner.val(
        data=str(context["training_data"]),
        split="val",
        imgsz=int(settings["imgsz"]),
        batch=int(settings["batch"]),
        workers=int(settings["workers"]),
        device=str(settings["device"]),
        conf=float(settings["conf"]),
        iou=float(settings["iou"]),
        max_det=int(settings["max_det"]),
        half=bool(settings["half"]),
        augment=bool(settings["augment"]),
        plots=bool(settings["plots"]),
        save_json=False,
        verbose=False,
        project=str(output_dir / "ultralytics"),
        name="ensemble_validation",
        exist_ok=False,
    )
    numeric_metrics = {
        str(name): float(value)
        for name, value in (getattr(metrics, "results_dict", {}) or {}).items()
        if isinstance(value, numbers.Real) and math.isfinite(float(value))
    }
    if METRIC not in numeric_metrics:
        raise RuntimeError(f"Ultralytics validation did not return finite {METRIC}.")
    ensemble_value = numeric_metrics[METRIC]
    reference = config["reference"]
    reference_value = float(reference["ensemble_map50_95"])
    reference_error = ensemble_value - reference_value
    reference_match = abs(reference_error) <= float(reference["absolute_tolerance"])

    import ultralytics

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_name": str(config["study_name"]),
        "config_sha256": config_sha,
        "validation_only": True,
        "selection_data": "deduplicated AL validation only",
        "current_al_test_forbidden": True,
        "ensemble_method": "concatenate raw predictions, then Ultralytics class-aware rotated NMS",
        "compatibility_fix": "accept empty embed keyword without changing member predictions",
        "data": {
            "yaml": str(context["training_data"]),
            "yaml_sha256": context["training_data_sha256"],
            "split": "val",
            "provenance": str(context["provenance"]),
            "provenance_sha256": context["provenance_sha256"],
        },
        "checkpoints": context["checkpoints"],
        "evaluation": dict(settings),
        "runtime": dict(context["runtime"]),
        "metrics": numeric_metrics,
        "comparison": {
            "baseline_mean_map50_95": float(context["baseline_mean"]),
            "best_single_map50_95": float(context["best_single"]),
            "ensemble_map50_95": ensemble_value,
            "ensemble_minus_baseline_mean": ensemble_value - float(context["baseline_mean"]),
            "ensemble_minus_best_single": ensemble_value - float(context["best_single"]),
        },
        "reference_exploratory_result": {
            "map50_95": reference_value,
            "absolute_tolerance": float(reference["absolute_tolerance"]),
            "observed_minus_reference": reference_error,
            "matches": reference_match,
        },
    }
    metrics_path = output_dir / "ensemble_metrics.json"
    comparison_path = output_dir / "comparison.csv"
    write_json(metrics_path, payload)
    write_csv(
        comparison_path,
        comparison_rows(context, ensemble_value),
        COMPARISON_FIELDS,
    )
    print(json.dumps(payload, indent=2), flush=True)
    if not reference_match:
        write_json(
            output_dir / "evaluation_failed_reference.json",
            {
                "complete": False,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": config_sha,
                "validation_only": True,
                "observed_minus_reference": reference_error,
                "ensemble_metrics_sha256": sha256(metrics_path),
                "comparison_sha256": sha256(comparison_path),
            },
        )
        raise RuntimeError(
            "Reproduced ensemble metric differs from the sealed exploratory reference by "
            f"{reference_error:+.10f}. Outputs were preserved for diagnosis."
        )
    write_json(
        output_dir / "evaluation_complete.json",
        {
            "complete": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_sha,
            "validation_only": True,
            "reference_match": True,
            "ensemble_metrics_sha256": sha256(metrics_path),
            "comparison_sha256": sha256(comparison_path),
        },
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = resolve(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = read_config(config_path)
    config_sha = canonical_hash(config)
    context = audit_inputs(config)
    preflight = {
        "preflight_passed": True,
        "validation_only": True,
        "config_sha256": config_sha,
        "data": str(context["training_data"]),
        "provenance": str(context["provenance"]),
        "seeds": list(SEEDS),
        "checkpoint_sha256": {
            str(record["seed"]): record["sha256"] for record in context["checkpoints"]
        },
        "baseline_mean_map50_95": context["baseline_mean"],
        "best_single_map50_95": context["best_single"],
        "reference_ensemble_map50_95": float(config["reference"]["ensemble_map50_95"]),
        "runtime": context["runtime"],
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if args.preflight_only:
        return 0
    evaluate(
        config, context, config_path, config_sha,
        reuse_existing=bool(args.reuse_existing),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
