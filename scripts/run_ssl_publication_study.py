"""Run a validation-gated, publication-ready pseudo-OBB study.

The workflow is deliberately staged so external-test measurements cannot
influence pseudo-label hyperparameter selection:

1. Pilot: one seed at 10%, validation-only ablations.
2. Confirmation: the chosen pseudo configuration plus matched supervised
   controls at 10% over three seeds, followed by external testing.
3. Label efficiency: the fixed configuration versus the COCO-initialized
   supervised baseline for nested label fractions and the same seeds.

Every stage is resumable.  Checkpoints, configuration snapshots, dataset
audits, commands, per-seed metrics, aggregates, and paired comparisons are
kept under one study directory for future paper preparation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
METRIC_FIELDS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)
RUN_MANIFEST_FIELDS = (
    "timestamp_utc",
    "stage",
    "fraction_percent",
    "seed",
    "method",
    "status",
    "run_dir",
    "checkpoint",
    "command",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def normalized(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())


def format_command(command: Iterable[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def append_manifest(metadata_dir: Path, **row: object) -> None:
    """Append an immutable command/status event for later reproduction."""
    path = metadata_dir / "run_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row_data = {field: str(row.get(field, "")) for field in RUN_MANIFEST_FIELDS}
    row_data["timestamp_utc"] = str(row.get("timestamp_utc", utc_now()))
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_MANIFEST_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row_data)


def run_command(
    command: list[str],
    *,
    metadata_dir: Path,
    stage: str,
    fraction_percent: int,
    seed: int,
    method: str,
    run_dir: Path,
) -> None:
    command_text = format_command(command)
    print("\n$ " + command_text, flush=True)
    append_manifest(
        metadata_dir,
        stage=stage,
        fraction_percent=fraction_percent,
        seed=seed,
        method=method,
        status="started",
        run_dir=run_dir,
        command=command_text,
    )
    completed = subprocess.run(command, cwd=ROOT)
    status = "completed" if completed.returncode == 0 else f"failed_exit_{completed.returncode}"
    append_manifest(
        metadata_dir,
        stage=stage,
        fraction_percent=fraction_percent,
        seed=seed,
        method=method,
        status=status,
        run_dir=run_dir,
        command=command_text,
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)


def marker_checkpoint(run_dir: Path, *, pseudo: bool) -> Path | None:
    """Return a completed checkpoint only when its provenance marker exists."""
    if pseudo:
        finished_dir = run_dir.with_name(run_dir.name + "_finetune")
        marker = finished_dir / "pseudo_obb_handoff.json"
        candidate = finished_dir / "weights" / "best.pt"
    else:
        marker = run_dir / "supervised_baseline.json"
        candidate = run_dir / "weights" / "best.pt"
    return candidate.resolve() if marker.is_file() and candidate.is_file() else None


def last_checkpoint(run_dir: Path) -> Path | None:
    candidate = run_dir / "weights" / "last.pt"
    return candidate.resolve() if candidate.is_file() else None


def numeric(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed


def find_metric_column(fieldnames: Iterable[str], metric: str) -> str | None:
    target = normalized(metric)
    aliases = {
        normalized("metrics/mAP50-95(B)"): ("metricsmap5095b", "metricsmap5095"),
        normalized("metrics/mAP50(B)"): ("metricsmap50b", "metricsmap50"),
        normalized("metrics/precision(B)"): ("metricsprecisionb", "metricsprecision"),
        normalized("metrics/recall(B)"): ("metricsrecallb", "metricsrecall"),
    }.get(target, (target,))
    for field in fieldnames:
        if normalized(field) in aliases:
            return field
    return None


def best_validation_metrics(run_dir: Path) -> dict[str, float | int]:
    """Read the validation-best epoch from an Ultralytics ``results.csv``."""
    csv_path = run_dir / "results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing validation curve: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        map_column = find_metric_column(fields, "metrics/mAP50-95(B)")
        if map_column is None:
            raise ValueError(f"No validation mAP50-95 column in {csv_path}")
        metric_columns = {metric: find_metric_column(fields, metric) for metric in METRIC_FIELDS}
        best_row: dict[str, str] | None = None
        best_score = float("-inf")
        for row in reader:
            score = numeric(row.get(map_column))
            if not math.isnan(score) and score > best_score:
                best_score = score
                best_row = row
    if best_row is None:
        raise RuntimeError(f"No numeric validation rows in {csv_path}")
    result: dict[str, float | int] = {"best_epoch": int(float(best_row.get("epoch", 0)))}
    for metric, column in metric_columns.items():
        result[metric] = numeric(best_row.get(column)) if column else float("nan")
    return result


def load_thresholds(path: Path) -> str:
    thresholds = read_json(path).get("thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError(f"Calibration file has no thresholds: {path}")
    return ",".join(f"{float(value):.4f}" for value in thresholds)


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def test_row_key(row: dict[str, object]) -> tuple[str, int, str, int]:
    return (
        str(row.get("stage", "")),
        int(float(row.get("fraction_percent", 0))),
        str(row.get("method", "")),
        int(float(row.get("seed", 0))),
    )


def upsert_test_row(rows: list[dict[str, object]], row: dict[str, object]) -> None:
    key = test_row_key(row)
    for index, existing in enumerate(rows):
        if test_row_key(existing) == key:
            rows[index] = row
            return
    rows.append(row)


def sample_mean_std(values: list[float]) -> tuple[float, float]:
    usable = [value for value in values if not math.isnan(value)]
    if not usable:
        return float("nan"), float("nan")
    return mean(usable), stdev(usable) if len(usable) > 1 else 0.0


def test_result_fields() -> list[str]:
    return [
        "stage",
        "fraction_percent",
        "method",
        "seed",
        "checkpoint",
        "test_data",
        "source",
        *METRIC_FIELDS,
    ]


def write_test_reports(
    rows: list[dict[str, object]],
    *,
    output_dir: Path,
    selected_method: str | None = None,
) -> None:
    """Write raw, aggregate, and paired final-test reports incrementally."""
    fields = test_result_fields()
    ordered = sorted(rows, key=lambda row: test_row_key(row))
    write_rows(output_dir / "per_seed_external_test_metrics.csv", ordered, fields)

    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in ordered:
        key = (str(row["stage"]), int(float(row["fraction_percent"])), str(row["method"]))
        grouped.setdefault(key, []).append(row)
    aggregate_rows: list[dict[str, object]] = []
    aggregate_fields = ["stage", "fraction_percent", "method", "n_seeds"]
    for metric in METRIC_FIELDS:
        aggregate_fields.extend([f"{metric}/mean", f"{metric}/std"])
    for (stage, fraction, method), subset in sorted(grouped.items()):
        aggregate: dict[str, object] = {
            "stage": stage,
            "fraction_percent": fraction,
            "method": method,
            "n_seeds": len(subset),
        }
        for metric in METRIC_FIELDS:
            average, deviation = sample_mean_std([numeric(item.get(metric)) for item in subset])
            aggregate[f"{metric}/mean"] = average
            aggregate[f"{metric}/std"] = deviation
        aggregate_rows.append(aggregate)
    write_rows(output_dir / "aggregate_external_test_metrics.csv", aggregate_rows, aggregate_fields)

    if selected_method is None:
        return
    for reference, filename in (
        ("ultralytics_burnin", "paired_ssl_minus_burnin.csv"),
        ("ultralytics_matched", "paired_ssl_minus_matched.csv"),
    ):
        paired_rows: list[dict[str, object]] = []
        paired_fields = ["stage", "fraction_percent", "seed", *METRIC_FIELDS]
        keys = {(str(row["stage"]), int(float(row["fraction_percent"]))) for row in ordered}
        for stage, fraction in sorted(keys):
            pseudo = {
                int(float(row["seed"])): row
                for row in ordered
                if str(row["stage"]) == stage
                and int(float(row["fraction_percent"])) == fraction
                and str(row["method"]) == selected_method
            }
            baseline = {
                int(float(row["seed"])): row
                for row in ordered
                if str(row["stage"]) == stage
                and int(float(row["fraction_percent"])) == fraction
                and str(row["method"]) == reference
            }
            for seed in sorted(set(pseudo) & set(baseline)):
                paired_rows.append(
                    {
                        "stage": stage,
                        "fraction_percent": fraction,
                        "seed": seed,
                        **{
                            metric: numeric(pseudo[seed].get(metric)) - numeric(baseline[seed].get(metric))
                            for metric in METRIC_FIELDS
                        },
                    }
                )
        if paired_rows:
            write_rows(output_dir / filename, paired_rows, paired_fields)


def select_candidate(rows: list[dict[str, object]]) -> dict[str, object]:
    """Pick the strongest SSL candidate using validation mAP50-95 only."""
    eligible = [row for row in rows if bool(row.get("eligible_for_selection", False))]
    if not eligible:
        raise RuntimeError("Pilot has no eligible pseudo-label candidates.")
    valid = [row for row in eligible if not math.isnan(numeric(row.get("metrics/mAP50-95(B)")))]
    if not valid:
        raise RuntimeError("No eligible pilot candidate produced a validation mAP50-95 value.")
    return max(valid, key=lambda row: numeric(row.get("metrics/mAP50-95(B)")))


class Study:
    def __init__(self, config_path: Path, config: dict[str, Any], output_dir: Path, *, plots: bool) -> None:
        self.config_path = config_path
        self.config = config
        self.output_dir = output_dir
        self.metadata_dir = output_dir / "00_metadata"
        self.plots = plots
        self.data_config = self.required_mapping("data")
        self.training = self.required_mapping("training")
        self.calibration = self.required_mapping("calibration")
        self.execution = self.required_mapping("execution")
        self.clean_data = resolve_path(self.data_config.get("clean_data", "configs/poultry_clean_raw_90_10.yaml"))
        self.subset_config_dir = resolve_path(self.data_config["subset_config_dir"])
        self.subset_output_dir = resolve_path(
            self.data_config.get("subset_output_dir", "datasets/poultry_clean_label_subsets")
        )
        self.final_test_data = resolve_path(self.data_config["final_test_data"])
        self.unlabeled_root = resolve_path(self.data_config["unlabeled_root"])
        self.model = resolve_path(self.data_config.get("model", "yolo11l-obb.pt"))
        self.final_test_split = str(self.data_config.get("final_test_split", "test"))

    def required_mapping(self, name: str) -> dict[str, Any]:
        value = self.config.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Configuration requires a mapping named '{name}'.")
        return value

    def subset_yaml(self, fraction_percent: int) -> Path:
        return self.subset_config_dir / f"poultry_clean_{fraction_percent}percent.yaml"

    def setting(self, candidate: dict[str, Any] | None, name: str, default: Any = None) -> Any:
        if candidate is not None and name in candidate:
            return candidate[name]
        return self.training.get(name, default)

    def validate_unlabeled_crop_sizes(self, pilot: dict[str, Any]) -> None:
        """Reject impossible unlabeled crops before an expensive pilot starts.

        The unlabeled loader samples a native-resolution crop; it does not
        upsample a smaller source tile. ``unlabeled_min_side`` is therefore a
        declared dataset contract for the selected unlabeled root.
        """
        minimum_side = self.data_config.get("unlabeled_min_side")
        if minimum_side is None:
            return
        try:
            minimum_side = int(minimum_side)
        except (TypeError, ValueError) as exc:
            raise ValueError("data.unlabeled_min_side must be a positive integer") from exc
        if minimum_side <= 0:
            raise ValueError("data.unlabeled_min_side must be a positive integer")

        candidates = pilot.get("candidates", [])
        if not isinstance(candidates, list):
            raise ValueError("pilot.candidates must be a list")
        settings: list[tuple[str, dict[str, Any] | None]] = [("training default", None)]
        settings.extend(
            (f"pilot candidate '{candidate.get('name', '<unnamed>')}'", candidate)
            for candidate in candidates
            if isinstance(candidate, dict)
        )
        for label, candidate in settings:
            try:
                size = int(self.setting(candidate, "unlabeled_imgsz"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} unlabeled_imgsz must be a positive integer") from exc
            if size <= 0 or size % 32:
                raise ValueError(f"{label} unlabeled_imgsz={size} must be a positive multiple of 32")
            if size > minimum_side:
                raise ValueError(
                    f"{label} unlabeled_imgsz={size} exceeds data.unlabeled_min_side={minimum_side}. "
                    "Choose a smaller crop or regenerate larger unlabeled tiles."
                )

    def prepare_metadata(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        config_bytes = self.config_path.read_bytes()
        config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        protocol_path = self.metadata_dir / "protocol.json"
        if protocol_path.is_file():
            previous = read_json(protocol_path)
            if previous.get("config_sha256") != config_sha256:
                raise RuntimeError(
                    "This output directory belongs to a different configuration. "
                    "Choose a new --output-dir instead of mixing studies."
                )
        else:
            protocol = {
                "created_utc": utc_now(),
                "study_name": self.config.get("study_name", self.output_dir.name),
                "config": str(self.config_path.resolve()),
                "config_sha256": config_sha256,
                "pipeline": "validation-only pilot -> fixed-configuration 10% confirmation -> nested label-efficiency curve",
                "checkpoint_selection": "best validation mAP50-95 only",
                "external_test_policy": "external test is used for leakage audit and final reporting only; never for candidate selection",
                "output_layout": {
                    "metadata": "00_metadata",
                    "pilot": "01_pilot_10pct",
                    "confirmation": "02_confirmation_10pct",
                    "label_efficiency": "03_label_efficiency",
                },
            }
            write_json(protocol_path, protocol)
            shutil.copy2(self.config_path, self.metadata_dir / "config_snapshot.yaml")
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
        }
        try:
            import torch

            environment.update({"torch": torch.__version__, "cuda_available": torch.cuda.is_available()})
            if torch.cuda.is_available():
                environment["cuda_device"] = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover - environment detail only
            environment["torch_error"] = str(exc)
        try:
            import ultralytics

            environment["ultralytics"] = ultralytics.__version__
        except Exception as exc:  # pragma: no cover - environment detail only
            environment["ultralytics_error"] = str(exc)
        write_json(self.metadata_dir / "environment.json", environment)

    def ensure_subsets(self, fractions: list[int]) -> None:
        missing = [fraction for fraction in fractions if not self.subset_yaml(fraction).is_file()]
        if not missing:
            return
        if not self.clean_data.is_file():
            raise FileNotFoundError(
                f"Missing subset YAMLs {missing} and clean source config is unavailable: {self.clean_data}"
            )
        command = [
            sys.executable,
            str(ROOT / "scripts" / "create_clean_label_subsets.py"),
            "--clean-data",
            str(self.clean_data),
            "--config-dir",
            str(self.subset_config_dir),
            "--output-dir",
            str(self.subset_output_dir),
            "--fractions",
            ",".join(str(fraction) for fraction in sorted(set(fractions))),
            "--overwrite",
        ]
        run_command(
            command,
            metadata_dir=self.metadata_dir,
            stage="preflight",
            fraction_percent=0,
            seed=-1,
            method="create_nested_subsets",
            run_dir=self.subset_config_dir,
        )

    def audit_fraction(self, fraction_percent: int) -> None:
        data = self.subset_yaml(fraction_percent)
        if not data.is_file():
            raise FileNotFoundError(data)
        sys.path.insert(0, str(ROOT))
        from scripts.run_ema_benchmark import audit_test_overlap

        audit = audit_test_overlap(data, self.final_test_data, final_test_split=self.final_test_split)
        if audit["training_image_count"] <= 0 or audit["final_test_image_count"] <= 0:
            raise RuntimeError(f"Empty train/validation or final-test split at {fraction_percent}%.")
        if audit["exact_sha256_matches"]:
            raise RuntimeError(
                f"Final-test leakage at {fraction_percent}%: {audit['exact_sha256_matches']} exact image matches."
            )
        audit.update({"fraction_percent": fraction_percent, "data": str(data.resolve())})
        write_json(self.metadata_dir / "dataset_overlap_audits" / f"fraction_{fraction_percent:03d}.json", audit)

    def preflight(self) -> list[int]:
        pilot = self.required_mapping("pilot")
        confirmation = self.required_mapping("confirmation")
        curve = self.required_mapping("label_efficiency")
        fractions = {
            int(pilot.get("fraction_percent", 10)),
            int(confirmation.get("fraction_percent", 10)),
            *[int(value) for value in curve.get("fractions_percent", [])],
        }
        if not fractions or any(value <= 0 or value > 100 for value in fractions):
            raise ValueError("All fractions_percent values must be in [1, 100].")
        self.validate_unlabeled_crop_sizes(pilot)
        for required in (self.final_test_data, self.unlabeled_root, self.model):
            if not required.exists():
                raise FileNotFoundError(required)
        self.ensure_subsets(sorted(fractions))
        for fraction in sorted(fractions):
            self.audit_fraction(fraction)
        return sorted(fractions)

    def train_supervised(
        self,
        *,
        stage: str,
        fraction_percent: int,
        seed: int,
        data: Path,
        run_dir: Path,
        init_model: Path,
        epochs: int,
        lr0: float,
        weight_decay: float,
        method: str,
    ) -> Path:
        completed = marker_checkpoint(run_dir, pseudo=False)
        if completed is not None:
            append_manifest(
                self.metadata_dir,
                stage=stage,
                fraction_percent=fraction_percent,
                seed=seed,
                method=method,
                status="reused_completed",
                run_dir=run_dir,
                checkpoint=completed,
            )
            return completed
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_supervised_obb.py"),
            "--data",
            str(data),
            "--init-model",
            str(init_model),
            "--output-dir",
            str(run_dir),
            "--epochs",
            str(epochs),
            "--batch",
            str(self.training["batch"]),
            "--imgsz",
            str(self.training["imgsz"]),
            "--workers",
            str(self.execution["workers"]),
            "--device",
            str(self.execution["device"]),
            "--seed",
            str(seed),
            "--fraction",
            "1.0",
            "--lr0",
            str(lr0),
            "--weight-decay",
            str(weight_decay),
        ]
        if last_checkpoint(run_dir) is not None:
            command.append("--resume")
        run_command(
            command,
            metadata_dir=self.metadata_dir,
            stage=stage,
            fraction_percent=fraction_percent,
            seed=seed,
            method=method,
            run_dir=run_dir,
        )
        completed = marker_checkpoint(run_dir, pseudo=False)
        if completed is None:
            raise FileNotFoundError(f"Supervised checkpoint missing after training: {run_dir}")
        return completed

    def calibrate(self, *, stage: str, fraction_percent: int, seed: int, data: Path, checkpoint: Path, output: Path) -> str:
        was_existing = output.is_file()
        if not was_existing:
            command = [
                sys.executable,
                str(ROOT / "scripts" / "calibrate_pseudo_thresholds.py"),
                "--checkpoint",
                str(checkpoint),
                "--data",
                str(data),
                "--split",
                "val",
                "--output",
                str(output),
                "--imgsz",
                str(self.training["imgsz"]),
                "--batch",
                str(self.training.get("calibration_batch", self.training["batch"])),
                "--workers",
                str(self.training.get("calibration_workers", self.execution["workers"])),
                "--device",
                str(self.execution["device"]),
                "--iou-match",
                str(self.calibration["iou_match"]),
                "--precision-target",
                str(self.calibration["precision_target"]),
                "--min-predictions",
                str(self.calibration["min_predictions"]),
                "--min-confidence",
                str(self.calibration["min_confidence"]),
                "--max-confidence",
                str(self.calibration["max_confidence"]),
                "--step",
                str(self.calibration["step"]),
                "--fallback",
                str(self.calibration["fallback"]),
            ]
            run_command(
                command,
                metadata_dir=self.metadata_dir,
                stage=stage,
                fraction_percent=fraction_percent,
                seed=seed,
                method="validation_threshold_calibration",
                run_dir=output.parent,
            )
        thresholds = load_thresholds(output)
        append_manifest(
            self.metadata_dir,
            stage=stage,
            fraction_percent=fraction_percent,
            seed=seed,
            method="validation_threshold_calibration",
            status="reused_completed" if was_existing else "completed",
            run_dir=output.parent,
            checkpoint=checkpoint,
            command=f"thresholds={thresholds}",
        )
        return thresholds

    def train_pseudo(
        self,
        *,
        stage: str,
        fraction_percent: int,
        seed: int,
        data: Path,
        run_dir: Path,
        baseline_checkpoint: Path,
        thresholds: str,
        candidate: dict[str, Any],
        method: str,
    ) -> Path:
        completed = marker_checkpoint(run_dir, pseudo=True)
        if completed is not None:
            append_manifest(
                self.metadata_dir,
                stage=stage,
                fraction_percent=fraction_percent,
                seed=seed,
                method=method,
                status="reused_completed",
                run_dir=run_dir,
                checkpoint=completed,
            )
            return completed
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_ema_pseudo_obb.py"),
            "--data",
            str(data),
            "--unlabeled-root",
            str(self.unlabeled_root),
            "--init-checkpoint",
            str(baseline_checkpoint),
            "--model",
            str(baseline_checkpoint),
            "--output-dir",
            str(run_dir),
            "--adapt-epochs",
            str(self.setting(candidate, "adapt_epochs")),
            "--finetune-epochs",
            str(self.setting(candidate, "finetune_epochs")),
            "--batch",
            str(self.setting(candidate, "batch")),
            "--unlabeled-batch",
            str(self.setting(candidate, "unlabeled_batch")),
            "--imgsz",
            str(self.setting(candidate, "imgsz")),
            "--unlabeled-imgsz",
            str(self.setting(candidate, "unlabeled_imgsz")),
            "--workers",
            str(self.execution["workers"]),
            "--unlabeled-workers",
            str(self.execution["unlabeled_workers"]),
            "--unlabeled-limit",
            str(self.setting(candidate, "unlabeled_limit")),
            "--prefetch-factor",
            str(self.setting(candidate, "prefetch_factor")),
            "--device",
            str(self.execution["device"]),
            "--seed",
            str(seed),
            "--fraction",
            "1.0",
            "--pseudo-obb-weight",
            str(self.setting(candidate, "pseudo_obb_weight")),
            "--pseudo-thresholds",
            thresholds,
            "--pseudo-iou-threshold",
            str(self.setting(candidate, "pseudo_iou_threshold")),
            "--pseudo-class-margin",
            str(self.setting(candidate, "pseudo_class_margin")),
            "--pseudo-max-det",
            str(self.setting(candidate, "pseudo_max_det")),
            "--pseudo-warmup-epochs",
            str(self.setting(candidate, "pseudo_warmup_epochs")),
            "--pseudo-rampup-epochs",
            str(self.setting(candidate, "pseudo_rampup_epochs")),
            "--feature-consistency-weight",
            str(self.setting(candidate, "feature_consistency_weight")),
            "--feature-rampup-epochs",
            str(self.setting(candidate, "feature_rampup_epochs")),
            "--foreground-feature-gain",
            str(self.setting(candidate, "foreground_feature_gain")),
            "--ema-decay",
            str(self.setting(candidate, "ema_decay")),
            "--adapt-lr0",
            str(self.setting(candidate, "adapt_lr0")),
            "--finetune-lr0",
            str(self.setting(candidate, "finetune_lr0")),
        ]
        if last_checkpoint(run_dir) is not None:
            command.append("--resume")
        run_command(
            command,
            metadata_dir=self.metadata_dir,
            stage=stage,
            fraction_percent=fraction_percent,
            seed=seed,
            method=method,
            run_dir=run_dir,
        )
        completed = marker_checkpoint(run_dir, pseudo=True)
        if completed is None:
            raise FileNotFoundError(f"Pseudo-OBB checkpoint missing after training: {run_dir}")
        return completed

    def evaluate(
        self,
        *,
        records: list[dict[str, object]],
        report_dir: Path,
        stage: str,
        fraction_percent: int,
        seed: int,
        method: str,
        checkpoint: Path,
        source: str,
        selected_method: str | None,
    ) -> None:
        pending_key = (stage, fraction_percent, method, seed)
        if any(test_row_key(row) == pending_key for row in records):
            print(f"[test] Reusing {stage} {fraction_percent}% seed {seed} {method}", flush=True)
            return
        from ultralytics import YOLO

        print(f"[test] {stage} {fraction_percent}% • seed {seed} • {method}: {checkpoint}", flush=True)
        append_manifest(
            self.metadata_dir,
            stage=stage,
            fraction_percent=fraction_percent,
            seed=seed,
            method=method,
            status="test_started",
            run_dir=report_dir / "final_test" / method / f"seed_{seed:03d}",
            checkpoint=checkpoint,
        )
        result = YOLO(str(checkpoint)).val(
            data=str(self.final_test_data),
            split=self.final_test_split,
            imgsz=int(self.training["imgsz"]),
            batch=int(self.training["test_batch"]),
            device=str(self.execution["device"]),
            workers=int(self.execution["workers"]),
            plots=self.plots,
            project=str(report_dir / "final_test" / method),
            name=f"seed_{seed:03d}",
            exist_ok=True,
        )
        values = getattr(result, "results_dict", {}) or {}
        row: dict[str, object] = {
            "stage": stage,
            "fraction_percent": fraction_percent,
            "method": method,
            "seed": seed,
            "checkpoint": str(checkpoint.resolve()),
            "test_data": str(self.final_test_data.resolve()),
            "source": source,
            **{metric: numeric(values.get(metric)) for metric in METRIC_FIELDS},
        }
        upsert_test_row(records, row)
        write_test_reports(records, output_dir=report_dir, selected_method=selected_method)
        append_manifest(
            self.metadata_dir,
            stage=stage,
            fraction_percent=fraction_percent,
            seed=seed,
            method=method,
            status="test_completed",
            run_dir=report_dir / "final_test" / method / f"seed_{seed:03d}",
            checkpoint=checkpoint,
        )

    def write_index(self) -> None:
        selection = self.output_dir / "01_pilot_10pct" / "selected_pseudo_configuration.json"
        selected = "Pending pilot selection."
        if selection.is_file():
            payload = read_json(selection)
            selected = (
                f"`{payload.get('selected_name')}` selected by validation mAP50-95 "
                f"({payload.get('selection_metric')})."
            )
        content = "\n".join(
            [
                f"# {self.config.get('study_name', self.output_dir.name)}",
                "",
                "This directory is a resumable, validation-gated SSL study.",
                "",
                "## Study stages",
                "",
                "1. `01_pilot_10pct`: validation-only ablations; no final-test metrics are produced.",
                "2. `02_confirmation_10pct`: selected SSL configuration versus supervised controls across seeds.",
                "3. `03_label_efficiency`: fixed SSL configuration versus Ultralytics baseline over nested fractions.",
                "",
                "## Provenance",
                "",
                "- `00_metadata/config_snapshot.yaml`: exact requested study configuration.",
                "- `00_metadata/protocol.json`: selection and test-isolation policy.",
                "- `00_metadata/environment.json`: runtime versions and GPU information.",
                "- `00_metadata/dataset_overlap_audits`: exact hash leakage checks for each fraction.",
                "- `00_metadata/run_manifest.csv`: every training/calibration/test command and resume event.",
                "",
                "## Current selection",
                "",
                selected,
                "",
                "External-test metrics are reporting-only. Do not select a model from these CSV files.",
                "",
            ]
        )
        (self.output_dir / "STUDY_INDEX.md").write_text(content, encoding="utf-8")


def base_candidate(name: str = "selected") -> dict[str, Any]:
    return {"name": name}


def pilot_rows_path(study: Study) -> Path:
    return study.output_dir / "01_pilot_10pct" / "pilot_validation_metrics.csv"


def run_pilot(study: Study) -> dict[str, Any]:
    pilot = study.required_mapping("pilot")
    fraction = int(pilot.get("fraction_percent", 10))
    seed = int(pilot.get("seed", 0))
    candidates = pilot.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("pilot.candidates must be a non-empty list")
    names = [str(candidate.get("name", "")) for candidate in candidates if isinstance(candidate, dict)]
    if len(names) != len(candidates) or not all(names) or len(set(names)) != len(names):
        raise ValueError("Every pilot candidate needs a unique non-empty name")

    stage = "pilot_validation_only"
    stage_dir = study.output_dir / "01_pilot_10pct"
    data = study.subset_yaml(fraction)
    shared_dir = stage_dir / "shared" / f"seed_{seed:03d}"
    baseline_dir = shared_dir / "ultralytics_burnin"
    baseline = study.train_supervised(
        stage=stage,
        fraction_percent=fraction,
        seed=seed,
        data=data,
        run_dir=baseline_dir,
        init_model=study.model,
        epochs=int(study.training["baseline_epochs"]),
        lr0=float(study.training["baseline_lr0"]),
        weight_decay=float(study.training["baseline_weight_decay"]),
        method="ultralytics_burnin",
    )
    thresholds = study.calibrate(
        stage=stage,
        fraction_percent=fraction,
        seed=seed,
        data=data,
        checkpoint=baseline,
        output=shared_dir / "calibration" / "thresholds.json",
    )
    rows: list[dict[str, object]] = []
    baseline_metrics = best_validation_metrics(baseline_dir)
    rows.append(
        {
            "method": "ultralytics_burnin",
            "candidate": "",
            "eligible_for_selection": False,
            "seed": seed,
            "fraction_percent": fraction,
            "checkpoint": str(baseline),
            **baseline_metrics,
        }
    )
    if bool(pilot.get("include_matched_control", True)):
        matched_dir = shared_dir / "ultralytics_matched"
        matched = study.train_supervised(
            stage=stage,
            fraction_percent=fraction,
            seed=seed,
            data=data,
            run_dir=matched_dir,
            init_model=baseline,
            epochs=int(study.training["matched_control_epochs"]),
            lr0=float(study.training["matched_lr0"]),
            weight_decay=float(study.training["matched_weight_decay"]),
            method="ultralytics_matched",
        )
        rows.append(
            {
                "method": "ultralytics_matched",
                "candidate": "",
                "eligible_for_selection": False,
                "seed": seed,
                "fraction_percent": fraction,
                "checkpoint": str(matched),
                **best_validation_metrics(matched_dir),
            }
        )
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise ValueError("Each pilot candidate must be a mapping")
        name = str(candidate["name"])
        candidate_dir = stage_dir / "candidates" / name / f"seed_{seed:03d}"
        write_json(
            candidate_dir / "candidate_parameters.json",
            {
                "name": name,
                "pilot_order": index,
                "candidate_overrides": candidate,
                "effective_training": {key: study.setting(candidate, key) for key in study.training},
                "thresholds": thresholds,
                "selection_uses_final_test": False,
            },
        )
        pseudo_dir = candidate_dir / "pseudo_obb"
        checkpoint = study.train_pseudo(
            stage=stage,
            fraction_percent=fraction,
            seed=seed,
            data=data,
            run_dir=pseudo_dir,
            baseline_checkpoint=baseline,
            thresholds=thresholds,
            candidate=candidate,
            method=f"pseudo_{name}",
        )
        metrics = best_validation_metrics(pseudo_dir.with_name(pseudo_dir.name + "_finetune"))
        rows.append(
            {
                "method": f"pseudo_{name}",
                "candidate": name,
                "eligible_for_selection": bool(candidate.get("eligible_for_selection", True)),
                "seed": seed,
                "fraction_percent": fraction,
                "checkpoint": str(checkpoint),
                **metrics,
            }
        )
        fields = [
            "method",
            "candidate",
            "eligible_for_selection",
            "seed",
            "fraction_percent",
            "checkpoint",
            "best_epoch",
            *METRIC_FIELDS,
        ]
        write_rows(pilot_rows_path(study), rows, fields)
        print(f"[pilot {index}/{len(candidates)}] {name} validation mAP50-95={metrics['metrics/mAP50-95(B)']:.5f}", flush=True)

    selected_row = select_candidate(rows)
    selected_name = str(selected_row["candidate"])
    selected_candidate = next(candidate for candidate in candidates if str(candidate["name"]) == selected_name)
    selection = {
        "created_utc": utc_now(),
        "selected_name": selected_name,
        "selected_method": str(selected_row["method"]),
        "selection_metric": "validation metrics/mAP50-95(B)",
        "selection_value": numeric(selected_row.get("metrics/mAP50-95(B)")),
        "pilot_fraction_percent": fraction,
        "pilot_seed": seed,
        "candidate_overrides": selected_candidate,
        "candidate_effective_training": {key: study.setting(selected_candidate, key) for key in study.training},
        "final_test_used_for_selection": False,
    }
    write_json(stage_dir / "selected_pseudo_configuration.json", selection)
    study.write_index()
    print(f"\nSelected validation-only pseudo configuration: {selected_name}", flush=True)
    return selection


def load_selection(study: Study) -> dict[str, Any]:
    path = study.output_dir / "01_pilot_10pct" / "selected_pseudo_configuration.json"
    if not path.is_file():
        raise FileNotFoundError(f"Pilot selection not found: {path}. Run --stage pilot first.")
    selection = read_json(path)
    if selection.get("final_test_used_for_selection") is not False:
        raise RuntimeError("Selection provenance does not prove test isolation.")
    return selection


def confirmation_records_path(study: Study) -> Path:
    return study.output_dir / "02_confirmation_10pct" / "per_seed_external_test_metrics.csv"


def run_confirmation(study: Study) -> None:
    selection = load_selection(study)
    confirmation = study.required_mapping("confirmation")
    fraction = int(confirmation.get("fraction_percent", 10))
    seeds = [int(value) for value in confirmation.get("seeds", [])]
    if not seeds:
        raise ValueError("confirmation.seeds must be non-empty")
    candidate = selection.get("candidate_overrides")
    if not isinstance(candidate, dict):
        raise ValueError("Selected candidate configuration is invalid")
    candidate_name = str(selection["selected_name"])
    pseudo_method = f"pseudo_{candidate_name}"
    stage = "confirmation_10pct"
    stage_dir = study.output_dir / "02_confirmation_10pct"
    data = study.subset_yaml(fraction)
    records: list[dict[str, object]] = [dict(row) for row in load_rows(confirmation_records_path(study))]

    for index, seed in enumerate(seeds, start=1):
        print(f"\n[confirmation {index}/{len(seeds)}] {fraction}% seed {seed}", flush=True)
        seed_dir = stage_dir / f"seed_{seed:03d}"
        baseline_dir = seed_dir / "ultralytics_burnin"
        baseline = study.train_supervised(
            stage=stage,
            fraction_percent=fraction,
            seed=seed,
            data=data,
            run_dir=baseline_dir,
            init_model=study.model,
            epochs=int(study.training["baseline_epochs"]),
            lr0=float(study.training["baseline_lr0"]),
            weight_decay=float(study.training["baseline_weight_decay"]),
            method="ultralytics_burnin",
        )
        thresholds = study.calibrate(
            stage=stage,
            fraction_percent=fraction,
            seed=seed,
            data=data,
            checkpoint=baseline,
            output=seed_dir / "calibration" / "thresholds.json",
        )
        matched: Path | None = None
        if bool(confirmation.get("include_matched_control", True)):
            matched = study.train_supervised(
                stage=stage,
                fraction_percent=fraction,
                seed=seed,
                data=data,
                run_dir=seed_dir / "ultralytics_matched",
                init_model=baseline,
                epochs=int(study.training["matched_control_epochs"]),
                lr0=float(study.training["matched_lr0"]),
                weight_decay=float(study.training["matched_weight_decay"]),
                method="ultralytics_matched",
            )
        pseudo = study.train_pseudo(
            stage=stage,
            fraction_percent=fraction,
            seed=seed,
            data=data,
            run_dir=seed_dir / "pseudo_obb",
            baseline_checkpoint=baseline,
            thresholds=thresholds,
            candidate=candidate,
            method=pseudo_method,
        )
        study.evaluate(
            records=records,
            report_dir=stage_dir,
            stage=stage,
            fraction_percent=fraction,
            seed=seed,
            method="ultralytics_burnin",
            checkpoint=baseline,
            source="confirmation_external_test",
            selected_method=pseudo_method,
        )
        if matched is not None:
            study.evaluate(
                records=records,
                report_dir=stage_dir,
                stage=stage,
                fraction_percent=fraction,
                seed=seed,
                method="ultralytics_matched",
                checkpoint=matched,
                source="confirmation_external_test",
                selected_method=pseudo_method,
            )
        study.evaluate(
            records=records,
            report_dir=stage_dir,
            stage=stage,
            fraction_percent=fraction,
            seed=seed,
            method=pseudo_method,
            checkpoint=pseudo,
            source="confirmation_external_test",
            selected_method=pseudo_method,
        )
    write_test_reports(records, output_dir=stage_dir, selected_method=pseudo_method)
    study.write_index()


def curve_records_path(study: Study) -> Path:
    return study.output_dir / "03_label_efficiency" / "curve_per_seed_external_test_metrics.csv"


def records_for_reuse(
    rows: list[dict[str, str]],
    *,
    fraction_percent: int,
    methods: set[str],
    seeds: set[int],
) -> list[dict[str, object]]:
    reused: list[dict[str, object]] = []
    for row in rows:
        if int(float(row.get("fraction_percent", 0))) != fraction_percent:
            continue
        if str(row.get("method")) not in methods:
            continue
        if int(float(row.get("seed", 0))) not in seeds:
            continue
        copied: dict[str, object] = dict(row)
        copied["stage"] = "label_efficiency"
        copied["source"] = "reused_confirmation_10pct"
        reused.append(copied)
    if {(str(row["method"]), int(float(row["seed"]))) for row in reused} != {
        (method, seed) for method in methods for seed in seeds
    }:
        raise RuntimeError("Confirmation results cannot be reused because they do not cover every curve seed/method.")
    return reused


def has_complete_test_matrix(
    rows: list[dict[str, object]], *, methods: set[str], seeds: set[int]
) -> bool:
    """Return true when every requested method/seed test row is present."""
    present = {
        (str(row.get("method", "")), int(float(row.get("seed", -1))))
        for row in rows
    }
    return {(method, seed) for method in methods for seed in seeds}.issubset(present)


def run_label_efficiency(study: Study) -> None:
    selection = load_selection(study)
    curve = study.required_mapping("label_efficiency")
    confirmation = study.required_mapping("confirmation")
    fractions = [int(value) for value in curve.get("fractions_percent", [])]
    seeds = [int(value) for value in curve.get("seeds", [])]
    if not fractions or not seeds:
        raise ValueError("label_efficiency requires non-empty fractions_percent and seeds")
    candidate = selection.get("candidate_overrides")
    if not isinstance(candidate, dict):
        raise ValueError("Selected candidate configuration is invalid")
    candidate_name = str(selection["selected_name"])
    pseudo_method = f"pseudo_{candidate_name}"
    stage = "label_efficiency"
    stage_dir = study.output_dir / "03_label_efficiency"
    records: list[dict[str, object]] = [dict(row) for row in load_rows(curve_records_path(study))]
    reused_confirmation = bool(curve.get("reuse_confirmation_for_10pct", True))
    confirmation_fraction = int(confirmation.get("fraction_percent", 10))
    confirmation_seeds = {int(value) for value in confirmation.get("seeds", [])}

    for fraction_index, fraction in enumerate(fractions, start=1):
        fraction_dir = stage_dir / f"fraction_{fraction:03d}pct"
        print(f"\n[curve {fraction_index}/{len(fractions)}] {fraction}% | seeds {seeds}", flush=True)

        # Recover rows written inside a fraction before an interrupted run had
        # a chance to update the curve-wide CSV. This prevents repeated final
        # testing and makes seed-level progress genuinely resumable.
        for stored in load_rows(fraction_dir / "per_seed_external_test_metrics.csv"):
            if (
                str(stored.get("stage")) == stage
                and int(float(stored.get("fraction_percent", 0))) == fraction
            ):
                upsert_test_row(records, dict(stored))
        fraction_records = [
            row
            for row in records
            if str(row.get("stage")) == stage
            and int(float(row.get("fraction_percent", 0))) == fraction
        ]
        expected_methods = {"ultralytics_burnin", pseudo_method}
        if has_complete_test_matrix(fraction_records, methods=expected_methods, seeds=set(seeds)):
            print(f"[curve] Reusing complete {fraction}% test matrix; no training required.", flush=True)
            write_test_reports(fraction_records, output_dir=fraction_dir, selected_method=pseudo_method)
            write_test_reports(records, output_dir=stage_dir, selected_method=pseudo_method)
            write_rows(curve_records_path(study), sorted(records, key=test_row_key), test_result_fields())
            continue

        if (
            reused_confirmation
            and fraction == confirmation_fraction
            and set(seeds) == confirmation_seeds
            and not fraction_records
        ):
            reused = records_for_reuse(
                load_rows(confirmation_records_path(study)),
                fraction_percent=fraction,
                methods={"ultralytics_burnin", pseudo_method},
                seeds=set(seeds),
            )
            records.extend(reused)
            write_json(
                fraction_dir / "reused_from_confirmation.json",
                {
                    "source": str(confirmation_records_path(study).resolve()),
                    "fraction_percent": fraction,
                    "methods": ["ultralytics_burnin", pseudo_method],
                    "seeds": seeds,
                    "selection_was_validation_only": True,
                },
            )
            write_test_reports(records, output_dir=stage_dir, selected_method=pseudo_method)
            write_rows(curve_records_path(study), sorted(records, key=test_row_key), test_result_fields())
            continue

        data = study.subset_yaml(fraction)
        for seed in seeds:
            seed_dir = fraction_dir / f"seed_{seed:03d}"
            baseline = study.train_supervised(
                stage=stage,
                fraction_percent=fraction,
                seed=seed,
                data=data,
                run_dir=seed_dir / "ultralytics_burnin",
                init_model=study.model,
                epochs=int(study.training["baseline_epochs"]),
                lr0=float(study.training["baseline_lr0"]),
                weight_decay=float(study.training["baseline_weight_decay"]),
                method="ultralytics_burnin",
            )
            thresholds = study.calibrate(
                stage=stage,
                fraction_percent=fraction,
                seed=seed,
                data=data,
                checkpoint=baseline,
                output=seed_dir / "calibration" / "thresholds.json",
            )
            pseudo = study.train_pseudo(
                stage=stage,
                fraction_percent=fraction,
                seed=seed,
                data=data,
                run_dir=seed_dir / "pseudo_obb",
                baseline_checkpoint=baseline,
                thresholds=thresholds,
                candidate=candidate,
                method=pseudo_method,
            )
            study.evaluate(
                records=fraction_records,
                report_dir=fraction_dir,
                stage=stage,
                fraction_percent=fraction,
                seed=seed,
                method="ultralytics_burnin",
                checkpoint=baseline,
                source="label_efficiency_external_test",
                selected_method=pseudo_method,
            )
            study.evaluate(
                records=fraction_records,
                report_dir=fraction_dir,
                stage=stage,
                fraction_percent=fraction,
                seed=seed,
                method=pseudo_method,
                checkpoint=pseudo,
                source="label_efficiency_external_test",
                selected_method=pseudo_method,
            )
        records = [
            row
            for row in records
            if not (str(row.get("stage")) == stage and int(float(row.get("fraction_percent", 0))) == fraction)
        ]
        records.extend(fraction_records)
        write_test_reports(fraction_records, output_dir=fraction_dir, selected_method=pseudo_method)
        write_test_reports(records, output_dir=stage_dir, selected_method=pseudo_method)
        write_rows(curve_records_path(study), sorted(records, key=test_row_key), test_result_fields())

    fields = test_result_fields()
    write_rows(curve_records_path(study), sorted(records, key=test_row_key), fields)
    write_test_reports(records, output_dir=stage_dir, selected_method=pseudo_method)
    study.write_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ssl_publication_study_clean.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stage", choices=("all", "pilot", "confirmation", "curve"), default="all")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--plots", action="store_true", help="Save Ultralytics plots during external-test evaluation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Study configuration must be a YAML mapping")
    output_dir = resolve_path(args.output_dir or config.get("output_dir", "runs/poultry_ssl_pseudo_obb_publication_v1"))
    study = Study(config_path, config, output_dir, plots=args.plots or bool(config.get("execution", {}).get("plots_on_final_test", False)))
    study.prepare_metadata()
    fractions = study.preflight()
    study.write_index()
    print(f"Preflight passed for fractions {fractions}. Study output: {output_dir}", flush=True)
    if args.preflight_only:
        return 0

    if args.stage in ("all", "pilot"):
        run_pilot(study)
    if args.stage in ("all", "confirmation"):
        run_confirmation(study)
    if args.stage in ("all", "curve"):
        run_label_efficiency(study)
    study.write_index()
    print(f"\nSSL publication study complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
