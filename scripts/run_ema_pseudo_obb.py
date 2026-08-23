"""Run the separate detector-level EMA pseudo-OBB experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(f'\"{part}\"' if " " in part else part for part in command), flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def find_checkpoint(run_dir: Path) -> Path | None:
    stage2 = run_dir.with_name(run_dir.name + "_finetune")
    checkpoint = stage2 / "weights" / "best.pt"
    marker = stage2 / "pseudo_obb_handoff.json"
    return checkpoint if checkpoint.is_file() and marker.is_file() else None


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    average = sum(values) / len(values)
    standard_deviation = (
        sum((value - average) ** 2 for value in values) / (len(values) - 1)
    ) ** 0.5 if len(values) > 1 else 0.0
    return average, standard_deviation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ema_pseudo_obb_clean_10pct.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--unlabeled-root", default=None, help="Override the config root; defaults to the restricted MS_tiles_512_256_jpg tree.")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Validate paths and write the protocol without training or testing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data = resolve_path(config["data"])
    final_test_data = resolve_path(config["final_test_data"])
    unlabeled_root = resolve_path(args.unlabeled_root or config["unlabeled_root"])
    output_dir = resolve_path(args.output_dir or config["output_dir"])
    seeds = list(args.seeds if args.seeds is not None else config.get("seeds", [0, 7, 42]))
    if not seeds:
        raise ValueError("At least one seed is required")
    for required in (data, final_test_data, unlabeled_root):
        if not required.exists():
            raise FileNotFoundError(required)

    # Import after path validation so --help/preflight remains lightweight.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    from scripts.run_ema_benchmark import audit_test_overlap

    audit = audit_test_overlap(data, final_test_data, final_test_split=config.get("final_test_split", "test"))
    if audit["training_image_count"] <= 0 or audit["final_test_image_count"] <= 0:
        raise RuntimeError("Preflight resolved an empty training or external test split.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_overlap_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "detector-level EMA pseudo-OBB adaptation",
        "config": str(config_path.resolve()),
        "data": str(data.resolve()),
        "final_test_data": str(final_test_data.resolve()),
        "unlabeled_root": str(unlabeled_root.resolve()),
        "seeds": seeds,
        "checkpoint_selection": "stage-2 best validation checkpoint only",
        "final_test_used_for_selection": False,
        "pseudo_obb_weight": config.get("pseudo_obb_weight", 0.05),
        "pseudo_thresholds": config.get("pseudo_thresholds", "0.90"),
        "pseudo_class_margin": config.get("pseudo_class_margin", 0.05),
        "feature_consistency_weight": config.get("feature_consistency_weight", 0.0),
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if args.preflight_only:
        print(f"Preflight complete: {output_dir}", flush=True)
        return 0

    fields = [
        "seed",
        "checkpoint",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    rows: list[dict[str, object]] = []
    train_script = ROOT / "scripts" / "train_ema_pseudo_obb.py"
    for seed in seeds:
        run_dir = output_dir / f"seed{seed}"
        checkpoint = find_checkpoint(run_dir)
        if checkpoint is None or not args.reuse_existing:
            command = [
                sys.executable,
                str(train_script),
                "--data", str(data),
                "--unlabeled-root", str(unlabeled_root),
                "--model", str(config.get("model", "yolo11l-obb.pt")),
                "--output-dir", str(run_dir),
                "--adapt-epochs", str(config.get("adapt_epochs", 60)),
                "--finetune-epochs", str(config.get("finetune_epochs", 100)),
                "--batch", str(config.get("batch", 16)),
                "--unlabeled-batch", str(config.get("unlabeled_batch", 16)),
                "--imgsz", str(config.get("imgsz", 640)),
                "--unlabeled-imgsz", str(config.get("unlabeled_imgsz", 448)),
                "--workers", str(config.get("workers", 12)),
                "--unlabeled-workers", str(config.get("unlabeled_workers", 2)),
                "--unlabeled-limit", str(config.get("unlabeled_limit", 250000)),
                "--prefetch-factor", str(config.get("prefetch_factor", 2)),
                "--device", str(config.get("device", "0")),
                "--seed", str(seed),
                "--fraction", str(config.get("fraction", 1.0)),
                "--pseudo-obb-weight", str(config.get("pseudo_obb_weight", 0.05)),
                "--pseudo-thresholds", str(config.get("pseudo_thresholds", "0.90")),
                "--pseudo-iou-threshold", str(config.get("pseudo_iou_threshold", 0.5)),
                "--pseudo-class-margin", str(config.get("pseudo_class_margin", 0.05)),
                "--pseudo-max-det", str(config.get("pseudo_max_det", 100)),
                "--pseudo-warmup-epochs", str(config.get("pseudo_warmup_epochs", 5)),
                "--pseudo-rampup-epochs", str(config.get("pseudo_rampup_epochs", 10)),
                "--feature-consistency-weight", str(config.get("feature_consistency_weight", 0.0)),
                "--feature-rampup-epochs", str(config.get("feature_rampup_epochs", 10)),
                "--foreground-feature-gain", str(config.get("foreground_feature_gain", 3.0)),
                "--ema-decay", str(config.get("ema_decay", 0.9998)),
            ]
            run_command(command)
            checkpoint = find_checkpoint(run_dir)
        if checkpoint is None:
            raise FileNotFoundError(f"Verified pseudo-OBB checkpoint missing for seed {seed}: {run_dir}")

    if args.skip_test:
        print("Training complete; final test was skipped by request.", flush=True)
        return 0

    from ultralytics import YOLO

    eval_root = output_dir / "final_test"
    for seed in seeds:
        checkpoint = find_checkpoint(output_dir / f"seed{seed}")
        assert checkpoint is not None
        print(f"\n[test] pseudo-OBB seed={seed}: {checkpoint}", flush=True)
        result = YOLO(str(checkpoint)).val(
            data=str(final_test_data),
            split=config.get("final_test_split", "test"),
            imgsz=int(config.get("imgsz", 640)),
            batch=int(config.get("test_batch", 32)),
            device=str(config.get("device", "0")),
            workers=int(config.get("workers", 12)),
            plots=True,
            project=str(eval_root),
            name=f"seed{seed}",
            exist_ok=True,
        )
        values = getattr(result, "results_dict", {})
        row: dict[str, object] = {"seed": seed, "checkpoint": str(checkpoint.resolve())}
        for field in fields[2:]:
            value = values.get(field)
            row[field] = float(value) if value is not None else float("nan")
        rows.append(row)

    with (output_dir / "per_seed_test_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregate: dict[str, object] = {"method": "ema_pseudo_obb", "n_seeds": len(rows)}
    for field in fields[2:]:
        values = [float(row[field]) for row in rows if not math.isnan(float(row[field]))]
        average, standard_deviation = mean_std(values)
        aggregate[f"{field}/mean"] = average
        aggregate[f"{field}/std"] = standard_deviation
        aggregate[f"{field}/min"] = min(values) if values else float("nan")
        aggregate[f"{field}/max"] = max(values) if values else float("nan")
    aggregate_fields = list(aggregate)
    with (output_dir / "aggregate_test_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerow(aggregate)
    print(f"\nPseudo-OBB experiment complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
