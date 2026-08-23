"""Run supervised burn-in, validation calibration, and matched pseudo-OBB tests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(f'"{part}"' if " " in part else part for part in command), flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ema_pseudo_obb_burnin_clean_10pct.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--allow-test-overlap", action="store_true", help="Allow exact train/final-test image matches for exploratory runs only.")
    parser.add_argument("--skip-matched-control", action="store_true", help="Skip the equal-update supervised continuation control.")
    return parser.parse_args()


def checkpoint_for(run_dir: Path, *, pseudo: bool = False) -> Path | None:
    if pseudo:
        marker = run_dir.with_name(run_dir.name + "_finetune") / "pseudo_obb_handoff.json"
        candidate = run_dir.with_name(run_dir.name + "_finetune") / "weights" / "best.pt"
    else:
        marker = run_dir / "supervised_baseline.json"
        candidate = run_dir / "weights" / "best.pt"
    return candidate if candidate.is_file() and marker.is_file() else None


def load_thresholds(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError(f"Calibration file has no thresholds: {path}")
    return ",".join(f"{float(value):.4f}" for value in thresholds)


def mean_std(values: list[float]) -> tuple[float, float]:
    return (mean(values), stdev(values) if len(values) > 1 else 0.0) if values else (float("nan"), float("nan"))


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data = resolve_path(config["data"])
    final_test_data = resolve_path(config["final_test_data"])
    unlabeled_root = resolve_path(config["unlabeled_root"])
    output_dir = resolve_path(args.output_dir or config["output_dir"])
    seeds = list(args.seeds if args.seeds is not None else config.get("seeds", [0, 7, 42]))
    if not seeds:
        raise ValueError("At least one seed is required")
    for required in (config_path, data, final_test_data, unlabeled_root):
        if not required.exists():
            raise FileNotFoundError(required)

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    from scripts.run_ema_benchmark import audit_test_overlap

    audit = audit_test_overlap(data, final_test_data, final_test_split=config.get("final_test_split", "test"))
    if audit["training_image_count"] <= 0 or audit["final_test_image_count"] <= 0:
        raise RuntimeError("Preflight resolved an empty training or external test split.")

    if audit["exact_sha256_matches"] and not args.allow_test_overlap:
        raise RuntimeError(
            f"Final-test leakage detected: {audit['exact_sha256_matches']} exact image matches. "
            "Use --allow-test-overlap only for exploratory runs."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_overlap_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    calibration_config = config.get("calibration", {})
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "supervised burn-in -> validation threshold calibration -> detector-level EMA pseudo-OBB",
        "config": str(config_path.resolve()),
        "data": str(data.resolve()),
        "final_test_data": str(final_test_data.resolve()),
        "unlabeled_root": str(unlabeled_root.resolve()),
        "seeds": seeds,
        "baseline_epochs": int(config.get("baseline_epochs", 100)),
        "pseudo_adapt_epochs": int(config.get("adapt_epochs", 60)),
        "pseudo_finetune_epochs": int(config.get("finetune_epochs", 100)),
        "matched_supervised_control": not args.skip_matched_control and bool(config.get("matched_supervised_control", True)),
        "matched_control_epochs": int(config.get("adapt_epochs", 60)) + int(config.get("finetune_epochs", 100)),
        "supervised_burn_in_checkpoint": "baseline_seed*/weights/best.pt",
        "threshold_calibration": calibration_config,
        "calibration_split": "val",
        "final_test_used_for_selection": False,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if args.preflight_only:
        print(f"Preflight complete: {output_dir}", flush=True)
        return 0

    supervised_script = ROOT / "scripts" / "train_supervised_obb.py"
    calibrate_script = ROOT / "scripts" / "calibrate_pseudo_thresholds.py"
    pseudo_script = ROOT / "scripts" / "train_ema_pseudo_obb.py"
    records: list[dict[str, object]] = []
    for index, seed in enumerate(seeds, start=1):
        print(f"\n[{index}/{len(seeds)}] Seed {seed}: supervised burn-in", flush=True)
        baseline_dir = output_dir / f"baseline_seed{seed}"
        baseline_checkpoint = checkpoint_for(baseline_dir)
        if baseline_checkpoint is None or not args.reuse_existing:
            run_command(
                [
                    sys.executable,
                    str(supervised_script),
                    "--data", str(data),
                    "--init-model", str(config.get("model", "yolo11l-obb.pt")),
                    "--output-dir", str(baseline_dir),
                    "--epochs", str(config.get("baseline_epochs", 100)),
                    "--batch", str(config.get("batch", 16)),
                    "--imgsz", str(config.get("imgsz", 640)),
                    "--workers", str(config.get("workers", 12)),
                    "--device", str(config.get("device", "0")),
                    "--seed", str(seed),
                    "--fraction", str(config.get("fraction", 1.0)),
                    "--lr0", str(config.get("baseline_lr0", 1e-3)),
                    "--weight-decay", str(config.get("baseline_weight_decay", 5e-4)),
                ]
            )
            baseline_checkpoint = checkpoint_for(baseline_dir)
        if baseline_checkpoint is None:
            raise FileNotFoundError(f"Supervised burn-in checkpoint missing for seed {seed}: {baseline_dir}")

        calibration_path = output_dir / "calibration" / f"seed{seed}.json"
        if not calibration_path.is_file() or not args.reuse_existing:
            run_command(
                [
                    sys.executable,
                    str(calibrate_script),
                    "--checkpoint", str(baseline_checkpoint),
                    "--data", str(data),
                    "--split", "val",
                    "--output", str(calibration_path),
                    "--imgsz", str(config.get("imgsz", 640)),
                    "--batch", str(config.get("calibration_batch", config.get("batch", 16))),
                    "--workers", str(config.get("calibration_workers", config.get("workers", 12))),
                    "--device", str(config.get("device", "0")),
                    "--iou-match", str(calibration_config.get("iou_match", 0.50)),
                    "--precision-target", str(calibration_config.get("precision_target", 0.90)),
                    "--min-predictions", str(calibration_config.get("min_predictions", 10)),
                    "--min-confidence", str(calibration_config.get("min_confidence", 0.50)),
                    "--max-confidence", str(calibration_config.get("max_confidence", 0.99)),
                    "--step", str(calibration_config.get("step", 0.01)),
                    "--fallback", str(calibration_config.get("fallback", 0.90)),
                ]
            )
        thresholds = load_thresholds(calibration_path)
        print(f"Seed {seed} calibrated thresholds: {thresholds}", flush=True)

        matched_dir = output_dir / f"matched_seed{seed}"
        matched_checkpoint = None
        if not args.skip_matched_control and config.get("matched_supervised_control", True):
            matched_checkpoint = checkpoint_for(matched_dir)
            if matched_checkpoint is None or not args.reuse_existing:
                run_command([
                    sys.executable,
                    str(supervised_script),
                    "--data", str(data),
                    "--init-model", str(baseline_checkpoint),
                    "--output-dir", str(matched_dir),
                    "--epochs", str(int(config.get("adapt_epochs", 60)) + int(config.get("finetune_epochs", 100))),
                    "--batch", str(config.get("batch", 16)),
                    "--imgsz", str(config.get("imgsz", 640)),
                    "--workers", str(config.get("workers", 12)),
                    "--device", str(config.get("device", "0")),
                    "--seed", str(seed),
                    "--fraction", str(config.get("fraction", 1.0)),
                    "--lr0", str(config.get("finetune_lr0", 3e-4)),
                    "--weight-decay", str(config.get("matched_weight_decay", 5e-4)),
                ])
                matched_checkpoint = checkpoint_for(matched_dir)
            if matched_checkpoint is None:
                raise FileNotFoundError(f"Matched supervised checkpoint missing for seed {seed}: {matched_dir}")

        pseudo_dir = output_dir / f"pseudo_seed{seed}"
        pseudo_checkpoint = checkpoint_for(pseudo_dir, pseudo=True)
        if pseudo_checkpoint is None or not args.reuse_existing:
            run_command(
                [
                    sys.executable,
                    str(pseudo_script),
                    "--data", str(data),
                    "--unlabeled-root", str(unlabeled_root),
                    "--init-checkpoint", str(baseline_checkpoint),
                    "--model", str(baseline_checkpoint),
                    "--output-dir", str(pseudo_dir),
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
                    "--pseudo-thresholds", thresholds,
                    "--pseudo-iou-threshold", str(config.get("pseudo_iou_threshold", 0.5)),
                    "--pseudo-class-margin", str(config.get("pseudo_class_margin", 0.05)),
                    "--pseudo-max-det", str(config.get("pseudo_max_det", 100)),
                    "--pseudo-warmup-epochs", str(config.get("pseudo_warmup_epochs", 5)),
                    "--pseudo-rampup-epochs", str(config.get("pseudo_rampup_epochs", 10)),
                    "--feature-consistency-weight", str(config.get("feature_consistency_weight", 0.0)),
                    "--feature-rampup-epochs", str(config.get("feature_rampup_epochs", 10)),
                    "--foreground-feature-gain", str(config.get("foreground_feature_gain", 3.0)),
                    "--ema-decay", str(config.get("ema_decay", 0.9998)),
                    "--adapt-lr0", str(config.get("adapt_lr0", 1e-5)),
                    "--finetune-lr0", str(config.get("finetune_lr0", 3e-4)),
                ]
            )
            pseudo_checkpoint = checkpoint_for(pseudo_dir, pseudo=True)
        if pseudo_checkpoint is None:
            raise FileNotFoundError(f"Pseudo-OBB checkpoint missing for seed {seed}: {pseudo_dir}")

        if args.skip_test:
            continue
        from ultralytics import YOLO

        evaluation_checkpoints = [("ultralytics_burnin", baseline_checkpoint)]
        if matched_checkpoint is not None:
            evaluation_checkpoints.append(("ultralytics_matched", matched_checkpoint))
        evaluation_checkpoints.append(("pseudo_obb", pseudo_checkpoint))
        for method, checkpoint in evaluation_checkpoints:
            print(f"[test] Seed {seed} • {method}: {checkpoint}", flush=True)
            result = YOLO(str(checkpoint)).val(
                data=str(final_test_data),
                split=config.get("final_test_split", "test"),
                imgsz=int(config.get("imgsz", 640)),
                batch=int(config.get("test_batch", 32)),
                device=str(config.get("device", "0")),
                workers=int(config.get("workers", 12)),
                plots=not args.no_plots,
                project=str(output_dir / "final_test" / method),
                name=f"seed{seed}",
                exist_ok=True,
            )
            values = getattr(result, "results_dict", {})
            records.append(
                {
                    "method": method,
                    "seed": seed,
                    "checkpoint": str(checkpoint.resolve()),
                    "test_data": str(final_test_data.resolve()),
                    **{key: float(values.get(key, float("nan"))) for key in (
                        "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"
                    )},
                }
            )

    if args.skip_test:
        print("Training and calibration complete; final test skipped.", flush=True)
        return 0
    fields = ["method", "seed", "checkpoint", "test_data", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    with (output_dir / "per_seed_test_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    aggregate_rows: list[dict[str, object]] = []
    for method in ("ultralytics_burnin", "ultralytics_matched", "pseudo_obb"):
        method_records = [row for row in records if row["method"] == method]
        row: dict[str, object] = {"method": method, "n_seeds": len(method_records)}
        for field in fields[4:]:
            values = [float(item[field]) for item in method_records if not math.isnan(float(item[field]))]
            average, standard_deviation = mean_std(values)
            row[f"{field}/mean"] = average
            row[f"{field}/std"] = standard_deviation
        aggregate_rows.append(row)
    aggregate_fields = list(aggregate_rows[0]) if aggregate_rows else ["method", "n_seeds"]
    with (output_dir / "aggregate_test_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    pseudo_by_seed = {int(row["seed"]): row for row in records if row["method"] == "pseudo_obb"}
    delta_fields = ["seed", *fields[4:]]
    for baseline_method, filename in (
        ("ultralytics_burnin", "paired_pseudo_minus_burnin.csv"),
        ("ultralytics_matched", "paired_pseudo_minus_matched.csv"),
    ):
        baseline_by_seed = {int(row["seed"]): row for row in records if row["method"] == baseline_method}
        if len(baseline_by_seed) != len(seeds):
            continue
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=delta_fields)
            writer.writeheader()
            for seed in seeds:
                pseudo = pseudo_by_seed[seed]
                baseline = baseline_by_seed[seed]
                writer.writerow({"seed": seed, **{field: float(pseudo[field]) - float(baseline[field]) for field in fields[4:]}})
    print(f"\nBurn-in pseudo-OBB benchmark complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
