"""Run a leakage-safe, validation-only CTQ reliability-gate comparison.

For each seed this runner trains a supervised baseline, calibrates classwise
pseudo thresholds only on validation, then compares a matched continuation
control with the reliability-gated CTQ candidate.  It has no test-set option.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
METRIC = "metrics/mAP50-95(B)"


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"protocol_name", "data", "unlabeled_root", "unlabeled_catalog", "model", "output_dir", "device", "seeds", "baseline", "calibration", "adaptation", "candidate"}
    if not isinstance(config, dict) or required - config.keys():
        missing = sorted(required - (config.keys() if isinstance(config, dict) else set()))
        raise ValueError(f"Invalid reliability-screen config; missing {missing}")
    seeds = [int(seed) for seed in config["seeds"]]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if float(config["candidate"]["pseudo_obb_weight"]) <= 0:
        raise ValueError("candidate pseudo_obb_weight must be positive")
    return config


def best_metric(directory: Path) -> tuple[float, int]:
    path = directory / "results.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scored = [(float(row[METRIC]), int(float(row["epoch"]))) for row in rows if row.get(METRIC)]
    if not scored:
        raise RuntimeError(f"No {METRIC} values in {path}")
    return max(scored)


def complete(directory: Path) -> bool:
    try:
        best_metric(directory)
    except (FileNotFoundError, RuntimeError, ValueError):
        return False
    return (directory / "weights" / "best.pt").is_file()


def launch(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ctq_reliable_v3_screen.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = load_config(config_path)
    data, unlabeled, catalog, model = (resolve(config[key]) for key in ("data", "unlabeled_root", "unlabeled_catalog", "model"))
    for required in (data, unlabeled, catalog, model):
        if not required.exists():
            raise FileNotFoundError(required)
    output = resolve(args.output_dir or config["output_dir"])
    if output.exists() and any(output.iterdir()) and not args.reuse_existing:
        raise FileExistsError(f"Output already exists: {output}; use a new directory or --reuse-existing")
    output.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        print(f"Preflight passed: {len(config['seeds']) * 3} validation-only training runs; no test data configured.")
        return 0
    rows: list[dict[str, Any]] = []
    base, adapt, calibration, candidate = config["baseline"], config["adaptation"], config["calibration"], config["candidate"]
    for seed in [int(item) for item in config["seeds"]]:
        baseline_dir = output / "baseline" / f"seed{seed}"
        if not (args.reuse_existing and complete(baseline_dir)):
            launch([sys.executable, "-m", "scripts.train_supervised_obb", "--data", str(data), "--init-model", str(model), "--output-dir", str(baseline_dir), "--epochs", str(base["epochs"]), "--batch", str(base["batch"]), "--imgsz", str(base["imgsz"]), "--workers", str(base["workers"]), "--seed", str(seed), "--device", str(config["device"])])
        score, epoch = best_metric(baseline_dir)
        rows.append({"method": "baseline", "seed": seed, "best_epoch": epoch, "best_val_map50_95": score, "thresholds": "", "run_dir": str(baseline_dir)})
        threshold_path = output / "calibration" / f"seed{seed}.json"
        if not (args.reuse_existing and threshold_path.is_file()):
            threshold_path.parent.mkdir(parents=True, exist_ok=True)
            launch([sys.executable, "-m", "scripts.calibrate_pseudo_thresholds", "--checkpoint", str(baseline_dir / "weights" / "best.pt"), "--data", str(data), "--split", "val", "--output", str(threshold_path), "--imgsz", str(adapt["imgsz"]), "--batch", str(adapt["batch"]), "--workers", str(adapt["workers"]), "--device", str(config["device"]), "--iou-match", str(calibration["iou_match"]), "--precision-target", str(calibration["precision_target"]), "--min-predictions", str(calibration["min_predictions"]), "--min-confidence", str(calibration["min_confidence"]), "--max-confidence", str(calibration["max_confidence"]), "--step", str(calibration["step"]), "--fallback", str(calibration["fallback"])])
        thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))["thresholds"]
        threshold_text = ",".join(f"{float(value):.4f}" for value in thresholds)
        for method, module, extra in (
            ("control", "scripts.train_ctq_quality_obb", ["--quality-alpha", "1.0", "--pseudo-obb-weight", "0", "--feature-consistency-weight", "0"]),
            ("reliable_ctq", "scripts.train_ctq_reliable_obb", ["--quality-alpha", str(candidate["quality_alpha"]), "--pseudo-obb-weight", str(candidate["pseudo_obb_weight"]), "--teacher-jitter-views", str(candidate["teacher_jitter_views"]), "--teacher-jitter-iou", str(candidate["teacher_jitter_iou"]), "--teacher-jitter-agreement", str(candidate["teacher_jitter_agreement"])]),
        ):
            directory = output / method / f"seed{seed}"
            if not (args.reuse_existing and complete(directory)):
                command = [sys.executable, "-m", module, *extra, "--data", str(data), "--unlabeled-root", str(unlabeled), "--unlabeled-catalog", str(catalog), "--model", str(model), "--init-checkpoint", str(baseline_dir / "weights" / "best.pt"), "--output-dir", str(directory), "--adapt-epochs", str(adapt["epochs"]), "--pseudo-thresholds", threshold_text, "--pseudo-iou-threshold", str(adapt["pseudo_iou_threshold"]), "--pseudo-class-margin", str(adapt["pseudo_class_margin"]), "--pseudo-warmup-epochs", str(adapt["pseudo_warmup_epochs"]), "--pseudo-rampup-epochs", str(adapt["pseudo_rampup_epochs"]), "--batch", str(adapt["batch"]), "--unlabeled-batch", str(adapt["unlabeled_batch"]), "--imgsz", str(adapt["imgsz"]), "--unlabeled-imgsz", str(adapt["unlabeled_imgsz"]), "--workers", str(adapt["workers"]), "--unlabeled-workers", str(adapt["unlabeled_workers"]), "--seed", str(seed), "--device", str(config["device"]), "--stage1-only", "--exist-ok"]
                launch(command)
            score, epoch = best_metric(directory)
            rows.append({"method": method, "seed": seed, "best_epoch": epoch, "best_val_map50_95": score, "thresholds": threshold_text, "run_dir": str(directory)})
        write_csv(output / "per_seed_validation.csv", rows)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(float(row["best_val_map50_95"]))
    ranking = [{"method": method, "n_seeds": len(values), "mean_best_val_map50_95": mean(values), "std_best_val_map50_95": stdev(values) if len(values) > 1 else 0.0} for method, values in grouped.items()]
    ranking.sort(key=lambda item: float(item["mean_best_val_map50_95"]), reverse=True)
    write_csv(output / "validation_ranking.csv", ranking)
    (output / "protocol.json").write_text(json.dumps({"config": str(config_path), "test_used_for_selection": False, "methods": ["baseline", "control", "reliable_ctq"]}, indent=2), encoding="utf-8")
    print(f"Validation-only screen complete: {output / 'validation_ranking.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
