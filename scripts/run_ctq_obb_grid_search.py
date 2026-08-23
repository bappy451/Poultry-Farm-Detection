"""Run a validation-only, multi-seed loss search for EMA pseudo-OBB training.

The runner never opens an external test dataset.  It records the best
validation mAP50-95 for each seed and candidate, ranks means across matched
seeds, and emits the exact confirmation command for the provisional winner.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
METRIC = "metrics/mAP50-95(B)"
REQUIRED_CANDIDATE_KEYS = {"id", "loss_family", "pseudo_obb_weight", "pseudo_thresholds", "feature_consistency_weight"}


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Grid-search config must contain a mapping.")
    for key in ("data", "unlabeled_root", "model", "output_dir", "seeds", "baseline", "training", "candidates"):
        if key not in config:
            raise ValueError(f"Grid-search config is missing '{key}'.")
    if not isinstance(config["seeds"], list) or not config["seeds"]:
        raise ValueError("'seeds' must be a non-empty list.")
    if not isinstance(config["candidates"], list) or not config["candidates"]:
        raise ValueError("'candidates' must be a non-empty list.")
    candidate_ids: set[str] = set()
    for candidate in config["candidates"]:
        if not isinstance(candidate, dict) or REQUIRED_CANDIDATE_KEYS - candidate.keys():
            missing = REQUIRED_CANDIDATE_KEYS - (candidate.keys() if isinstance(candidate, dict) else set())
            raise ValueError(f"Candidate is missing required keys: {sorted(missing)}")
        candidate_id = str(candidate["id"])
        if not candidate_id.replace("_", "").isalnum() or candidate_id in candidate_ids:
            raise ValueError(f"Candidate id must be unique alphanumeric/underscore: {candidate_id!r}")
        candidate_ids.add(candidate_id)
        if float(candidate["pseudo_obb_weight"]) < 0 or float(candidate["feature_consistency_weight"]) < 0:
            raise ValueError(f"Candidate {candidate_id} has a negative loss weight.")
    return config


def best_validation_map(run_dir: Path) -> float:
    results = run_dir / "results.csv"
    if not results.is_file():
        raise FileNotFoundError(f"Missing validation results: {results}")
    with results.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = []
    for row in rows:
        try:
            value = float(row[METRIC])
        except (KeyError, TypeError, ValueError):
            continue
        if value == value:  # exclude NaN without importing math
            values.append(value)
    if not values:
        raise RuntimeError(f"No finite {METRIC} values in {results}")
    return max(values)


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(f'\"{item}\"' if " " in item else item for item in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def baseline_command(config: dict[str, Any], *, seed: int, run_dir: Path) -> list[str]:
    baseline = config["baseline"]
    training = config["training"]
    return [
        sys.executable, str(ROOT / "scripts" / "train_supervised_obb.py"),
        "--data", str(project_path(config["data"])), "--init-model", str(project_path(config["model"])),
        "--output-dir", str(run_dir), "--epochs", str(baseline["epochs"]),
        "--batch", str(training["batch"]), "--imgsz", str(training["imgsz"]),
        "--workers", str(training["workers"]), "--device", str(config.get("device", "0")),
        "--seed", str(seed), "--fraction", str(config.get("fraction", 1.0)),
        "--lr0", str(baseline["lr0"]), "--weight-decay", str(baseline["weight_decay"]),
    ]


def candidate_command(config: dict[str, Any], candidate: dict[str, Any], *, seed: int, baseline: Path, run_dir: Path) -> list[str]:
    training = config["training"]
    command = [
        sys.executable, str(ROOT / "scripts" / "train_ema_pseudo_obb.py"),
        "--data", str(project_path(config["data"])), "--unlabeled-root", str(project_path(config["unlabeled_root"])),
        "--model", str(project_path(config["model"])), "--init-checkpoint", str(baseline),
        "--output-dir", str(run_dir), "--adapt-epochs", str(training["adapt_epochs"]),
        "--finetune-epochs", str(training["finetune_epochs"]), "--batch", str(training["batch"]),
        "--unlabeled-batch", str(training["unlabeled_batch"]), "--imgsz", str(training["imgsz"]),
        "--unlabeled-imgsz", str(training["unlabeled_imgsz"]), "--workers", str(training["workers"]),
        "--unlabeled-workers", str(training["unlabeled_workers"]), "--unlabeled-limit", str(training["unlabeled_limit"]),
        "--prefetch-factor", str(training["prefetch_factor"]), "--device", str(config.get("device", "0")),
        "--seed", str(seed), "--fraction", str(config.get("fraction", 1.0)),
        "--pseudo-obb-weight", str(candidate["pseudo_obb_weight"]), "--pseudo-thresholds", str(candidate["pseudo_thresholds"]),
        "--pseudo-iou-threshold", str(candidate.get("pseudo_iou_threshold", 0.5)),
        "--pseudo-class-margin", str(candidate.get("pseudo_class_margin", 0.05)),
        "--pseudo-max-det", str(candidate.get("pseudo_max_det", 100)),
        "--pseudo-warmup-epochs", str(candidate.get("pseudo_warmup_epochs", 5)),
        "--pseudo-rampup-epochs", str(candidate.get("pseudo_rampup_epochs", 10)),
        "--feature-consistency-weight", str(candidate["feature_consistency_weight"]),
        "--feature-rampup-epochs", str(candidate.get("feature_rampup_epochs", 10)),
        "--foreground-feature-gain", str(candidate.get("foreground_feature_gain", 3.0)),
        "--ema-decay", str(training["ema_decay"]), "--adapt-lr0", str(training["adapt_lr0"]),
        "--finetune-lr0", str(training["finetune_lr0"]), "--exist-ok",
    ]
    return command


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_candidate[str(row["candidate_id"])].append(float(row["best_val_map50_95"]))
    summary = []
    for candidate_id, values in by_candidate.items():
        summary.append({
            "candidate_id": candidate_id,
            "n_seeds": len(values),
            "mean_best_val_map50_95": mean(values),
            "std_best_val_map50_95": stdev(values) if len(values) > 1 else 0.0,
            "min_best_val_map50_95": min(values),
            "max_best_val_map50_95": max(values),
        })
    return sorted(summary, key=lambda item: float(item["mean_best_val_map50_95"]), reverse=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ctq_obb_grid_search.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--candidates", nargs="+", default=None, help="Run only these candidate ids.")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_config(config_path)
    data = project_path(config["data"])
    model = project_path(config["model"])
    unlabeled_root = project_path(config["unlabeled_root"])
    for required in (data, model, unlabeled_root):
        if not required.exists():
            raise FileNotFoundError(required)
    seeds = args.seeds if args.seeds is not None else [int(seed) for seed in config["seeds"]]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique and non-empty.")
    candidate_filter = set(args.candidates or [])
    candidates = [item for item in config["candidates"] if not candidate_filter or item["id"] in candidate_filter]
    if candidate_filter and {item["id"] for item in candidates} != candidate_filter:
        raise ValueError("Unknown candidate id supplied to --candidates.")
    output_dir = project_path(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "config": str(config_path), "data": str(data), "model": str(model), "unlabeled_root": str(unlabeled_root), "seeds": seeds, "candidate_ids": [item["id"] for item in candidates], "selection_metric": METRIC, "external_test_used": False}
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if args.preflight_only:
        print(f"Preflight complete: {len(candidates)} candidates × {len(seeds)} seeds; output={output_dir}")
        return 0

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        baseline_dir = output_dir / "baselines" / f"seed{seed}"
        baseline_checkpoint = baseline_dir / "weights" / "best.pt"
        if not (args.reuse_existing and baseline_checkpoint.is_file()):
            run_command(baseline_command(config, seed=seed, run_dir=baseline_dir))
        baseline_score = best_validation_map(baseline_dir)
        rows.append({"candidate_id": "ultralytics_baseline", "loss_family": "supervised", "seed": seed, "best_val_map50_95": baseline_score, "run_dir": str(baseline_dir)})
        for candidate in candidates:
            run_dir = output_dir / "candidates" / str(candidate["id"]) / f"seed{seed}"
            checkpoint = run_dir.with_name(run_dir.name + "_finetune") / "weights" / "best.pt"
            if not (args.reuse_existing and checkpoint.is_file()):
                run_command(candidate_command(config, candidate, seed=seed, baseline=baseline_checkpoint, run_dir=run_dir))
            score = best_validation_map(run_dir.with_name(run_dir.name + "_finetune"))
            rows.append({"candidate_id": candidate["id"], "loss_family": candidate["loss_family"], "seed": seed, "best_val_map50_95": score, "run_dir": str(run_dir)})
            write_csv(output_dir / "per_seed_validation.csv", rows)

    summary = aggregate(rows)
    baseline = next(item for item in summary if item["candidate_id"] == "ultralytics_baseline")
    winner = next((item for item in summary if item["candidate_id"] != "ultralytics_baseline"), None)
    if winner is not None:
        winner["delta_vs_baseline"] = float(winner["mean_best_val_map50_95"]) - float(baseline["mean_best_val_map50_95"])
        confirmation = f"python scripts/run_ctq_obb_grid_search.py --config {config_path.relative_to(ROOT)} --seeds 7 24 42 --candidates {winner['candidate_id']} --output-dir {output_dir / 'confirmation'}"
        (output_dir / "confirmation_command.txt").write_text(confirmation + "\n", encoding="utf-8")
    write_csv(output_dir / "validation_ranking.csv", summary)
    (output_dir / "validation_summary.json").write_text(json.dumps({"ranking": summary, "provisional_winner": winner, "external_test_used": False}, indent=2), encoding="utf-8")
    print(f"Validation ranking written to {output_dir / 'validation_ranking.csv'}")
    if winner is not None:
        print(f"Provisional winner: {winner['candidate_id']} (Δ mAP50-95={winner['delta_vs_baseline']:+.4f} vs baseline).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
