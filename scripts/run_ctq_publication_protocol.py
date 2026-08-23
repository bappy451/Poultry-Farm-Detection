"""Run a restartable, validation-only CTQ publication experiment protocol.

The protocol creates matched baseline, continuation-control, and CTQ runs for
every configured seed.  It does not access a test dataset and never overwrites
an existing run unless --reuse-existing is explicitly selected.
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


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"protocol_name", "data", "unlabeled_root", "init_model", "output_dir", "seeds", "baseline", "adaptation", "control", "candidates"}
    if not isinstance(config, dict) or required - config.keys():
        missing = sorted(required - (config.keys() if isinstance(config, dict) else set()))
        raise ValueError(f"Invalid protocol config; missing: {missing}")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or len({int(seed) for seed in seeds}) != len(seeds):
        raise ValueError("'seeds' must be a non-empty list of unique integers.")
    candidate_ids: set[str] = set()
    for candidate in config["candidates"]:
        if not isinstance(candidate, dict) or {"id", "quality_alpha", "pseudo_obb_weight"} - candidate.keys():
            raise ValueError("Every candidate requires id, quality_alpha, and pseudo_obb_weight.")
        identifier = str(candidate["id"])
        if not identifier.replace("_", "").isalnum() or identifier in candidate_ids:
            raise ValueError(f"Candidate ids must be unique alphanumeric/underscore strings: {identifier!r}")
        if float(candidate["quality_alpha"]) < 0 or float(candidate["pseudo_obb_weight"]) < 0:
            raise ValueError(f"Candidate {identifier} has a negative hyperparameter.")
        candidate_ids.add(identifier)
    return config


def best_metric(run_dir: Path) -> tuple[float, int]:
    results = run_dir / "results.csv"
    if not results.is_file():
        raise FileNotFoundError(f"Missing results file: {results}")
    with results.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scored = []
    for row in rows:
        try:
            score = float(row[METRIC])
            epoch = int(float(row["epoch"]))
        except (KeyError, TypeError, ValueError):
            continue
        if score == score:
            scored.append((score, epoch))
    if not scored:
        raise RuntimeError(f"No finite {METRIC} values in {results}")
    return max(scored)


def complete(run_dir: Path) -> bool:
    try:
        best_metric(run_dir)
    except (FileNotFoundError, RuntimeError):
        return False
    return (run_dir / "weights" / "best.pt").is_file()


def launch(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def baseline_command(config: dict[str, Any], seed: int, output: Path) -> list[str]:
    base = config["baseline"]
    return [
        sys.executable, "-m", "scripts.train_supervised_obb",
        "--data", str(resolve(config["data"])), "--init-model", str(resolve(config["init_model"])),
        "--output-dir", str(output), "--epochs", str(base["epochs"]), "--batch", str(base["batch"]),
        "--imgsz", str(base["imgsz"]), "--workers", str(base["workers"]), "--seed", str(seed),
        "--device", str(config["device"]),
    ]


def adaptation_command(config: dict[str, Any], setting: dict[str, Any], seed: int, checkpoint: Path, output: Path) -> list[str]:
    adapt = config["adaptation"]
    return [
        sys.executable, "-m", "scripts.train_ctq_quality_obb",
        "--quality-alpha", str(setting["quality_alpha"]), "--data", str(resolve(config["data"])),
        "--unlabeled-root", str(resolve(config["unlabeled_root"])), "--model", str(resolve(config["init_model"])),
        "--init-checkpoint", str(checkpoint), "--output-dir", str(output),
        "--adapt-epochs", str(adapt["epochs"]), "--pseudo-obb-weight", str(setting["pseudo_obb_weight"]),
        "--pseudo-thresholds", str(adapt["pseudo_thresholds"]), "--pseudo-iou-threshold", str(adapt["pseudo_iou_threshold"]),
        "--pseudo-class-margin", str(adapt["pseudo_class_margin"]), "--pseudo-warmup-epochs", str(adapt["pseudo_warmup_epochs"]),
        "--pseudo-rampup-epochs", str(adapt["pseudo_rampup_epochs"]), "--feature-consistency-weight", str(setting.get("feature_consistency_weight", 0.0)),
        "--batch", str(adapt["batch"]), "--unlabeled-batch", str(adapt["unlabeled_batch"]),
        "--imgsz", str(adapt["imgsz"]), "--unlabeled-imgsz", str(adapt["unlabeled_imgsz"]),
        "--seed", str(seed), "--device", str(config["device"]), "--stage1-only", "--exist-ok",
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method_id"])].append(row)
    baseline_by_seed = {int(row["seed"]): float(row["best_val_map50_95"]) for row in grouped["baseline"]}
    summary = []
    for method_id, group in grouped.items():
        scores = [float(row["best_val_map50_95"]) for row in group]
        deltas = [float(row["best_val_map50_95"]) - baseline_by_seed[int(row["seed"])] for row in group]
        summary.append({
            "method_id": method_id,
            "n_seeds": len(group),
            "mean_best_val_map50_95": mean(scores),
            "std_best_val_map50_95": stdev(scores) if len(scores) > 1 else 0.0,
            "mean_delta_vs_baseline": mean(deltas),
        })
    return sorted(summary, key=lambda row: float(row["mean_best_val_map50_95"]), reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ctq_publication_protocol_v1.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true", help="Resume only fully completed runs in this protocol output folder.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_config(config_path)
    for required in (resolve(config["data"]), resolve(config["unlabeled_root"]), resolve(config["init_model"])):
        if not required.exists():
            raise FileNotFoundError(required)
    output = resolve(args.output_dir or config["output_dir"])
    if output.exists() and any(output.iterdir()) and not args.reuse_existing:
        raise FileExistsError(f"Refusing to mix experiments in existing output directory: {output}. Use a new --output-dir or --reuse-existing.")
    output.mkdir(parents=True, exist_ok=True)
    jobs_per_seed = 2 + len(config["candidates"])
    manifest = {
        "protocol_name": config["protocol_name"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": [int(seed) for seed in config["seeds"]], "selection_metric": METRIC,
        "external_test_used": False, "jobs": jobs_per_seed * len(config["seeds"]), "config": str(config_path),
    }
    (output / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.preflight_only:
        print(f"Preflight passed: {manifest['jobs']} runs ({len(config['seeds'])} seeds × {jobs_per_seed} methods).")
        return 0

    rows: list[dict[str, Any]] = []
    for raw_seed in config["seeds"]:
        seed = int(raw_seed)
        baseline_dir = output / "baseline" / f"seed{seed}"
        if not (args.reuse_existing and complete(baseline_dir)):
            launch(baseline_command(config, seed, baseline_dir))
        score, epoch = best_metric(baseline_dir)
        rows.append({"method_id": "baseline", "seed": seed, "best_epoch": epoch, "best_val_map50_95": score, "run_dir": str(baseline_dir)})
        checkpoint = baseline_dir / "weights" / "best.pt"
        settings = [{"id": "control", **config["control"]}, *config["candidates"]]
        for setting in settings:
            method_id = str(setting["id"])
            run_dir = output / method_id / f"seed{seed}"
            if not (args.reuse_existing and complete(run_dir)):
                launch(adaptation_command(config, setting, seed, checkpoint, run_dir))
            score, epoch = best_metric(run_dir)
            rows.append({"method_id": method_id, "seed": seed, "best_epoch": epoch, "best_val_map50_95": score, "run_dir": str(run_dir)})
        write_csv(output / "per_seed_validation.csv", rows)

    ranking = summarize(rows)
    write_csv(output / "validation_ranking.csv", ranking)
    (output / "validation_summary.json").write_text(json.dumps({"ranking": ranking, "external_test_used": False}, indent=2), encoding="utf-8")
    print(f"Completed validation-only protocol. Ranking: {output / 'validation_ranking.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
