"""Screen supervised YOLO11l-OBB image sizes on validation only."""

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


def score(path: Path) -> tuple[float, int]:
    results = path / "results.csv"
    with results.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = [(float(row[METRIC]), int(float(row["epoch"]))) for row in rows if row.get(METRIC)]
    if not values:
        raise RuntimeError(f"No {METRIC} in {results}")
    return max(values)


def done(path: Path) -> bool:
    try:
        score(path)
    except (FileNotFoundError, RuntimeError, ValueError):
        return False
    return (path / "weights" / "best.pt").is_file()


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ctq_supervised_resolution_v4.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    required = {"data", "model", "output_dir", "device", "seeds", "epochs", "workers", "candidates"}
    if required - config.keys():
        raise ValueError(f"Missing config keys: {sorted(required - config.keys())}")
    data, model = resolve(config["data"]), resolve(config["model"])
    if not data.is_file() or not model.is_file():
        raise FileNotFoundError("Source-disjoint YAML or model checkpoint is missing.")
    candidates = config["candidates"]
    if not candidates or len({item["id"] for item in candidates}) != len(candidates):
        raise ValueError("Candidates must have unique ids.")
    for item in candidates:
        if int(item["imgsz"]) % 32 or int(item["batch"]) <= 0:
            raise ValueError("Each imgsz must be divisible by 32 and batch positive.")
    output = resolve(args.output_dir or config["output_dir"])
    if output.exists() and any(output.iterdir()) and not args.reuse_existing:
        raise FileExistsError(f"Output exists: {output}; use a new directory or --reuse-existing")
    output.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        print(f"Preflight passed: {len(candidates) * len(config['seeds'])} validation-only supervised runs.")
        return 0
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for seed in [int(value) for value in config["seeds"]]:
            directory = output / str(candidate["id"]) / f"seed{seed}"
            if not (args.reuse_existing and done(directory)):
                command = [sys.executable, "-m", "scripts.train_supervised_obb", "--data", str(data), "--init-model", str(model), "--output-dir", str(directory), "--epochs", str(config["epochs"]), "--batch", str(candidate["batch"]), "--imgsz", str(candidate["imgsz"]), "--workers", str(config["workers"]), "--seed", str(seed), "--device", str(config["device"])]
                print("$ " + " ".join(command), flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
            value, epoch = score(directory)
            rows.append({"candidate_id": candidate["id"], "imgsz": candidate["imgsz"], "batch": candidate["batch"], "seed": seed, "best_epoch": epoch, "best_val_map50_95": value, "run_dir": str(directory)})
            write(output / "per_seed_validation.csv", rows)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate_id"])].append(float(row["best_val_map50_95"]))
    ranking = [{"candidate_id": key, "n_seeds": len(values), "mean_best_val_map50_95": mean(values), "std_best_val_map50_95": stdev(values) if len(values) > 1 else 0.0} for key, values in grouped.items()]
    ranking.sort(key=lambda row: float(row["mean_best_val_map50_95"]), reverse=True)
    write(output / "validation_ranking.csv", ranking)
    (output / "protocol.json").write_text(json.dumps({"config": str(config_path), "test_used_for_selection": False}, indent=2), encoding="utf-8")
    print(f"Resolution screen complete: {output / 'validation_ranking.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
