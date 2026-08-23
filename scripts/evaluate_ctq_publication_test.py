"""Evaluate the frozen CTQ publication finalists exactly once on held-out test data.

This script never trains models and never uses final-test metrics for model
selection. It evaluates the validation-selected CTQ candidate, its matched
continuation control, and the supervised baseline for every protocol seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("baseline", "control", "alpha2_weight0025")
METRICS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_protocol(protocol_dir: Path) -> list[int]:
    manifest = protocol_dir / "protocol_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing publication protocol manifest: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("external_test_used") is not False:
        raise ValueError("Protocol manifest does not declare a validation-only selection stage.")
    seeds = [int(seed) for seed in payload.get("seeds", [])]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Protocol manifest has invalid seeds.")
    return seeds


def checkpoint(protocol_dir: Path, method: str, seed: int) -> Path:
    candidate = protocol_dir / method / f"seed{seed}" / "weights" / "best.pt"
    if not candidate.is_file():
        raise FileNotFoundError(f"Frozen checkpoint missing: {candidate}")
    return candidate.resolve()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        item: dict[str, Any] = {"method": method, "n_seeds": len(group)}
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            item[f"{metric}/mean"] = mean(values)
            item[f"{metric}/std"] = stdev(values) if len(values) > 1 else 0.0
        result.append(item)
    return result


def paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed = {(str(row["method"]), int(row["seed"])): row for row in rows}
    output: list[dict[str, Any]] = []
    for compare_to in ("baseline", "control"):
        differences = []
        for seed in sorted({int(row["seed"]) for row in rows}):
            ctq = float(by_seed[("alpha2_weight0025", seed)]["metrics/mAP50-95(B)"])
            reference = float(by_seed[(compare_to, seed)]["metrics/mAP50-95(B)"])
            differences.append(ctq - reference)
        output.append({
            "comparison": f"alpha2_weight0025_minus_{compare_to}",
            "n_seeds": len(differences),
            "mAP50-95_delta_mean": mean(differences),
            "mAP50-95_delta_std": stdev(differences) if len(differences) > 1 else 0.0,
        })
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", default="runs/ctq_publication_v2_run1")
    parser.add_argument("--training-data", default="configs/poultry_clean_label_subsets/poultry_clean_10percent.yaml")
    parser.add_argument("--final-test-data", default="temp/poultry_houses_test/data.yaml")
    parser.add_argument("--final-test-split", default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="0")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true", help="Resume only rows already written to per_seed_test_metrics.csv.")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_dir = resolve(args.protocol_dir)
    training_data = resolve(args.training_data)
    final_test_data = resolve(args.final_test_data)
    output_dir = resolve(args.output_dir) if args.output_dir else protocol_dir / "final_test_frozen"
    if not training_data.is_file() or not final_test_data.is_file():
        raise FileNotFoundError("Training or final-test YAML is missing.")
    seeds = load_protocol(protocol_dir)
    checkpoints = {(method, seed): checkpoint(protocol_dir, method, seed) for method in METHODS for seed in seeds}

    sys.path.insert(0, str(ROOT))
    from scripts.run_ema_benchmark import audit_test_overlap

    overlap = audit_test_overlap(training_data, final_test_data, final_test_split=args.final_test_split)
    if overlap["training_image_count"] <= 0 or overlap["final_test_image_count"] <= 0:
        raise RuntimeError("Overlap audit resolved an empty training/validation or final-test split.")
    if overlap["exact_sha256_matches"]:
        raise RuntimeError("Exact train/validation-to-test image overlap detected; refusing final-test evaluation.")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.reuse_existing:
        raise FileExistsError(f"Refusing to reuse final-test output directory: {output_dir}. Use a new directory or --reuse-existing.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_overlap_audit.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")
    frozen = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_dir": str(protocol_dir), "training_data": str(training_data), "final_test_data": str(final_test_data),
        "final_test_split": args.final_test_split, "selection_metric": "validation metrics/mAP50-95(B)",
        "selected_ctq_method": "alpha2_weight0025", "test_used_for_selection": False,
        "methods": list(METHODS), "seeds": seeds,
        "checkpoints": {f"{method}_seed{seed}": str(path) for (method, seed), path in checkpoints.items()},
    }
    (output_dir / "frozen_test_protocol.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    if args.preflight_only:
        print(f"Preflight passed: {len(METHODS) * len(seeds)} frozen test evaluations; output={output_dir}")
        return 0

    csv_path = output_dir / "per_seed_test_metrics.csv"
    rows = read_rows(csv_path) if args.reuse_existing else []
    completed = {(str(row["method"]), int(row["seed"])) for row in rows}
    from ultralytics import YOLO

    for method in METHODS:
        for seed in seeds:
            key = (method, seed)
            if key in completed:
                continue
            path = checkpoints[key]
            print(f"[test] {method}, seed={seed}: {path}", flush=True)
            result = YOLO(str(path)).val(
                data=str(final_test_data), split=args.final_test_split, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=args.workers, plots=args.plots,
                project=str(output_dir / "runs"), name=f"{method}_seed{seed}", exist_ok=True, verbose=False,
            )
            values = getattr(result, "results_dict", {}) or {}
            row: dict[str, Any] = {"method": method, "seed": seed, "checkpoint": str(path)}
            for metric in METRICS:
                value = values.get(metric)
                if value is None or not math.isfinite(float(value)):
                    raise RuntimeError(f"Missing finite {metric} for {method}, seed={seed}")
                row[metric] = float(value)
            rows.append(row)
            write_csv(csv_path, rows)

    if len(rows) != len(METHODS) * len(seeds):
        raise RuntimeError("Incomplete final-test result table.")
    summary = aggregate(rows)
    deltas = paired_deltas(rows)
    write_csv(output_dir / "aggregate_test_metrics.csv", summary)
    write_csv(output_dir / "paired_test_deltas.csv", deltas)
    (output_dir / "test_summary.json").write_text(json.dumps({"aggregate": summary, "paired_deltas": deltas}, indent=2), encoding="utf-8")
    print(f"Final-test evaluation complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
