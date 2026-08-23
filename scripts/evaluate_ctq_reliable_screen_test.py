"""Evaluate frozen v4 baseline/control/reliable-CTQ checkpoints on test-v2.

Run only after ``validation_ranking.csv`` has selected the method. This script
does not train, tune, or read validation metrics for selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", required=True)
    parser.add_argument("--data", required=True, help="Source-disjoint test-v2 YAML.")
    parser.add_argument("--methods", nargs="+", default=["baseline", "control", "reliable_ctq"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    screen, data, output = resolve(args.screen_dir), resolve(args.data), resolve(args.output_dir)
    ranking_path = screen / "validation_ranking.csv"
    if not ranking_path.is_file() or not data.is_file():
        raise FileNotFoundError("Validation ranking or source-disjoint data YAML is missing.")
    if output.exists():
        raise FileExistsError(f"Final-test output already exists: {output}")
    with ranking_path.open(encoding="utf-8", newline="") as handle:
        ranking = list(csv.DictReader(handle))
    if not ranking:
        raise RuntimeError("Validation ranking is empty.")
    methods = [str(value) for value in args.methods]
    selected = ranking[0]["method"]
    if selected not in methods:
        raise ValueError(f"Validation winner is {selected!r}; include it in --methods or explicitly choose a new protocol.")
    seeds = sorted(int(path.name.removeprefix("seed")) for path in (screen / methods[0]).glob("seed*") if path.is_dir())
    if not seeds:
        raise RuntimeError("No seed directories found in screen output.")
    for method in methods:
        for seed in seeds:
            if not (screen / method / f"seed{seed}" / "weights" / "best.pt").is_file():
                raise FileNotFoundError(screen / method / f"seed{seed}" / "weights" / "best.pt")
    from scripts.run_ema_benchmark import audit_test_overlap
    overlap = audit_test_overlap(data, data, final_test_split="test")
    if overlap["exact_sha256_matches"]:
        raise RuntimeError("Source-disjoint data YAML has train/validation-to-test exact overlap.")
    output.mkdir(parents=True)
    (output / "test_overlap_audit.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")
    (output / "frozen_protocol.json").write_text(json.dumps({"screen_dir": str(screen), "data": str(data), "validation_winner": selected, "methods": methods, "seeds": seeds, "test_used_for_selection": False}, indent=2), encoding="utf-8")
    from ultralytics import YOLO
    rows: list[dict[str, Any]] = []
    for method in methods:
        for seed in seeds:
            checkpoint = screen / method / f"seed{seed}" / "weights" / "best.pt"
            result = YOLO(str(checkpoint)).val(data=str(data), split="test", imgsz=args.imgsz, batch=args.batch, workers=args.workers, device=args.device, plots=False, project=str(output / "runs"), name=f"{method}_seed{seed}", exist_ok=True, verbose=False)
            values = getattr(result, "results_dict", {}) or {}
            row = {"method": method, "seed": seed, "checkpoint": str(checkpoint.resolve())}
            row.update({metric: float(values[metric]) for metric in METRICS})
            rows.append(row)
    write(output / "per_seed_test_metrics.csv", rows)
    summary = []
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        item: dict[str, Any] = {"method": method, "n_seeds": len(group)}
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            item[f"{metric}/mean"] = mean(values)
            item[f"{metric}/std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    write(output / "aggregate_test_metrics.csv", summary)
    print(f"Frozen test-v2 evaluation complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
