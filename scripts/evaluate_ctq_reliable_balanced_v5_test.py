"""Evaluate the frozen validation-selected CTQ screen once on untouched test-v2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

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


def verify_v2_provenance(data: Path) -> dict[str, Any]:
    payload = yaml.safe_load(data.read_text(encoding="utf-8")) or {}
    root_value = payload.get("path")
    if not root_value:
        raise ValueError("test-v2 YAML must contain its dataset root in 'path'.")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (data.parent / root).resolve()
    audit_path = root / "source_disjoint_split_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing source-disjoint audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("implementation") != "create_ctq_source_disjoint_split_v2":
        raise RuntimeError("test-v2 was not created by the collision-safe source-disjoint builder.")
    if any(audit.get(key) for key in ("source_overlap_train_val", "source_overlap_train_test", "source_overlap_val_test")):
        raise RuntimeError("Source-disjoint audit reports source overlap involving test-v2.")
    if int(audit.get("counts", {}).get("test", 0)) <= 0:
        raise RuntimeError("Source-disjoint audit reports an empty test-v2 split.")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", nargs="+", default=["baseline", "control", "reliable_ctq"])
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
    methods = [str(item) for item in args.methods]
    winner = ranking[0].get("method")
    if winner not in methods:
        raise ValueError(f"Validation winner is {winner!r}; include it in --methods.")
    audit = verify_v2_provenance(data)
    seeds = sorted(int(path.name.removeprefix("seed")) for path in (screen / methods[0]).glob("seed*") if path.is_dir())
    if not seeds:
        raise RuntimeError("No seed directories found in screen output.")
    for method in methods:
        for seed in seeds:
            checkpoint = screen / method / f"seed{seed}" / "weights" / "best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
    from scripts.run_ema_benchmark import audit_test_overlap
    exact = audit_test_overlap(data, data, final_test_split="test")
    if exact["exact_sha256_matches"]:
        raise RuntimeError("test-v2 exact-overlap audit failed.")
    output.mkdir(parents=True)
    (output / "frozen_protocol.json").write_text(json.dumps({"screen_dir": str(screen), "data": str(data), "validation_winner": winner, "methods": methods, "seeds": seeds, "source_disjoint_audit": audit, "exact_overlap_audit": exact, "test_used_for_selection": False}, indent=2), encoding="utf-8")
    from ultralytics import YOLO
    rows: list[dict[str, Any]] = []
    for method in methods:
        for seed in seeds:
            checkpoint = screen / method / f"seed{seed}" / "weights" / "best.pt"
            result = YOLO(str(checkpoint)).val(data=str(data), split="test", imgsz=args.imgsz, batch=args.batch, workers=args.workers, device=args.device, plots=False, project=str(output / "runs"), name=f"{method}_seed{seed}", exist_ok=True, verbose=False)
            values = getattr(result, "results_dict", {}) or {}
            rows.append({"method": method, "seed": seed, "checkpoint": str(checkpoint.resolve()), **{metric: float(values[metric]) for metric in METRICS}})
    write(output / "per_seed_test_metrics.csv", rows)
    summary: list[dict[str, Any]] = []
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
