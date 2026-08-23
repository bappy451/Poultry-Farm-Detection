"""Run the clean raw-train-only label-efficiency experiment.

The script creates deterministic, nested labeled subsets and runs the same
EMA-vs-Ultralytics benchmark for every fraction.  It deliberately delegates
training/evaluation to ``run_ema_benchmark.py`` so checkpoint selection stays
validation-only and the independent test set is only used for final reporting.

Pilot mode runs one seed at every fraction.  The default full mode runs all
requested seeds (0, 7 and 42).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _csv_value(row: dict[str, str], names: Iterable[str]) -> str:
    lowered = {str(k).lower().replace("_", "-"): v for k, v in row.items()}
    for name in names:
        key = name.lower().replace("_", "-")
        if key in lowered and lowered[key] not in (None, ""):
            return str(lowered[key])
    return ""


def _metric_value(row: dict[str, str]) -> str:
    """Find mAP50-95, preferring raw per-seed values over aggregate min/max."""
    aggregate_mean = ""
    for key, value in row.items():
        normalized = str(key).lower().replace("_", "-")
        if "map50-95" not in normalized:
            continue
        if "/mean" in normalized:
            aggregate_mean = str(value)
        elif all(token not in normalized for token in ("/std", "/min", "/max")):
            return str(value)
    return aggregate_mean


def _read_final_rows(output_dir: Path, fraction: float) -> list[dict[str, object]]:
    """Read the benchmark's final-test rows without assuming column spelling."""

    # Prefer per-seed rows; aggregate_test_metrics.csv contains only mean/std.
    candidates = sorted(output_dir.rglob("per_seed_test_metrics.csv"))
    if not candidates:
        candidates = sorted(output_dir.rglob("test_metrics.csv"))
    if not candidates:
        candidates = sorted(output_dir.rglob("aggregate_test_metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No final-test CSV was produced under {output_dir}")

    rows: list[dict[str, object]] = []
    with candidates[-1].open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            method = _csv_value(raw, ("method", "experiment"))
            method_key = method.lower().replace(" ", "_").replace("-", "_")
            if method and method_key not in {"ema", "ema_adaptation", "ema_adapted", "ultralytics_pretrained", "baseline"}:
                continue
            seed = _csv_value(raw, ("seed", "random_seed"))
            metric = _metric_value(raw)
            if not metric:
                continue
            try:
                value = float(metric)
            except ValueError:
                continue
            rows.append(
                {
                    "fraction_percent": fraction * 100.0,
                    "method": method or "unknown",
                    "seed": int(seed) if seed else "",
                    "test_map50_95": value,
                    "source_csv": str(candidates[-1]),
                }
            )
    if not rows:
        raise RuntimeError(f"Could not find mAP50-95 rows in {candidates[-1]}")
    return rows


def _write_curve(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "label_efficiency_curve_raw.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["fraction_percent", "method", "seed", "test_map50_95", "source_csv"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[float, str], list[float]] = {}
    for row in rows:
        key = (float(row["fraction_percent"]), str(row["method"]))
        grouped.setdefault(key, []).append(float(row["test_map50_95"]))
    summary: list[dict[str, object]] = []
    for (fraction, method), values in sorted(grouped.items()):
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1) if len(values) > 1 else 0.0
        summary.append(
            {
                "fraction_percent": fraction,
                "method": method,
                "n_seeds": len(values),
                "test_map50_95_mean": mean,
                "test_map50_95_std": variance**0.5,
            }
        )
    with (output_dir / "label_efficiency_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["fraction_percent", "method", "n_seeds", "test_map50_95_mean", "test_map50_95_std"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("\n$ " + " ".join(f'\"{x}\"' if " " in x else x for x in cmd), flush=True)
    completed = subprocess.run(cmd, cwd=cwd)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-data", default="configs/poultry_clean_raw_90_10.yaml")
    parser.add_argument("--final-test-data", default="temp/poultry_houses_test/data.yaml")
    parser.add_argument("--unlabeled-root", default=r"E:\NAIP_Dataset_PH\MS_tiles_512_256_jpg")
    parser.add_argument("--subset-config-dir", default="configs/poultry_clean_label_subsets")
    parser.add_argument("--subset-output-dir", default="datasets/poultry_clean_label_subsets")
    parser.add_argument("--output-dir", default="runs/label_efficiency_clean")
    parser.add_argument("--fractions", default="1,5,10,25,50,100")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 7, 42])
    parser.add_argument("--pilot", action="store_true", help="Run only the first seed at every fraction")
    parser.add_argument("--device", default="0")
    parser.add_argument("--model", default="yolo11l-obb.pt")
    parser.add_argument("--random-model", default="yolo11l-obb.yaml")
    parser.add_argument("--adapt-epochs", type=int, default=60)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--unlabeled-batch", type=int, default=16)
    parser.add_argument("--unlabeled-limit", type=int, default=250000)
    parser.add_argument("--test-batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--unlabeled-imgsz", type=int, default=448)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--unlabeled-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--overwrite-subsets", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse completed pilot checkpoints when extending to more seeds")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    clean_data = _path(args.clean_data)
    final_test_data = _path(args.final_test_data)
    subset_config_dir = _path(args.subset_config_dir)
    subset_output_dir = _path(args.subset_output_dir)
    output_dir = _path(args.output_dir)
    fractions = [float(x.strip()) / 100.0 for x in args.fractions.split(",") if x.strip()]
    if not fractions or any(f <= 0 or f > 1 for f in fractions):
        raise ValueError("--fractions must be comma-separated percentages in (0, 100]")
    seeds = [args.seeds[0]] if args.pilot else list(args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required")

    generator = ROOT / "scripts" / "create_clean_label_subsets.py"
    generation_cmd = [
        sys.executable,
        str(generator),
        "--clean-data",
        str(clean_data),
        "--output-dir",
        str(subset_output_dir),
        "--config-dir",
        str(subset_config_dir),
        "--fractions",
        ",".join(str(int(round(f * 100))) for f in fractions),
    ]
    # Regeneration is deterministic and only rewrites manifests/YAMLs; it never touches images.
    generation_cmd.append("--overwrite")
    _run(generation_cmd, cwd=ROOT)

    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "raw-train-only, deterministic nested labeled subsets",
        "clean_data": str(clean_data),
        "final_test_data": str(final_test_data),
        "fractions": [f * 100.0 for f in fractions],
        "seeds": seeds,
        "pilot": bool(args.pilot),
        "methods": ["ultralytics_pretrained", "ema_adaptation"],
        "checkpoint_selection": "best validation metric; final test is reporting-only",
        "adapt_epochs": args.adapt_epochs,
        "finetune_epochs": args.finetune_epochs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    all_rows: list[dict[str, object]] = []
    for fraction in fractions:
        percent = int(round(fraction * 100))
        subset_yaml = subset_config_dir / f"poultry_clean_{percent}percent.yaml"
        fraction_dir = output_dir / f"fraction_{percent:03d}"
        fraction_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== Clean label-efficiency fraction {percent}% | seeds {seeds} =====", flush=True)
        benchmark_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_ema_benchmark.py"),
            "--data",
            str(subset_yaml),
            "--final-test-data",
            str(final_test_data),
            "--final-test-split",
            "test",
            "--unlabeled-root",
            args.unlabeled_root,
            "--model",
            args.model,
            "--random-model",
            args.random_model,
            "--output-dir",
            str(fraction_dir),
            # The subset YAML itself defines the labeled set.  Do not sample it again.
            "--fraction",
            "1.0",
            "--adapt-epochs",
            str(args.adapt_epochs),
            "--finetune-epochs",
            str(args.finetune_epochs),
            "--batch",
            str(args.batch),
            "--unlabeled-batch",
            str(args.unlabeled_batch),
            "--unlabeled-limit",
            str(args.unlabeled_limit),
            "--test-batch",
            str(args.test_batch),
            "--imgsz",
            str(args.imgsz),
            "--unlabeled-imgsz",
            str(args.unlabeled_imgsz),
            "--workers",
            str(args.workers),
            "--unlabeled-workers",
            str(args.unlabeled_workers),
            "--prefetch-factor",
            str(args.prefetch_factor),
            "--device",
            args.device,
            "--seeds",
            *[str(s) for s in seeds],
        ]
        # No --include-random: this protocol compares exactly the requested two methods.
        if args.reuse_existing:
            benchmark_cmd.append("--reuse-existing")
        _run(benchmark_cmd, cwd=ROOT)
        all_rows.extend(_read_final_rows(fraction_dir, fraction))
        _write_curve(all_rows, output_dir)
        print(f"Completed {percent}%; curve updated at {output_dir / 'label_efficiency_curve.csv'}", flush=True)

    _write_curve(all_rows, output_dir)
    print("\nClean label-efficiency experiment complete.", flush=True)
    print(f"Raw results: {output_dir / 'label_efficiency_curve_raw.csv'}", flush=True)
    print(f"Summary:     {output_dir / 'label_efficiency_curve.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())








