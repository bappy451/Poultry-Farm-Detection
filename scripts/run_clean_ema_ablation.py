"""Run the clean 10% EMA consistency-weight ablation.

Each weight is trained independently for seeds 0, 7 and 42.  The checkpoint
used for the external test is the stage-2 checkpoint selected by validation
inside ``train_ema_adapt.py``; no test metric is used for model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
except ImportError:  # pragma: no cover - the training environment includes Rich
    Progress = None

ROOT = Path(__file__).resolve().parents[1]


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("\n$ " + " ".join(f'\"{x}\"' if " " in x else x for x in cmd), flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode:
        raise SystemExit(result.returncode)


def _find_best(run_dir: Path) -> Path | None:
    """Return only a stage-2 checkpoint with the verified EMA handoff marker."""
    candidates = []
    preferred = run_dir.with_name(run_dir.name + "_finetune") / "weights" / "best.pt"
    if preferred.exists():
        candidates.append(preferred)
    candidates.extend(sorted(run_dir.parent.glob(run_dir.name + "*/weights/best.pt")))
    candidates.extend(sorted(run_dir.rglob("best.pt")) if run_dir.exists() else [])
    seen: set[Path] = set()
    for candidate in reversed(candidates):
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        marker = candidate.parent.parent / "ema_handoff.json"
        if candidate.is_file() and marker.is_file():
            return candidate
    return None


def _metrics_dict(result: object) -> tuple[float, float, float, float]:
    box = getattr(result, "box", None)
    values: list[float] = []
    for attr in ("map50", "map", "mp", "mr"):
        value = getattr(box, attr, float("nan")) if box is not None else float("nan")
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(float("nan"))
    if math.isnan(values[1]):
        result_dict = getattr(result, "results_dict", {}) or {}
        for i, names in enumerate(
            (("metrics/mAP50(B)", "metrics/mAP50"), ("metrics/mAP50-95(B)", "metrics/mAP50-95"), ("metrics/precision(B)",), ("metrics/recall(B)",))
        ):
            for name in names:
                if name in result_dict:
                    values[i] = float(result_dict[name])
                    break
    return tuple(values)  # type: ignore[return-value]


def _aggregate(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["consistency_weight", "n_seeds", "map50_mean", "map50_std", "map50_95_mean", "map50_95_std", "precision_mean", "recall_mean"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        weights = sorted({float(row["consistency_weight"]) for row in rows})
        for weight in weights:
            subset = [row for row in rows if float(row["consistency_weight"]) == weight]
            def vals(key: str) -> list[float]:
                return [float(row[key]) for row in subset if not math.isnan(float(row[key]))]
            def mean_std(key: str) -> tuple[float, float]:
                values = vals(key)
                if not values:
                    return float("nan"), float("nan")
                mean = sum(values) / len(values)
                std = (sum((x - mean) ** 2 for x in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
                return mean, std
            map50_mean, map50_std = mean_std("map50")
            map_mean, map_std = mean_std("map50_95")
            precision_mean, _ = mean_std("precision")
            recall_mean, _ = mean_std("recall")
            writer.writerow({
                "consistency_weight": weight,
                "n_seeds": len(subset),
                "map50_mean": map50_mean,
                "map50_std": map50_std,
                "map50_95_mean": map_mean,
                "map50_95_std": map_std,
                "precision_mean": precision_mean,
                "recall_mean": recall_mean,
            })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/poultry_clean_label_subsets/poultry_clean_10percent.yaml")
    parser.add_argument("--clean-data", default="configs/poultry_clean_raw_90_10.yaml")
    parser.add_argument("--final-test-data", default="temp/poultry_houses_test/data.yaml")
    parser.add_argument("--unlabeled-root", default=r"E:\NAIP_Dataset_PH\MS_tiles_512_256_jpg")
    parser.add_argument("--output-dir", default="runs/ema_consistency_ablation_clean")
    parser.add_argument("--weights", default="0,0.01,0.02,0.05")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 7, 42])
    parser.add_argument("--model", default="yolo11l-obb.pt")
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
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume-partial", action="store_true", help="Resume a stage-1 run when its last.pt exists")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data = _path(args.data)
    clean_data = _path(args.clean_data)
    final_test_data = _path(args.final_test_data)
    output_dir = _path(args.output_dir)
    weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
    if not weights or any(w < 0 for w in weights):
        raise ValueError("--weights must be comma-separated non-negative values")
    if not args.seeds:
        raise ValueError("At least one seed is required")
    if not data.exists():
        raise FileNotFoundError(f"10% subset config not found: {data}. Run the label-efficiency runner first.")

    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "clean raw-train-only 10% EMA consistency-weight ablation",
        "data": str(data),
        "clean_data": str(clean_data),
        "final_test_data": str(final_test_data),
        "weights": weights,
        "seeds": args.seeds,
        "checkpoint_selection": "best validation metric from EMA fine-tuning; independent test reporting only",
        "adapt_epochs": args.adapt_epochs,
        "finetune_epochs": args.finetune_epochs,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    # Audit the exact train+validation images against the external test split.
    try:
        sys.path.insert(0, str(ROOT))
        from scripts.run_ema_benchmark import audit_test_overlap
        audit = audit_test_overlap(data, final_test_data, final_test_split="test")
        if audit["training_image_count"] <= 0 or audit["final_test_image_count"] <= 0:
            raise RuntimeError("Overlap audit resolved an empty training or final-test split; refusing to run the ablation.")
        (output_dir / "test_overlap_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    except Exception as exc:
        (output_dir / "test_overlap_audit_error.txt").write_text(str(exc), encoding="utf-8")
        raise

    total = len(weights) * len(args.seeds) * 2
    rows: list[dict[str, object]] = []
    eval_root = output_dir / "final_test"
    progress = None
    task = None
    if Progress is not None:
        progress = Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn())
        progress.start()
        task = progress.add_task("EMA consistency ablation", total=total)
    try:
        for weight in weights:
            label = f"{weight:.2f}".replace(".", "p")
            for seed in args.seeds:
                run_dir = output_dir / f"consistency_{label}_seed{seed}"
                print(f"\n===== EMA consistency weight {weight:g} | seed {seed} =====", flush=True)
                train_cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "train_ema_adapt.py"),
                    "--unlabeled-root", args.unlabeled_root,
                    "--data", str(data),
                    "--output-dir", str(run_dir),
                    "--model", args.model,
                    "--adapt-epochs", str(args.adapt_epochs),
                    "--finetune-epochs", str(args.finetune_epochs),
                    "--batch", str(args.batch),
                    "--unlabeled-batch", str(args.unlabeled_batch),
                    "--unlabeled-limit", str(args.unlabeled_limit),
                    "--imgsz", str(args.imgsz),
                    "--unlabeled-imgsz", str(args.unlabeled_imgsz),
                    "--workers", str(args.workers),
                    "--unlabeled-workers", str(args.unlabeled_workers),
                    "--prefetch-factor", str(args.prefetch_factor),
                    "--consistency-weight", str(weight),
                    "--fraction", "1.0",
                    "--device", args.device,
                    "--seed", str(seed),
                ]
                best = _find_best(run_dir)
                if best is None:
                    if args.resume_partial and (run_dir / "weights" / "last.pt").exists():
                        train_cmd.append("--resume")
                    _run(train_cmd, cwd=ROOT)
                    best = _find_best(run_dir)
                if best is None:
                    raise FileNotFoundError(f"No fine-tuning best.pt found for {run_dir}")
                if progress is not None and task is not None:
                    progress.update(task, advance=1, description=f"trained weight={weight:g}, seed={seed}")

                from ultralytics import YOLO
                eval_name = f"consistency_{label}_seed{seed}"
                result = YOLO(str(best)).val(
                    data=str(final_test_data),
                    split="test",
                    imgsz=args.imgsz,
                    batch=args.test_batch,
                    device=args.device,
                    workers=args.workers,
                    plots=False,
                    project=str(eval_root),
                    name=eval_name,
                    exist_ok=True,
                    verbose=False,
                )
                map50, map50_95, precision, recall = _metrics_dict(result)
                rows.append({
                    "consistency_weight": weight,
                    "seed": seed,
                    "checkpoint": str(best),
                    "map50": map50,
                    "map50_95": map50_95,
                    "precision": precision,
                    "recall": recall,
                })
                with (output_dir / "ablation_results_raw.csv").open("w", encoding="utf-8", newline="") as handle:
                    fields = ["consistency_weight", "seed", "checkpoint", "map50", "map50_95", "precision", "recall"]
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                _aggregate(rows, output_dir / "ablation_results.csv")
                if progress is not None and task is not None:
                    progress.update(task, advance=1, description=f"tested weight={weight:g}, seed={seed}")
    finally:
        if progress is not None:
            progress.stop()

    print(f"\nAblation complete. Summary: {output_dir / 'ablation_results.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




