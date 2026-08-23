"""
downstream_compare.py
=====================
Trains and evaluates two YOLO11l-OBB models on the poultry house detection task:

  Run A  â€” Official Ultralytics ImageNet weights  (baseline)
  Run B  â€” Our SparK-pretrained backbone           (ours)

Both runs are trained identically (same hyperparameters, same data splits) so
any difference in final mAP is directly attributable to the pretraining strategy.

Usage
-----
  python scripts/downstream_compare.py

Output
------
  runs/downstream/
    imagenet/          -- Ultralytics YOLO run dir (weights, results.csv, â€¦)
    spark_pretrained/  -- SparK-pretrained run dir
  runs/downstream/comparison_results.csv
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import torch

# â”€â”€ Project root (two levels up from this script) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SPARK_BACKBONE_PT = (
    ROOT
    / "trained_models"
    / "spark-search_new"
    / "5d1be34f2942_yolo11l_fp16"
    / "backbone_dense_best.pt"
)
DATA_YAML = ROOT / "configs" / "poultry_obb.yaml"
OUTPUT_DIR = ROOT / "runs" / "downstream"

# â”€â”€ Shared fine-tuning hyperparameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
COMMON_TRAIN_ARGS: dict = dict(
    data=str(DATA_YAML),
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,           # RTX 4090
    optimizer="AdamW",
    lr0=1e-3,
    lrf=0.01,           # final_lr = lr0 * lrf  (cosine decay)
    warmup_epochs=3,
    cos_lr=True,
    weight_decay=0.0005,
    momentum=0.937,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,       # rotation augmentation (relevant for OBB)
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    close_mosaic=10,
    amp=True,           # automatic mixed precision
    workers=8,
    verbose=True,
    exist_ok=True,
    plots=True,
)


def _load_spark_weights_into_yolo(model, backbone_pt: Path) -> None:
    raise RuntimeError(
        "SparK-to-YOLO transfer is disabled: the supported SparK encoder is ResNet50, "
        "and this project has no verified architecture-preserving YOLO adapter."
    )
def _extract_metrics(run_dir: Path) -> dict:
    """Read the final-epoch metrics from Ultralytics results.csv."""
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return {}
    with open(results_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    last = rows[-1]
    # Strip whitespace from keys produced by Ultralytics
    return {k.strip(): v.strip() for k, v in last.items()}


def run_experiment(name: str, use_spark: bool) -> dict:
    """Train one model variant and return its final metrics."""
    from ultralytics import YOLO

    save_dir = OUTPUT_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Starting experiment: {name}")
    print(f"  SparK backbone : {use_spark}")
    print(f"  Save dir       : {save_dir}")
    print(f"{'='*70}\n")

    # Ultralytics always initialises from the official weights when you pass
    # "yolo11l-obb.pt".  For the SparK variant we load official weights first
    # (to get the detection heads), then overwrite only the backbone layers.
    model = YOLO("yolo11l-obb.pt")

    if use_spark:
        _load_spark_weights_into_yolo(model, SPARK_BACKBONE_PT)

    model.train(
        project=str(OUTPUT_DIR),
        name=name,
        **COMMON_TRAIN_ARGS,
    )

    metrics = _extract_metrics(OUTPUT_DIR / name)
    metrics["experiment"] = name
    metrics["spark_pretrained"] = str(use_spark)
    return metrics


def main() -> None:
    if not SPARK_BACKBONE_PT.exists():
        print(f"ERROR: SparK backbone not found at:\n  {SPARK_BACKBONE_PT}")
        sys.exit(1)

    if not DATA_YAML.exists():
        print(f"ERROR: Dataset YAML not found at:\n  {DATA_YAML}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    # â”€â”€ Run A: Official ImageNet baseline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    metrics_a = run_experiment("imagenet_baseline", use_spark=False)
    results.append(metrics_a)

    # â”€â”€ Run B: SparK pretrained â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    metrics_b = run_experiment("spark_pretrained", use_spark=True)
    results.append(metrics_b)

    # â”€â”€ Print comparison table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n" + "=" * 70)
    print("  COMPARISON RESULTS")
    print("=" * 70)

    # Key OBB metrics reported by Ultralytics
    metric_keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    header = f"{'Experiment':<25} " + "  ".join(f"{k:<22}" for k in metric_keys)
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r.get('experiment',''):<25} " + "  ".join(
            f"{r.get(k,'N/A'):<22}" for k in metric_keys
        )
        print(row)

    # â”€â”€ Save comparison CSV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    out_csv = OUTPUT_DIR / "comparison_results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    print(f"\nResults saved to: {out_csv}")

    # â”€â”€ Quick delta â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        map50_a = float(metrics_a.get("metrics/mAP50(B)", "nan"))
        map50_b = float(metrics_b.get("metrics/mAP50(B)", "nan"))
        delta = map50_b - map50_a
        winner = "SparK-pretrained" if delta > 0 else "ImageNet baseline"
        print(f"\n  mAP50 delta (SparK - ImageNet): {delta:+.4f}")
        print(f"  Winner: {winner}")
    except (ValueError, TypeError):
        pass


if __name__ == "__main__":
    main()
