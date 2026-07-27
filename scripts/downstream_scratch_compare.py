"""
downstream_scratch_compare.py
=============================
Trains and evaluates three YOLO11l-OBB models on the poultry house detection task
for a limited training budget (e.g., 50 epochs) to demonstrate data/compute efficiency:

  Run A  — Official Ultralytics ImageNet weights               (baseline)
  Run B  — Random weight initialization                        (scratch ablation)
  Run C  — Our SparK-from-scratch backbone with LP-FT + LLRD   (ours)

LP-FT (Linear Probing followed by End-to-End Fine-Tuning) + LLRD:
-----------------------------------------------------------------
For the SparK Trained-from-Scratch backbone:
  * Stage 1: Linear Probing (LP)
      Backbone layers (0 to 10) are frozen for `lp_epochs` (default: 5).
      Only detection neck and OBB heads are trained.
  * Stage 2: End-to-End Fine-Tuning (FT) with Layer-Wise LR Decay (LLRD)
      Backbone layers are unfrozen at epoch `lp_epochs`. All layers are fine-tuned
      using geometric decay (alpha=0.75) across backbone depths.

Usage
-----
  python scripts/downstream_scratch_compare.py --config configs/downstream_scratch_compare.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml

# ── Project root (two levels up from this script) ────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── Paths & Default Hyperparameters ──────────────────────────────────────────
DEFAULT_SPARK_BACKBONE_PT = (
    ROOT
    / "trained_models"
    / "spark-scratch"
    / "af8e64a7ed89_yolo11l_fp16"
    / "backbone_dense_ep299.pt"
)
DEFAULT_DATA_YAML = ROOT / "configs" / "poultry_obb.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "downstream_scratch"

# Default 50-epoch budget for fair efficiency evaluation
DEFAULT_TRAIN_ARGS: dict = dict(
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    optimizer="AdamW",
    lr0=1e-3,
    lrf=0.01,
    warmup_epochs=3,
    cos_lr=True,
    weight_decay=0.0005,
    momentum=0.937,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    close_mosaic=10,
    amp=True,
    workers=8,
    verbose=True,
    exist_ok=True,
    plots=True,
)


def _load_spark_weights_into_yolo(model, backbone_pt: Path) -> None:
    """
    Surgically inject SparK-from-scratch backbone weights into YOLO11l.
    Loads with strict=False so detection neck/heads keep their standard initialisation.
    """
    print(f"\n[SparK] Loading backbone from: {backbone_pt}")
    backbone_state = torch.load(str(backbone_pt), map_location="cpu", weights_only=True)

    yolo_state = model.model.state_dict()
    matched, skipped = 0, 0
    patch: dict = {}
    for key, tensor in backbone_state.items():
        if key in yolo_state and yolo_state[key].shape == tensor.shape:
            patch[key] = tensor
            matched += 1
        else:
            skipped += 1

    yolo_state.update(patch)
    model.model.load_state_dict(yolo_state, strict=False)
    print(f"[SparK] Injected {matched} tensors  |  Skipped {skipped} tensors")


def _extract_metrics(run_dir: Path) -> dict:
    """Extract final-epoch OBB detection metrics from results.csv."""
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return {}
    with open(results_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    last = rows[-1]
    return {k.strip(): v.strip() for k, v in last.items()}


def run_experiment(
    name: str,
    use_spark: bool,
    is_random: bool,
    use_lpft_llrd: bool,
    train_args: dict,
    lp_epochs: int = 5,
    layer_decay: float = 0.75,
    spark_backbone_pt: Path = DEFAULT_SPARK_BACKBONE_PT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Train one model variant and return its final metrics."""
    from ultralytics import YOLO
    from downstream_finetune.lpft_llrd_trainer import LPFT_LLRD_OBBTrainer

    save_dir = output_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Starting experiment : {name}")
    print(f"  SparK backbone      : {use_spark}")
    print(f"  Random Init         : {is_random}")
    print(f"  LP-FT + LLRD        : {use_lpft_llrd} (lp_epochs={lp_epochs}, alpha={layer_decay})")
    print(f"  Save dir            : {save_dir}")
    print(f"{'='*70}\n")

    if is_random:
        # Load architecture YAML for random weight initialization from scratch
        model = YOLO("yolo11l-obb.yaml")
    else:
        # Load official weights for detection neck and OBB heads
        model = YOLO("yolo11l-obb.pt")
        if use_spark:
            _load_spark_weights_into_yolo(model, spark_backbone_pt)

    # Prepare training kwargs
    kwargs = dict(train_args)
    kwargs["project"] = str(output_dir)
    kwargs["name"] = name

    if use_lpft_llrd:
        # Configure class attributes on custom trainer
        LPFT_LLRD_OBBTrainer.lp_epochs = lp_epochs
        LPFT_LLRD_OBBTrainer.layer_decay = layer_decay
        kwargs["trainer"] = LPFT_LLRD_OBBTrainer

    model.train(**kwargs)

    metrics = _extract_metrics(save_dir)
    metrics["experiment"] = name
    metrics["spark_pretrained"] = str(use_spark)
    metrics["random_init"] = str(is_random)
    metrics["lpft_llrd"] = str(use_lpft_llrd)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SparK Trained-from-Scratch vs ImageNet vs Random Init for YOLO11l-OBB."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "downstream_scratch_compare.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config file if available, otherwise fallback to defaults
    config: dict = {}
    if args.config and args.config.exists():
        print(f"[Config] Loading experiment configuration from: {args.config}")
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        print(f"[Config] Config file not found at {args.config}. Using default configuration.")

    paths_cfg = config.get("paths", {})
    spark_pt = Path(paths_cfg.get("spark_backbone_pt", DEFAULT_SPARK_BACKBONE_PT))
    if not spark_pt.is_absolute():
        spark_pt = ROOT / spark_pt

    data_yaml = Path(paths_cfg.get("data_yaml", DEFAULT_DATA_YAML))
    if not data_yaml.is_absolute():
        data_yaml = ROOT / data_yaml

    output_dir = Path(paths_cfg.get("output_dir", DEFAULT_OUTPUT_DIR))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    if not spark_pt.exists():
        print(f"ERROR: SparK backbone checkpoint not found at:\n  {spark_pt}")
        sys.exit(1)

    if not data_yaml.exists():
        print(f"ERROR: Dataset YAML not found at:\n  {data_yaml}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build training arguments
    train_args = dict(DEFAULT_TRAIN_ARGS)
    train_args.update(config.get("training", {}))
    train_args["data"] = str(data_yaml)

    lpft_cfg = config.get("lpft_llrd", {})
    lp_epochs = int(lpft_cfg.get("lp_epochs", 5))
    layer_decay = float(lpft_cfg.get("layer_decay", 0.75))

    # Define 3 experiments
    default_experiments = [
        dict(name="imagenet_baseline", use_spark=False, is_random=False, use_lpft_llrd=False),
        dict(name="random_init", use_spark=False, is_random=True, use_lpft_llrd=False),
        dict(name="spark_scratch_lpft", use_spark=True, is_random=False, use_lpft_llrd=True),
    ]
    experiments = config.get("experiments", default_experiments)

    results: list[dict] = []
    metrics_by_name: dict[str, dict] = {}

    for exp in experiments:
        exp_name = exp["name"]
        m = run_experiment(
            name=exp_name,
            use_spark=bool(exp.get("use_spark", False)),
            is_random=bool(exp.get("is_random", False)),
            use_lpft_llrd=bool(exp.get("use_lpft_llrd", False)),
            train_args=train_args,
            lp_epochs=lp_epochs,
            layer_decay=layer_decay,
            spark_backbone_pt=spark_pt,
            output_dir=output_dir,
        )
        results.append(m)
        metrics_by_name[exp_name] = m

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  THREE-WAY COMPARISON RESULTS ({train_args.get('epochs', 50)} Epochs)")
    print("=" * 80)

    metric_keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    header = f"{'Experiment':<24} " + "  ".join(f"{k:<18}" for k in metric_keys)
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r.get('experiment',''):<24} " + "  ".join(
            f"{float(r.get(k, 0.0)):<18.4f}" if r.get(k) and r.get(k) != "N/A" else f"{'N/A':<18}"
            for k in metric_keys
        )
        print(row)

    out_csv = output_dir / "comparison_results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    print(f"\nResults saved to: {out_csv}")

    # ── Quick Delta Analysis ──────────────────────────────────────────────────
    try:
        map50_imgnet = float(metrics_by_name.get("imagenet_baseline", {}).get("metrics/mAP50(B)", "nan"))
        map50_rand = float(metrics_by_name.get("random_init", {}).get("metrics/mAP50(B)", "nan"))
        map50_spark = float(metrics_by_name.get("spark_scratch_lpft", {}).get("metrics/mAP50(B)", "nan"))

        print("\n" + "-" * 70)
        print("  mAP50 DELTA ANALYSIS:")
        print("-" * 70)
        print(f"  SparK (LP-FT + LLRD)  vs  ImageNet Baseline   : {map50_spark - map50_imgnet:+.4f}")
        print(f"  SparK (LP-FT + LLRD)  vs  Random Init         : {map50_spark - map50_rand:+.4f}")
        print(f"  ImageNet Baseline     vs  Random Init         : {map50_imgnet - map50_rand:+.4f}")

        best_score = max(map50_imgnet, map50_rand, map50_spark)
        if best_score == map50_spark:
            winner = "SparK-from-Scratch (with LP-FT + LLRD)"
        elif best_score == map50_imgnet:
            winner = "ImageNet Baseline"
        else:
            winner = "Random Initialization"
        print(f"\n  WINNER: {winner} (mAP50 = {best_score:.4f})")
    except (ValueError, TypeError, KeyError):
        pass


if __name__ == "__main__":
    main()
