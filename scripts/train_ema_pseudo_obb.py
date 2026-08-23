"""Train detector-level EMA pseudo-OBB adaptation, then supervised fine-tune."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# Windows DataLoader workers use spawn and re-import this module.  Keep the
# project root on sys.path and import the trainer through its package name so
# workers resolve it exactly like the parent process.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from scripts.checkpoint_retry import RetryingOBBTrainer
from scripts.ema_pseudo_obb import PseudoLabelEMADomainAdaptOBBTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(PROJECT_ROOT / "configs" / "poultry_clean_label_subsets" / "poultry_clean_10percent.yaml"))
    parser.add_argument("--unlabeled-root", default=r"E:\NAIP_Dataset_PH\MS_tiles_512_256_jpg")
    parser.add_argument("--unlabeled-catalog", default=None)
    parser.add_argument("--model", default="yolo11l-obb.pt")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Supervised poultry checkpoint used to initialize both the student and EMA teacher.",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "ema_pseudo_obb_10pct"))
    parser.add_argument("--adapt-epochs", type=int, default=60)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--unlabeled-batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--unlabeled-imgsz", type=int, default=448)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--unlabeled-workers", type=int, default=2)
    parser.add_argument("--unlabeled-limit", type=int, default=250000)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--pseudo-obb-weight", type=float, default=0.05)
    parser.add_argument("--pseudo-thresholds", default="0.90", help="One threshold or comma-separated threshold per class.")
    parser.add_argument("--pseudo-iou-threshold", type=float, default=0.5)
    parser.add_argument("--pseudo-class-margin", type=float, default=0.05, help="Reject ambiguous teacher classes when top-1 minus top-2 score is below this margin.")
    parser.add_argument("--pseudo-max-det", type=int, default=100)
    parser.add_argument("--pseudo-warmup-epochs", type=float, default=5.0)
    parser.add_argument("--pseudo-rampup-epochs", type=float, default=10.0)
    parser.add_argument("--feature-consistency-weight", type=float, default=0.0)
    parser.add_argument("--feature-rampup-epochs", type=float, default=10.0)
    parser.add_argument("--foreground-feature-gain", type=float, default=3.0)
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument("--adapt-lr0", type=float, default=3e-4)
    parser.add_argument("--finetune-lr0", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if args.adapt_epochs <= 0 or args.finetune_epochs <= 0:
        raise ValueError("adapt-epochs and finetune-epochs must be positive")
    if args.imgsz % 32 or args.unlabeled_imgsz % 32:
        raise ValueError("imgsz and unlabeled-imgsz must be divisible by 32")
    if args.batch <= 0 or args.unlabeled_batch <= 0:
        raise ValueError("batch sizes must be positive")
    if args.unlabeled_workers < 0 or args.unlabeled_limit < 0 or args.prefetch_factor <= 0:
        raise ValueError("loader settings are invalid")
    if not Path(args.data).expanduser().is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if not Path(args.unlabeled_root).expanduser().exists():
        raise FileNotFoundError(f"Unlabeled root not found: {args.unlabeled_root}")
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if args.pseudo_obb_weight < 0 or args.feature_consistency_weight < 0:
        raise ValueError("loss weights cannot be negative")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("ema-decay must be in (0, 1)")
    thresholds = [float(x.strip()) for x in args.pseudo_thresholds.split(",") if x.strip()]
    if not thresholds or any(not 0.0 < x < 1.0 for x in thresholds):
        raise ValueError("pseudo-thresholds must be in (0, 1)")
    if not 0.0 <= args.pseudo_iou_threshold <= 1.0:
        raise ValueError("pseudo-iou-threshold must be in [0, 1]")
    if not 0.0 <= args.pseudo_class_margin < 1.0:
        raise ValueError("pseudo-class-margin must be in [0, 1)")
    if args.pseudo_max_det <= 0:
        raise ValueError("pseudo-max-det must be positive")

    if args.adapt_lr0 <= 0 or args.finetune_lr0 <= 0:
        raise ValueError("adapt-lr0 and finetune-lr0 must be positive")


def main() -> int:
    args = parse_args()
    validate(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    stage1_dir = output_dir
    stage2_dir = output_dir.parent / f"{output_dir.name}_finetune"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    last = stage1_dir / "weights" / "last.pt"
    initial_checkpoint = Path(args.init_checkpoint or args.model).expanduser()
    if not initial_checkpoint.is_absolute():
        initial_checkpoint = (PROJECT_ROOT / initial_checkpoint).resolve()
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(f"Initialization checkpoint not found: {initial_checkpoint}")
    if args.resume:
        if not last.exists():
            raise FileNotFoundError(f"--resume requested but missing {last}")
        adaptation_model = YOLO(str(last))
        resume = True
    else:
        adaptation_model = YOLO(str(initial_checkpoint))
        resume = False

    adaptation_model.train(
        trainer=PseudoLabelEMADomainAdaptOBBTrainer,
        data=str(Path(args.data).resolve()),
        epochs=args.adapt_epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        optimizer="AdamW",
        lr0=args.adapt_lr0,
        lrf=0.01,
        warmup_epochs=5.0,
        weight_decay=0.01,
        cos_lr=True,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        amp=True,
        pretrained=str(initial_checkpoint),
        resume=resume,
        seed=args.seed,
        fraction=args.fraction,
        project=str(stage1_dir.parent),
        name=stage1_dir.name,
        exist_ok=True,
        plots=False,
        val=True,
        patience=max(20, min(50, args.adapt_epochs // 2)),
        unlabeled_root=str(Path(args.unlabeled_root).resolve()),
        unlabeled_catalog=args.unlabeled_catalog,
        unlabeled_batch=args.unlabeled_batch,
        unlabeled_imgsz=args.unlabeled_imgsz,
        unlabeled_workers=args.unlabeled_workers,
        unlabeled_limit=(args.unlabeled_limit or None),
        prefetch_factor=args.prefetch_factor,
        pseudo_obb_weight=args.pseudo_obb_weight,
        pseudo_thresholds=args.pseudo_thresholds,
        pseudo_iou_threshold=args.pseudo_iou_threshold,
        pseudo_class_margin=args.pseudo_class_margin,
        pseudo_max_det=args.pseudo_max_det,
        pseudo_warmup_epochs=args.pseudo_warmup_epochs,
        pseudo_rampup_epochs=args.pseudo_rampup_epochs,
        feature_consistency_weight=args.feature_consistency_weight,
        feature_rampup_epochs=args.feature_rampup_epochs,
        foreground_feature_gain=args.foreground_feature_gain,
        ema_decay=args.ema_decay,
    )

    # The adaptation stage is a fixed-length (or early-stopped) SSL procedure.
    # Its validation-best checkpoint can occur before pseudo supervision is
    # meaningfully active (for example during pseudo-label warm-up). Handing
    # that checkpoint to stage 2 would silently turn a pseudo-label ablation
    # into an almost identical supervised fine-tune. Use the final adaptation
    # state instead; stage 2 remains the sole place where the downstream model
    # is selected by validation mAP50-95.
    adaptation_checkpoint = last
    if not adaptation_checkpoint.exists():
        adaptation_checkpoint = stage1_dir / "weights" / "best.pt"
    if not adaptation_checkpoint.exists():
        raise FileNotFoundError(f"Adaptation finished without checkpoint: {stage1_dir / 'weights'}")
    if args.stage1_only:
        return 0

    stage2_dir.mkdir(parents=True, exist_ok=True)
    final_model = YOLO(str(adaptation_checkpoint.resolve()))
    final_model.train(
        trainer=RetryingOBBTrainer,
        data=str(Path(args.data).resolve()),
        epochs=args.finetune_epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        optimizer="AdamW",
        lr0=args.finetune_lr0,
        lrf=0.01,
        warmup_epochs=3.0,
        weight_decay=5e-4,
        cos_lr=True,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        amp=True,
        pretrained=str(adaptation_checkpoint.resolve()),
        seed=args.seed,
        fraction=args.fraction,
        project=str(stage2_dir.parent),
        name=stage2_dir.name,
        exist_ok=True,
        plots=True,
        val=True,
        patience=max(30, min(60, args.finetune_epochs // 2)),
    )
    final_checkpoint = stage2_dir / "weights" / "best.pt"
    if not final_checkpoint.exists():
        raise FileNotFoundError(f"Fine-tuning finished without {final_checkpoint}")
    marker = {
        "version": 1,
        "pipeline": "ema_pseudo_obb",
        "source_checkpoint": str(adaptation_checkpoint.resolve()),
        "stage1_handoff_policy": "last.pt (final adaptation state; stage-2 best.pt is validation-selected)",
        "initial_checkpoint": str(initial_checkpoint),
        "stage2_checkpoint": str(final_checkpoint.resolve()),
        "pseudo_obb_weight": args.pseudo_obb_weight,
        "pseudo_thresholds": args.pseudo_thresholds,
        "pseudo_class_margin": args.pseudo_class_margin,
        "feature_consistency_weight": args.feature_consistency_weight,
        "adapt_lr0": args.adapt_lr0,
        "finetune_lr0": args.finetune_lr0,
        "test_used_for_selection": False,
    }
    (stage2_dir / "pseudo_obb_handoff.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    print(f"Final pseudo-OBB checkpoint: {final_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

