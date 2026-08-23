"""Run conservative COCO-pretrained YOLO11l EMA adaptation, then supervised fine-tuning.

The adaptation stage uses only the 512/256 NAIP JPEG tile directory, a small
learning rate, and optional early-layer freezing to protect COCO features.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make the project package importable when this file is launched directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultralytics import YOLO

from poultry_geo.ema_adapt import EMADomainAdaptOBBTrainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "COCO-pretrained YOLO11 OBB domain adaptation: EMA teacher/student feature "
            "consistency on unlabeled NAIP, followed by native supervised fine-tuning."
        )
    )
    parser.add_argument("--data", default=str(PROJECT_ROOT / "configs" / "poultry_80_10_10.yaml"))
    parser.add_argument("--unlabeled-root", default=r"E:\NAIP_Dataset_PH\MS_tiles_512_256_jpg")
    parser.add_argument("--unlabeled-catalog", default=None)
    parser.add_argument("--model", default="yolo11l-obb.pt", help="COCO-pretrained Ultralytics OBB checkpoint.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "yolo11l_ema_adapt"))
    parser.add_argument("--adapt-epochs", type=int, default=60)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--unlabeled-batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--unlabeled-imgsz", type=int, default=448)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--unlabeled-workers", type=int, default=2)
    parser.add_argument("--unlabeled-limit", type=int, default=250000, help="Maximum deterministic NAIP tiles kept in memory (0 disables the limit).")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of labeled training data (use a small value only for smoke tests).")
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--consistency-rampup", type=float, default=10.0)
    parser.add_argument("--adapt-lr0", type=float, default=1e-5, help="Initial stage-1 learning rate; conservative to protect COCO weights.")
    parser.add_argument("--finetune-lr0", type=float, default=3e-4, help="Initial stage-2 learning rate for a smooth handoff.")
    parser.add_argument("--freeze-backbone", type=int, default=10, help="Freeze the first N YOLO modules during stage 1; 0 disables freezing.")
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument("--resume", action="store_true", help="Resume stage 1 from output-dir/weights/last.pt.")
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def _checkpoint_pretrained_argument(checkpoint: Path) -> str:
    """Return an explicit checkpoint path for Ultralytics stage-2 loading.

    Passing ``pretrained=False`` to ``YOLO(checkpoint).train`` causes Ultralytics
    to build a fresh model and drop the already-loaded checkpoint.  Supplying the
    concrete checkpoint path forces the trainer to load the adapted weights.
    """
    resolved = checkpoint.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {resolved}")
    return str(resolved)


def _common_validation(args: argparse.Namespace) -> None:
    if args.adapt_epochs <= 0 or args.finetune_epochs <= 0:
        raise ValueError("adapt-epochs and finetune-epochs must be positive.")
    if args.imgsz % 32 or args.unlabeled_imgsz % 32:
        raise ValueError("imgsz and unlabeled-imgsz must be divisible by 32.")
    if args.batch <= 0 or args.unlabeled_batch <= 0:
        raise ValueError("batch and unlabeled-batch must be positive.")
    if args.unlabeled_workers < 0 or args.unlabeled_limit < 0 or args.prefetch_factor <= 0:
        raise ValueError("unlabeled-workers and unlabeled-limit cannot be negative; prefetch-factor must be positive.")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Labeled dataset YAML not found: {args.data}")
    if not Path(args.unlabeled_root).exists():
        raise FileNotFoundError(f"Unlabeled NAIP root not found: {args.unlabeled_root}")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("ema-decay must be between 0 and 1.")
    if args.consistency_weight < 0:
        raise ValueError("consistency-weight cannot be negative.")
    if args.adapt_lr0 <= 0 or args.finetune_lr0 <= 0:
        raise ValueError("adapt-lr0 and finetune-lr0 must be positive.")
    if args.freeze_backbone < 0:
        raise ValueError("freeze-backbone cannot be negative.")
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")


def main() -> int:
    args = _parse_args()
    _common_validation(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    stage1_dir = output_dir
    stage2_dir = output_dir.parent / f"{output_dir.name}_finetune"
    stage1_dir.mkdir(parents=True, exist_ok=True)

    last = stage1_dir / "weights" / "last.pt"
    if args.resume:
        if not last.exists():
            raise FileNotFoundError(f"--resume requested but stage-1 checkpoint is missing: {last}")
        adaptation_model = YOLO(str(last))
        resume = True
        print(f"Resuming EMA adaptation from {last}")
    else:
        adaptation_model = YOLO(args.model)
        resume = False
        print(f"Starting EMA adaptation from COCO checkpoint: {args.model}")
    print(f"Unlabeled tile root: {Path(args.unlabeled_root).expanduser().resolve()}")
    print(f"Conservative settings: adapt_lr0={args.adapt_lr0:g}, freeze_backbone={args.freeze_backbone}, finetune_lr0={args.finetune_lr0:g}")

    adaptation_model.train(
        trainer=EMADomainAdaptOBBTrainer,
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
        freeze=args.freeze_backbone,
        pretrained=True,
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
        consistency_weight=args.consistency_weight,
        consistency_rampup=args.consistency_rampup,
        ema_decay=args.ema_decay,
    )

    best = stage1_dir / "weights" / "best.pt"
    if not best.exists():
        best = last
    if not best.exists():
        raise FileNotFoundError(f"EMA adaptation finished without a checkpoint: {stage1_dir / 'weights'}")
    print(f"EMA adaptation checkpoint: {best}")
    if args.stage1_only:
        return 0

    stage2_dir.mkdir(parents=True, exist_ok=True)
    stage1_checkpoint = _checkpoint_pretrained_argument(best)
    print(f"Starting supervised fine-tuning from {stage1_checkpoint}")
    print("Preserving EMA stage-1 weights during stage-2 fine-tuning.")
    finetune_model = YOLO(stage1_checkpoint)
    finetune_model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.finetune_epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        optimizer="AdamW",
        lr0=args.finetune_lr0,
        lrf=0.01,
        warmup_epochs=5.0,
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
        freeze=0,
        # Pass the concrete checkpoint path: pretrained=False would reset this model to random weights.
        pretrained=stage1_checkpoint,
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
        raise FileNotFoundError(f"Supervised fine-tuning finished without a checkpoint: {final_checkpoint}")
    handoff_metadata = {
        "version": 1,
        "source_checkpoint": stage1_checkpoint,
        "pretrained_argument": stage1_checkpoint,
        "weights_preserved": True,
        "unlabeled_root": str(Path(args.unlabeled_root).expanduser().resolve()),
        "adapt_lr0": args.adapt_lr0,
        "finetune_lr0": args.finetune_lr0,
        "freeze_backbone": args.freeze_backbone,
        "stage2_checkpoint": str(final_checkpoint.resolve()),
    }
    (stage2_dir / "ema_handoff.json").write_text(
        json.dumps(handoff_metadata, indent=2), encoding="utf-8"
    )
    print(f"Final supervised checkpoint: {final_checkpoint}")
    print(f"EMA handoff verified: {stage2_dir / 'ema_handoff.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
