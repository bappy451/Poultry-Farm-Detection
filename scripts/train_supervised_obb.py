"""Train a reproducible supervised YOLO-OBB baseline/burn-in checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from scripts.checkpoint_retry import RetryingOBBTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--init-model", default="yolo11l-obb.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch <= 0 or args.workers < 0:
        raise ValueError("epochs and batch must be positive; workers cannot be negative")
    if args.imgsz % 32:
        raise ValueError("imgsz must be divisible by 32")
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    data = resolve_project_path(args.data)
    output_dir = resolve_project_path(args.output_dir)
    init_model = resolve_project_path(args.init_model)
    if not data.is_file():
        raise FileNotFoundError(data)
    if not init_model.is_file():
        raise FileNotFoundError(init_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    last = output_dir / "weights" / "last.pt"
    if args.resume:
        if not last.is_file():
            raise FileNotFoundError(f"--resume requested but missing {last}")
        model = YOLO(str(last))
    else:
        model = YOLO(str(init_model))
    model.train(
        trainer=RetryingOBBTrainer,
        data=str(data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        fraction=args.fraction,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.01,
        weight_decay=args.weight_decay,
        warmup_epochs=3.0,
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
        pretrained=str(init_model),
        resume=args.resume,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        plots=True,
        val=True,
        patience=max(30, min(60, args.epochs // 2)),
    )
    best = output_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Supervised training finished without {best}")
    marker = {
        "version": 1,
        "pipeline": "supervised_yolo_obb_burn_in",
        "data": str(data),
        "init_model": str(init_model),
        "best_checkpoint": str(best.resolve()),
        "epochs": args.epochs,
        "seed": args.seed,
        "fraction": args.fraction,
        "test_used_for_selection": False,
    }
    (output_dir / "supervised_baseline.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    print(f"Supervised baseline checkpoint: {best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
