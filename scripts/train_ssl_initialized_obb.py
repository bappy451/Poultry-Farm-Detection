"""Fine-tune YOLO-OBB after anchored self-supervised MS pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for location in (ROOT, SRC):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from ultralytics import YOLO

from scripts.checkpoint_retry import RetryingOBBTrainer
from scripts.ssl_anchored_dino_yolo_obb import load_artifact, load_artifact_into_detector


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_completed_ssl_artifact(artifact_path: Path) -> None:
    marker_path = artifact_path.parent / "pretraining_complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(
            f"Refusing to fine-tune from a partial SSL artifact without {marker_path}."
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid SSL completion marker: {marker_path}") from exc
    if (
        marker.get("complete") is not True
        or Path(str(marker.get("best_artifact", ""))).resolve() != artifact_path.resolve()
        or marker.get("best_artifact_sha256") != _sha256(artifact_path)
        or marker.get("final_test_used") is not False
        or marker.get("validation_sampling_contract") != "source_group_round_robin_v1"
        or marker.get("optimizer_contract") != "adamw_kernel_matrix_decay_bias_norm_no_decay_v1"
    ):
        raise ValueError(f"SSL completion marker does not verify artifact: {artifact_path}")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--init-model", default="yolo11l-obb.pt")
    parser.add_argument("--ssl-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch <= 0 or args.workers < 0:
        raise ValueError("epochs and batch must be positive; workers cannot be negative.")
    if args.imgsz <= 0 or args.imgsz % 32:
        raise ValueError("imgsz must be a positive multiple of 32.")

    data, init_model = _resolve(args.data), _resolve(args.init_model)
    artifact_path, output_dir = _resolve(args.ssl_artifact), _resolve(args.output_dir)
    for path in (data, init_model, artifact_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    _require_completed_ssl_artifact(artifact_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    last = output_dir / "weights" / "last.pt"
    if args.resume:
        if not last.is_file():
            raise FileNotFoundError(f"--resume requested but missing {last}")
        model = YOLO(str(last))
        print(f"Resuming SSL-initialized AL fine-tuning from {last}.", flush=True)
    else:
        artifact = load_artifact(artifact_path)
        artifact_init = Path(str(artifact["init_model"])).resolve()
        if artifact_init != init_model:
            raise ValueError(
                f"SSL artifact was anchored to {artifact_init}, but --init-model is {init_model}."
            )
        model = YOLO(str(init_model))
        loaded = load_artifact_into_detector(model, artifact)
        print(f"Loaded {loaded} exact anchored-SSL backbone tensors; pretrained OBB neck/head retained.", flush=True)
    model.train(
        trainer=RetryingOBBTrainer,
        data=str(data), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        workers=args.workers, device=args.device, seed=args.seed,
        optimizer="AdamW", lr0=args.lr0, lrf=0.01, weight_decay=args.weight_decay,
        warmup_epochs=3.0, cos_lr=True, close_mosaic=10,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=15.0,
        flipud=0.5, fliplr=0.5, mosaic=1.0, amp=True,
        # Keep this boolean True. Ultralytics then passes the current in-memory
        # model (including the injected SSL backbone) into the trainer. Passing
        # the original checkpoint path here would silently overwrite SSL.
        pretrained=True, project=str(output_dir.parent), name=output_dir.name,
        exist_ok=True, plots=True, val=True, resume=args.resume,
        patience=max(30, min(60, args.epochs // 2)),
    )
    best = output_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"SSL-initialized fine-tuning completed without {best}.")
    marker = {
        "version": 1,
        "pipeline": "anchored_dinov2_ms_ssl_then_supervised_al_obb",
        "data": str(data), "init_model": str(init_model),
        "ssl_artifact": str(artifact_path), "best_checkpoint": str(best.resolve()),
        "epochs": args.epochs, "seed": args.seed,
        "resumed": args.resume,
        "test_used_for_selection": False,
    }
    _atomic_json(output_dir / "ssl_finetune.json", marker)
    print(f"SSL-initialized AL checkpoint: {best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
