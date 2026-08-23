"""Measure pseudo-label coverage, calibration, and oriented localization on validation only.

The script evaluates frozen teacher checkpoints on a labeled validation split.
It emits classwise threshold curves and selected class-specific thresholds for
each seed.  It never opens a test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_pseudo_thresholds import _label_path, _polygon_iou, _read_ground_truth, choose_threshold
from scripts.run_ema_benchmark import _dataset_image_paths


def threshold_grid(values: str) -> list[float]:
    result = [float(item.strip()) for item in values.split(",") if item.strip()]
    if not result or any(not 0.0 < item < 1.0 for item in result):
        raise ValueError("--thresholds must be comma-separated values in (0, 1).")
    return sorted(set(result))


def match_with_iou(
    predictions: Iterable[tuple[int, float, np.ndarray]],
    targets: Iterable[tuple[int, np.ndarray]],
    *,
    iou_threshold: float,
) -> dict[int, list[tuple[float, bool, float]]]:
    """Confidence-greedy, classwise one-to-one OBB matching with rotated IoU."""
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for class_id, polygon in targets:
        grouped[int(class_id)].append(polygon)
    matched: dict[int, set[int]] = defaultdict(set)
    records: dict[int, list[tuple[float, bool, float]]] = defaultdict(list)
    for class_id, confidence, polygon in sorted(predictions, key=lambda item: float(item[1]), reverse=True):
        class_id = int(class_id)
        best_iou, best_index = 0.0, -1
        for index, target in enumerate(grouped[class_id]):
            if index in matched[class_id]:
                continue
            iou = _polygon_iou(polygon, target)
            if iou > best_iou:
                best_iou, best_index = iou, index
        is_tp = best_index >= 0 and best_iou >= iou_threshold
        if is_tp:
            matched[class_id].add(best_index)
        records[class_id].append((float(confidence), is_tp, float(best_iou)))
    return records


def score_curve(records: Iterable[tuple[float, bool, float]], target_count: int, threshold: float) -> dict[str, float | int]:
    selected = [(score, is_tp, iou) for score, is_tp, iou in records if score >= threshold]
    true_positives = sum(int(is_tp) for _, is_tp, _ in selected)
    precision = true_positives / len(selected) if selected else 0.0
    recall = true_positives / target_count if target_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    tp_ious = [iou for _, is_tp, iou in selected if is_tp]
    return {
        "threshold": threshold,
        "predictions": len(selected),
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accepted_boxes_per_image": 0.0,  # populated by caller using its image count
        "mean_matched_rotated_iou": float(sum(tp_ious) / len(tp_ious)) if tp_ious else 0.0,
    }


def parse_names(data: Path) -> dict[int, str]:
    payload = yaml.safe_load(data.read_text(encoding="utf-8")) or {}
    names = payload.get("names", {})
    if isinstance(names, list):
        names = {index: item for index, item in enumerate(names)}
    return {int(key): str(value) for key, value in dict(names).items()}


def diagnose_checkpoint(
    checkpoint: Path,
    data: Path,
    *,
    split: str,
    thresholds: list[float],
    precision_target: float,
    min_predictions: int,
    fallback: float,
    iou_match: float,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
) -> dict[str, Any]:
    from ultralytics import YOLO

    images = _dataset_image_paths(data, (split,))
    if not images:
        raise RuntimeError(f"No images found for split '{split}'.")
    names = parse_names(data)
    target_counts: dict[int, int] = defaultdict(int)
    records: dict[int, list[tuple[float, bool, float]]] = defaultdict(list)
    all_prediction_count = 0
    model = YOLO(str(checkpoint))
    results = model.predict(
        source=[str(path) for path in images], imgsz=imgsz, batch=batch, device=device, workers=workers,
        conf=min(thresholds) * 0.5, iou=0.7, max_det=300, verbose=False, stream=True,
    )
    for index, result in enumerate(results):
        if index >= len(images):
            raise RuntimeError("Inference emitted more results than validation images.")
        height, width = result.orig_shape
        targets = _read_ground_truth(_label_path(images[index]), width=width, height=height)
        for class_id, _ in targets:
            target_counts[class_id] += 1
        predictions: list[tuple[int, float, np.ndarray]] = []
        if result.obb is not None and len(result.obb):
            polygons = result.obb.xyxyxyxy.cpu().numpy()
            confidence = result.obb.conf.cpu().numpy()
            classes = result.obb.cls.cpu().numpy().astype(int)
            predictions = [(int(cls), float(score), polygon) for cls, score, polygon in zip(classes, confidence, polygons)]
            all_prediction_count += len(predictions)
        matched = match_with_iou(predictions, targets, iou_threshold=iou_match)
        for class_id, rows in matched.items():
            records[class_id].extend(rows)

    nc = max(max(names, default=-1), max(target_counts, default=-1), max(records, default=-1)) + 1
    classes: list[dict[str, Any]] = []
    selected_thresholds: list[float] = []
    curve_rows: list[dict[str, Any]] = []
    for class_id in range(nc):
        class_records = records.get(class_id, [])
        basic_records = [(score, is_tp) for score, is_tp, _ in class_records]
        selected, selected_score, used_fallback = choose_threshold(
            basic_records, target_counts.get(class_id, 0), precision_target=precision_target,
            min_predictions=min_predictions, candidates=thresholds, fallback=fallback,
        )
        selected_thresholds.append(float(selected))
        for threshold in thresholds:
            score = score_curve(class_records, target_counts.get(class_id, 0), threshold)
            score.update({"class_id": class_id, "class_name": names.get(class_id, f"class_{class_id}")})
            score["accepted_boxes_per_image"] = float(score["predictions"]) / len(images)
            curve_rows.append(score)
        classes.append({
            "class_id": class_id,
            "class_name": names.get(class_id, f"class_{class_id}"),
            "targets": target_counts.get(class_id, 0),
            "available_predictions": len(class_records),
            "selected_threshold": float(selected),
            "selected_score": selected_score,
            "fallback_used": used_fallback,
        })
    return {
        "checkpoint": str(checkpoint), "data": str(data), "split": split, "images": len(images),
        "iou_match": iou_match, "precision_target": precision_target, "min_predictions": min_predictions,
        "raw_predictions": all_prediction_count, "selected_thresholds": selected_thresholds,
        "classes": classes, "threshold_curve": curve_rows, "final_test_used": False,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", required=True, help="Completed validation-only protocol directory.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--method", default="baseline", help="Teacher checkpoint method within protocol-dir.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default="0.75,0.80,0.85,0.90,0.93")
    parser.add_argument("--precision-target", type=float, default=0.90)
    parser.add_argument("--min-predictions", type=int, default=20)
    parser.add_argument("--fallback", type=float, default=0.93)
    parser.add_argument("--iou-match", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_dir = Path(args.protocol_dir).expanduser().resolve()
    data = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = protocol_dir / "protocol_manifest.json"
    if not manifest.is_file() or not data.is_file():
        raise FileNotFoundError("Protocol manifest or data YAML is missing.")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostics directory: {output_dir}")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in metadata.get("seeds", [])]
    if not seeds:
        raise ValueError("Protocol manifest contains no seeds.")
    thresholds = threshold_grid(args.thresholds)
    if not (0.0 < args.precision_target <= 1.0 and 0.0 < args.iou_match <= 1.0):
        raise ValueError("Precision target and IoU match must be in (0, 1].")
    if args.min_predictions < 1 or args.imgsz % 32:
        raise ValueError("min-predictions must be positive and imgsz divisible by 32.")
    output_dir.mkdir(parents=True)
    reports = []
    all_curve_rows: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint = protocol_dir / args.method / f"seed{seed}" / "weights" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        report = diagnose_checkpoint(
            checkpoint, data, split=args.split, thresholds=thresholds, precision_target=args.precision_target,
            min_predictions=args.min_predictions, fallback=args.fallback, iou_match=args.iou_match,
            imgsz=args.imgsz, batch=args.batch, workers=args.workers, device=args.device,
        )
        report["seed"] = seed
        reports.append(report)
        for row in report["threshold_curve"]:
            all_curve_rows.append({"seed": seed, **row})
        (output_dir / f"seed{seed}_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    class_count = len(reports[0]["selected_thresholds"])
    consensus = [round(float(median([report["selected_thresholds"][index] for report in reports])), 4) for index in range(class_count)]
    write_csv(output_dir / "threshold_curve.csv", all_curve_rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "method": args.method, "seeds": seeds,
        "validation_only": True, "threshold_candidates": thresholds, "precision_target": args.precision_target,
        "per_seed_thresholds": {str(report["seed"]): report["selected_thresholds"] for report in reports},
        "median_consensus_thresholds": consensus,
        "recommended_cli_value": ",".join(f"{value:.4f}" for value in consensus),
    }
    (output_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
