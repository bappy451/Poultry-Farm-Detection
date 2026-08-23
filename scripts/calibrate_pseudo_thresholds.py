"""Calibrate class-specific teacher confidence thresholds on a YOLO-OBB val split.

The calibration split is used only to choose pseudo-label confidence thresholds.
The final test split is never read by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _label_path(image_path: Path) -> Path | None:
    """Resolve YOLO-OBB labels for common images/labels and manifest layouts."""
    direct = image_path.with_suffix(".txt")
    candidates = [direct]
    parts = list(image_path.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        index = len(lowered) - 1 - lowered[::-1].index("images")
        label_parts = parts[:]
        label_parts[index] = "labels"
        candidates.append(Path(*label_parts).with_suffix(".txt"))
    if image_path.parent.parent:
        candidates.append(image_path.parent.parent / "labels" / image_path.name)
        candidates[-1] = candidates[-1].with_suffix(".txt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    from shapely.geometry import Polygon

    first_polygon = Polygon(first).buffer(0)
    second_polygon = Polygon(second).buffer(0)
    if first_polygon.is_empty or second_polygon.is_empty:
        return 0.0
    union = first_polygon.union(second_polygon).area
    return float(first_polygon.intersection(second_polygon).area / union) if union else 0.0


def _read_ground_truth(label_path: Path | None, *, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    if label_path is None:
        return []
    targets: list[tuple[int, np.ndarray]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if not values:
            continue
        numbers = [float(value) for value in values]
        class_id = int(numbers[0])
        if len(numbers) >= 9 and (len(numbers) - 1) % 2 == 0:
            polygon = np.asarray(numbers[1:], dtype=np.float32).reshape(-1, 2)
            polygon[:, 0] *= width
            polygon[:, 1] *= height
        elif len(numbers) == 6:
            # Also accept normalized xywhr labels for datasets exported that way.
            cx, cy, box_width, box_height, angle = numbers[1:]
            angle = float(angle)
            cos_angle, sin_angle = math.cos(angle), math.sin(angle)
            corners = np.asarray(
                [
                    (-box_width / 2, -box_height / 2),
                    (box_width / 2, -box_height / 2),
                    (box_width / 2, box_height / 2),
                    (-box_width / 2, box_height / 2),
                ],
                dtype=np.float32,
            )
            rotation = np.asarray([[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=np.float32)
            polygon = corners @ rotation.T
            polygon += np.asarray([cx, cy], dtype=np.float32)
            polygon[:, 0] *= width
            polygon[:, 1] *= height
        else:
            raise ValueError(f"Unsupported OBB label format in {label_path}: {len(numbers)} values")
        targets.append((class_id, polygon))
    return targets


def match_predictions(
    predictions: Iterable[tuple[int, float, np.ndarray]],
    targets: Iterable[tuple[int, np.ndarray]],
    *,
    iou_threshold: float,
) -> dict[int, list[tuple[float, bool]]]:
    """Greedily match confidence-sorted predictions to same-class OBB targets."""
    grouped_targets: dict[int, list[np.ndarray]] = defaultdict(list)
    for class_id, polygon in targets:
        grouped_targets[int(class_id)].append(polygon)
    matched: dict[int, set[int]] = defaultdict(set)
    records: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    ordered = sorted(predictions, key=lambda item: float(item[1]), reverse=True)
    for class_id, confidence, polygon in ordered:
        class_id = int(class_id)
        best_iou = 0.0
        best_index = -1
        for index, target in enumerate(grouped_targets[class_id]):
            if index in matched[class_id]:
                continue
            overlap = _polygon_iou(polygon, target)
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        is_true_positive = best_index >= 0 and best_iou >= float(iou_threshold)
        if is_true_positive:
            matched[class_id].add(best_index)
        records[class_id].append((float(confidence), is_true_positive))
    return records


def score_threshold(
    records: Iterable[tuple[float, bool]],
    target_count: int,
    threshold: float,
) -> dict[str, float | int]:
    selected = [(confidence, true_positive) for confidence, true_positive in records if confidence >= threshold]
    true_positives = sum(1 for _, true_positive in selected if true_positive)
    precision = true_positives / len(selected) if selected else 0.0
    recall = true_positives / target_count if target_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "predictions": len(selected),
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def choose_threshold(
    records: Iterable[tuple[float, bool]],
    target_count: int,
    *,
    precision_target: float,
    min_predictions: int,
    candidates: Iterable[float],
    fallback: float,
) -> tuple[float, dict[str, float | int], bool]:
    scored = [score_threshold(records, target_count, threshold) for threshold in candidates]
    eligible = [
        row for row in scored
        if int(row["predictions"]) >= min_predictions and float(row["precision"]) >= precision_target
    ]
    if eligible:
        selected = max(eligible, key=lambda row: (float(row["recall"]), -float(row["threshold"])))
        return float(selected["threshold"]), selected, False
    selected = max(scored, key=lambda row: (float(row["f1"]), float(row["precision"]), -float(row["threshold"]))) if scored else score_threshold(records, target_count, fallback)
    if not scored or int(selected["predictions"]) == 0:
        selected = score_threshold(records, target_count, fallback)
    selected = dict(selected)
    selected["fallback_threshold"] = float(fallback)
    return float(fallback), selected, True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--iou-match", type=float, default=0.50)
    parser.add_argument("--precision-target", type=float, default=0.90)
    parser.add_argument("--min-predictions", type=int, default=10)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--max-confidence", type=float, default=0.99)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--fallback", type=float, default=0.90)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not 0.0 < args.precision_target <= 1.0:
        raise ValueError("precision-target must be in (0, 1].")
    if not 0.0 < args.iou_match <= 1.0:
        raise ValueError("iou-match must be in (0, 1].")
    if args.imgsz % 32:
        raise ValueError("imgsz must be divisible by 32.")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    data = Path(args.data).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not data.is_file():
        raise FileNotFoundError(data)

    from scripts.run_ema_benchmark import _dataset_image_paths
    from ultralytics import YOLO
    import yaml

    payload = yaml.safe_load(data.read_text(encoding="utf-8")) or {}
    names = payload.get("names", {0: "class_0"})
    if isinstance(names, list):
        names = {index: value for index, value in enumerate(names)}
    names = {int(key): str(value) for key, value in names.items()}
    images = _dataset_image_paths(data, (args.split,))
    if not images:
        raise RuntimeError(f"No images found for split '{args.split}' in {data}")

    model = YOLO(str(checkpoint))
    records: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    target_counts: dict[int, int] = defaultdict(int)
    predictions_seen = 0
    for image_index, result in enumerate(model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=0.001,
        iou=0.7,
        max_det=300,
        verbose=False,
        stream=True,
    )):
        height, width = result.orig_shape
        # Ultralytics may expose ``image0.jpg`` for list-based streamed input.
        # The source order is deterministic, so use it as the authoritative
        # mapping back to the validation label file.
        source_path = images[image_index] if image_index < len(images) else Path(result.path)
        targets = _read_ground_truth(_label_path(source_path), width=width, height=height)
        for class_id, _ in targets:
            target_counts[class_id] += 1
        image_predictions: list[tuple[int, float, np.ndarray]] = []
        if result.obb is not None and len(result.obb) > 0:
            polygons = result.obb.xyxyxyxy.cpu().numpy()
            confidences = result.obb.conf.cpu().numpy()
            classes = result.obb.cls.cpu().numpy().astype(int)
            predictions_seen += len(classes)
            image_predictions = [
                (int(class_id), float(confidence), polygon)
                for class_id, confidence, polygon in zip(classes, confidences, polygons)
            ]
        image_records = match_predictions(image_predictions, targets, iou_threshold=args.iou_match)
        for class_id, class_records in image_records.items():
            records[class_id].extend(class_records)

    candidates = np.arange(args.min_confidence, args.max_confidence + args.step / 2.0, args.step)
    thresholds: list[float] = []
    calibration: list[dict[str, object]] = []
    nc = max(max(names, default=-1) + 1, max(target_counts, default=-1) + 1, max(records, default=-1) + 1)
    for class_id in range(nc):
        threshold, score, fallback = choose_threshold(
            records.get(class_id, []),
            target_counts.get(class_id, 0),
            precision_target=args.precision_target,
            min_predictions=args.min_predictions,
            candidates=candidates.tolist(),
            fallback=args.fallback,
        )
        thresholds.append(threshold)
        calibration.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, f"class_{class_id}"),
                "threshold": threshold,
                "targets": target_counts.get(class_id, 0),
                "available_predictions": len(records.get(class_id, [])),
                "selected": score,
                "fallback": fallback,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "data": str(data),
                "split": args.split,
                "images": len(images),
                "predictions_seen": predictions_seen,
                "iou_match": args.iou_match,
                "precision_target": args.precision_target,
                "min_predictions": args.min_predictions,
                "thresholds": [round(value, 4) for value in thresholds],
                "classes": calibration,
                "final_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"thresholds": thresholds, "output": str(output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
