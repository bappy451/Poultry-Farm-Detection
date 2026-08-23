"""Measure why reliable CTQ pseudo-labels are sparse on unlabeled MS tiles.

This is a read-only, validation-safe diagnostic.  It never reads AL test data
and never updates a model.  For each frozen baseline teacher it reports the
number of pseudo OBBs surviving: (1) confidence/NMS, (2) class-margin, and
(3) the current two-view reliability gate.  It also reports two diagnostic
comparators for the gate.  Comparators are measurements, not training runs or
new selected hyperparameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for location in (ROOT, SRC):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from poultry_geo.naip_data import NAIPDataset
from scripts.ctq_reliable_obb import filter_pseudo_batch, stable_pseudo_mask, teacher_jitter
from scripts.ema_pseudo_obb import build_pseudo_batch


def _device(value: str) -> torch.device:
    if value.isdigit():
        return torch.device(f"cuda:{value}")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return result


def _count_by_class(pseudo: dict[str, torch.Tensor], nc: int) -> list[int]:
    classes = pseudo["cls"].detach().to(device="cpu", dtype=torch.long)
    return torch.bincount(classes, minlength=nc).tolist() if classes.numel() else [0] * nc


def _accumulate(target: dict[str, Any], pseudo: dict[str, torch.Tensor], image_count: int, nc: int) -> None:
    count = int(pseudo["bboxes"].shape[0])
    target["boxes"] += count
    target["images"] += image_count
    target["nonempty_images"] += int(torch.unique(pseudo["batch_idx"]).numel()) if count else 0
    per_class = _count_by_class(pseudo, nc)
    for index, value in enumerate(per_class):
        target["class_boxes"][index] += int(value)


def _empty(nc: int) -> dict[str, Any]:
    return {"images": 0, "nonempty_images": 0, "boxes": 0, "class_boxes": [0] * nc}


def _report_stage(name: str, values: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    images = int(values["images"])
    boxes = int(values["boxes"])
    return {
        "stage": name,
        "images": images,
        "images_with_boxes": int(values["nonempty_images"]),
        "image_coverage": float(values["nonempty_images"]) / images if images else 0.0,
        "boxes": boxes,
        "boxes_per_image": boxes / images if images else 0.0,
        "class_boxes": {class_names[index]: int(value) for index, value in enumerate(values["class_boxes"])},
    }


def _retention(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["seed", "stage", "images", "images_with_boxes", "image_coverage", "boxes", "boxes_per_image"]
    class_fields = sorted({key for row in rows for key in row if key.startswith("class_")})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fields, *class_fields])
        writer.writeheader()
        writer.writerows(rows)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Diagnostic config must be a YAML mapping.")
    required = {"protocol_dir", "unlabeled_root", "output_dir", "seeds", "sample_size", "subset_seed", "imgsz", "batch", "workers", "device"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Diagnostic config is missing: {sorted(missing)}")
    if int(config["imgsz"]) != 512:
        raise ValueError("This AL/MS diagnostic requires imgsz: 512, matching the native MS tile size.")
    if int(config["sample_size"]) <= 0 or int(config["batch"]) <= 0 or int(config["workers"]) < 0:
        raise ValueError("sample_size/batch must be positive and workers cannot be negative.")
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)
    protocol_dir = Path(str(config["protocol_dir"])).expanduser().resolve()
    unlabeled_root = Path(str(config["unlabeled_root"])).expanduser().resolve()
    output_dir = Path(str(config["output_dir"])).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse diagnostic output directory: {output_dir}")
    if not (protocol_dir / "protocol.json").is_file():
        raise FileNotFoundError(f"Missing completed protocol metadata: {protocol_dir / 'protocol.json'}")
    protocol = json.loads((protocol_dir / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("test_used_for_selection") is not False:
        raise ValueError("This diagnostic requires a validation-only protocol.")

    seeds = [int(seed) for seed in config["seeds"]]
    if sorted(seeds) != [7, 17, 42]:
        raise ValueError("Use the frozen publication seeds [7, 17, 42].")
    calibration: dict[int, list[float]] = {}
    for seed in seeds:
        checkpoint = protocol_dir / "baseline" / f"seed{seed}" / "weights" / "best.pt"
        threshold_path = protocol_dir / "calibration" / f"seed{seed}.json"
        if not checkpoint.is_file() or not threshold_path.is_file():
            raise FileNotFoundError(f"Missing baseline checkpoint or calibration for seed {seed}.")
        thresholds = json.loads(threshold_path.read_text(encoding="utf-8")).get("thresholds")
        if not isinstance(thresholds, list) or not thresholds or any(not 0.0 < float(value) < 1.0 for value in thresholds):
            raise ValueError(f"Invalid calibrated thresholds for seed {seed}.")
        calibration[seed] = [float(value) for value in thresholds]

    device = _device(str(config["device"]))
    dataset = NAIPDataset(unlabeled_root, crop_size=int(config["imgsz"]), channels=None, precision="fp32", augment=False, max_files=int(config["sample_size"]), subset_seed=int(config["subset_seed"]))
    if len(dataset) != int(config["sample_size"]):
        raise RuntimeError(f"Expected {config['sample_size']} sampled MS tiles, found {len(dataset)}.")
    loader = DataLoader(dataset, batch_size=int(config["batch"]), shuffle=False, num_workers=int(config["workers"]), pin_memory=device.type == "cuda", persistent_workers=int(config["workers"]) > 0)

    output_dir.mkdir(parents=True)
    (output_dir / "sample_manifest.txt").write_text("\n".join(str(path) for path in dataset.files) + "\n", encoding="utf-8")
    (output_dir / "config_used.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    class_names = ["broiler breeder", "broiler house"]
    all_rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    for seed in seeds:
        from ultralytics import YOLO

        torch.manual_seed(int(config["subset_seed"]) + seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(config["subset_seed"]) + seed)
        teacher = YOLO(str(protocol_dir / "baseline" / f"seed{seed}" / "weights" / "best.pt")).model.to(device).eval()
        nc = int(getattr(teacher, "nc", len(calibration[seed])))
        if nc != len(calibration[seed]) or nc != len(class_names):
            raise ValueError(f"Unexpected class count for seed {seed}: {nc}.")
        stages = {name: _empty(nc) for name in ("confidence_nms", "class_margin", "current_reliability", "one_of_two_agreement", "lower_iou_reliability")}
        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(device, dtype=torch.float32, non_blocking=True)
                image_count = int(images.shape[0])
                reference_inference = teacher.predict(images)[0]
                confidence = build_pseudo_batch(reference_inference, image_size=(int(config["imgsz"]), int(config["imgsz"])), nc=nc, class_thresholds=calibration[seed], iou_threshold=float(config.get("pseudo_iou_threshold", 0.5)), class_margin=0.0)
                margin = build_pseudo_batch(reference_inference, image_size=(int(config["imgsz"]), int(config["imgsz"])), nc=nc, class_thresholds=calibration[seed], iou_threshold=float(config.get("pseudo_iou_threshold", 0.5)), class_margin=float(config.get("pseudo_class_margin", 0.05)))
                jittered = []
                for _ in range(int(config.get("teacher_jitter_views", 2))):
                    predicted = teacher.predict(teacher_jitter(images))[0]
                    jittered.append(build_pseudo_batch(predicted, image_size=(int(config["imgsz"]), int(config["imgsz"])), nc=nc, class_thresholds=calibration[seed], iou_threshold=float(config.get("pseudo_iou_threshold", 0.5)), class_margin=float(config.get("pseudo_class_margin", 0.05))))
                current = filter_pseudo_batch(margin, stable_pseudo_mask(margin, jittered, min_iou=float(config.get("teacher_jitter_iou", 0.70)), min_agreement=float(config.get("teacher_jitter_agreement", 1.0))))
                one_of_two = filter_pseudo_batch(margin, stable_pseudo_mask(margin, jittered, min_iou=float(config.get("teacher_jitter_iou", 0.70)), min_agreement=float(config.get("comparison_agreement", 0.5))))
                lower_iou = filter_pseudo_batch(margin, stable_pseudo_mask(margin, jittered, min_iou=float(config.get("comparison_iou", 0.5)), min_agreement=float(config.get("teacher_jitter_agreement", 1.0))))
                for name, pseudo in (("confidence_nms", confidence), ("class_margin", margin), ("current_reliability", current), ("one_of_two_agreement", one_of_two), ("lower_iou_reliability", lower_iou)):
                    _accumulate(stages[name], pseudo, image_count, nc)
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
        stage_reports = [_report_stage(name, value, class_names) for name, value in stages.items()]
        by_name = {row["stage"]: row for row in stage_reports}
        seed_report = {
            "seed": seed,
            "checkpoint": str(protocol_dir / "baseline" / f"seed{seed}" / "weights" / "best.pt"),
            "calibrated_thresholds": calibration[seed],
            "sample_images": len(dataset),
            "stages": stage_reports,
            "retention": {
                "margin_from_confidence_nms": _retention(by_name["class_margin"]["boxes"], by_name["confidence_nms"]["boxes"]),
                "current_gate_from_margin": _retention(by_name["current_reliability"]["boxes"], by_name["class_margin"]["boxes"]),
                "current_gate_from_confidence_nms": _retention(by_name["current_reliability"]["boxes"], by_name["confidence_nms"]["boxes"]),
            },
            "final_test_used": False,
        }
        reports.append(seed_report)
        (output_dir / f"seed{seed}.json").write_text(json.dumps(seed_report, indent=2), encoding="utf-8")
        for report in stage_reports:
            all_rows.append({"seed": seed, **{key: value for key, value in report.items() if key != "class_boxes"}, **{f"class_{key.replace(' ', '_')}": value for key, value in report["class_boxes"].items()}})

    _write_csv(output_dir / "retention_by_seed.csv", all_rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_safe": True,
        "final_test_used": False,
        "protocol_dir": str(protocol_dir),
        "unlabeled_root": str(unlabeled_root),
        "sample_images": len(dataset),
        "sample_manifest": str(output_dir / "sample_manifest.txt"),
        "per_seed_reports": [str(output_dir / f"seed{seed}.json") for seed in seeds],
        "retention_table": str(output_dir / "retention_by_seed.csv"),
        "interpretation": "Use this output to identify whether confidence/NMS, class margin, or two-view stability is the dominant pseudo-label bottleneck. Do not select a new setting using AL test data.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
