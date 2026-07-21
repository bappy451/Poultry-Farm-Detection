"""
prepare_and_train_dota.py
=========================
1. Converts the DOTA COCO JSON format (segmentations) to Ultralytics YOLO OBB format.
   Instead of copying gigabytes of images, it writes the .txt labels directly into
   a `labels` directory inside the DOTA dataset folder, which YOLO understands natively.
2. Generates the `configs/dota_obb.yaml` dynamically from the JSON classes.
3. Trains and evaluates two YOLO11l-OBB models for 50 epochs:
     Run A — Official Ultralytics ImageNet weights
     Run B — Our SparK-from-scratch backbone
4. Saves results in `runs/downstream_dota`.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── Source Paths ──────────────────────────────────────────────────────────────
DOTA_DIR = Path(r"C:\Users\mza0288\Documents\Poultry-Farm-Detection\temp\dota_coco_format\dota")
DATA_YAML = ROOT / "configs" / "dota_obb.yaml"
OUTPUT_DIR = ROOT / "runs" / "downstream_dota"

# We point directly to the SparK-from-scratch weights at epoch 300
SPARK_BACKBONE_PT = (
    ROOT
    / "trained_models"
    / "spark-scratch"
    / "af8e64a7ed89_yolo11l_fp16"
    / "backbone_dense_ep299.pt"
)

# ── Data Preparation Function ─────────────────────────────────────────────────
def convert_coco_to_yolo_obb(json_path: Path, labels_out_dir: Path) -> dict:
    if not json_path.exists():
        print(f"Skipping {json_path.name}: file not found.")
        return {}
        
    print(f"\n[Dataset] Processing {json_path.name} -> {labels_out_dir}")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Extract classes for the YAML file
    categories = data.get('categories', [])
    class_map = {cat['id']: cat['name'] for cat in categories}
        
    img_dict = {img['id']: img for img in data.get('images', [])}
    
    ann_dict = {}
    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in ann_dict:
            ann_dict[img_id] = []
        ann_dict[img_id].append(ann)
        
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    
    for img_id, img_info in tqdm(img_dict.items()):
        w = float(img_info['width'])
        h = float(img_info['height'])
        file_name = img_info['file_name']
        stem = Path(file_name).stem
        
        anns = ann_dict.get(img_id, [])
        label_lines = []
        for ann in anns:
            cat_id = ann['category_id']
            seg = ann.get('segmentation', [])
            if not seg or len(seg[0]) != 8:
                continue
                
            pts = seg[0]
            # Normalize coordinates
            def norm(v, max_v): return max(0.0, min(1.0, float(v) / max_v))
            
            x1, y1 = norm(pts[0], w), norm(pts[1], h)
            x2, y2 = norm(pts[2], w), norm(pts[3], h)
            x3, y3 = norm(pts[4], w), norm(pts[5], h)
            x4, y4 = norm(pts[6], w), norm(pts[7], h)
            
            label_lines.append(f"{cat_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}")
            
        with open(labels_out_dir / f"{stem}.txt", "w") as f:
            if label_lines:
                f.write("\n".join(label_lines))
                
    return class_map

def prepare_dota_dataset():
    labels_train = DOTA_DIR / "labels" / "train"
    labels_val = DOTA_DIR / "labels" / "val"
    
    class_map = {}
    if not labels_train.exists():
        json_train = DOTA_DIR / "annotations" / "instances_train.json"
        cmap = convert_coco_to_yolo_obb(json_train, labels_train)
        if cmap: class_map.update(cmap)
    else:
        print(f"[Dataset] Train labels already exist at {labels_train}. Skipping train conversion.")
        
    if not labels_val.exists():
        json_val = DOTA_DIR / "annotations" / "instances_val.json"
        cmap = convert_coco_to_yolo_obb(json_val, labels_val)
        if cmap: class_map.update(cmap)
    else:
        print(f"[Dataset] Val labels already exist at {labels_val}. Skipping val conversion.")

    # We need class map to create YAML, if we skipped conversion we must load it
    if not class_map:
        json_val = DOTA_DIR / "annotations" / "instances_val.json"
        if json_val.exists():
            with open(json_val, 'r') as f:
                data = json.load(f)
                class_map = {cat['id']: cat['name'] for cat in data.get('categories', [])}

    if not DATA_YAML.exists() and class_map:
        # Create YAML
        # Format names correctly
        names_str = "\n".join([f"  {k}: {v}" for k, v in sorted(class_map.items())])
        yaml_content = f"""path: {DOTA_DIR.absolute()}
train: train
val: val
test: test

# Classes
names:
{names_str}
"""
        with open(DATA_YAML, "w") as f:
            f.write(yaml_content)
        print(f"[Dataset] YAML created at {DATA_YAML}")

# ── Shared fine-tuning hyperparameters ───────────────────────────────────────
COMMON_TRAIN_ARGS: dict = dict(
    data=str(DATA_YAML),
    epochs=20,          # 20-epoch budget
    imgsz=640,          # Standard DOTA crop size is typically 1024, but keeping 640 for fair comparison with previous runs
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


def run_experiment(name: str, use_spark: bool) -> dict:
    from ultralytics import YOLO

    save_dir = OUTPUT_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Starting experiment: {name}")
    print(f"  SparK backbone : {use_spark}")
    print(f"  Save dir       : {save_dir}")
    print(f"{'='*70}\n")

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

    prepare_dota_dataset()
    
    if not DATA_YAML.exists():
        print(f"ERROR: Dataset YAML not found at:\n  {DATA_YAML}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    # ── Run A: Official ImageNet baseline ─────────────────────────────────────
    metrics_a = run_experiment("imagenet_baseline", use_spark=False)
    results.append(metrics_a)

    # ── Run B: SparK from scratch ─────────────────────────────────────────────
    metrics_b = run_experiment("spark_scratch", use_spark=True)
    results.append(metrics_b)

    print("\n" + "=" * 70)
    print("  COMPARISON RESULTS (50 Epochs - DOTA Dataset)")
    print("=" * 70)

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

    out_csv = OUTPUT_DIR / "comparison_results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    print(f"\nResults saved to: {out_csv}")

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
