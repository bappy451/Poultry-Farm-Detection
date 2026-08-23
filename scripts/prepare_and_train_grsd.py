"""
prepare_and_train_grsd.py
=========================
1. Automatically converts the GRSD (Greenhouse Remote Sensing Dataset) from 
   its native XML (robndbox) format to Ultralytics YOLO OBB format.
2. Splits the dataset into Train and Val sets (80/20).
3. Trains and evaluates two YOLO11l-OBB models for 50 epochs:
     Run A â€” Official Ultralytics ImageNet weights
     Run B â€” Our SparK-from-scratch backbone
4. Saves results in `runs/downstream_grsd`.
"""

from __future__ import annotations

import csv
import math
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from tqdm import tqdm

# â”€â”€ Project root â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# â”€â”€ Source and Target Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SOURCE_DIR = Path(r"C:\Users\mza0288\Documents\Poultry-Farm-Detection\temp\the greenhouse remote sensing image dataset (GRSD)")
TARGET_DIR = ROOT / "datasets" / "GRSD_YOLO"
DATA_YAML = ROOT / "configs" / "grsd_obb.yaml"
OUTPUT_DIR = ROOT / "runs" / "downstream_grsd"

# We point directly to the SparK-from-scratch weights at epoch 300
SPARK_BACKBONE_PT = (
    ROOT
    / "trained_models"
    / "spark-scratch"
    / "af8e64a7ed89_yolo11l_fp16"
    / "backbone_dense_best.pt"
)

# â”€â”€ Data Preparation Function â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def convert_grsd_to_yolo_obb():
    """Converts the GRSD XML files to YOLO OBB format and structures the dataset."""
    if (TARGET_DIR / "images").exists() and (TARGET_DIR / "labels").exists() and DATA_YAML.exists():
        print(f"[Dataset] Found existing YOLO OBB dataset at {TARGET_DIR}. Skipping conversion.")
        return

    print(f"\n[Dataset] Preparing GRSD dataset at {TARGET_DIR}...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    img_dir = SOURCE_DIR / "JPEGImages"
    ann_dir = SOURCE_DIR / "Annotations"
    
    if not img_dir.exists() or not ann_dir.exists():
        print(f"ERROR: Cannot find JPEGImages or Annotations in {SOURCE_DIR}")
        sys.exit(1)

    all_images = list(img_dir.glob("*.png"))
    if not all_images:
        print(f"ERROR: No images found in {img_dir}")
        sys.exit(1)
        
    # Shuffle and split 80/20
    random.seed(42)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]
    
    # Create directories
    for split in ["train", "val"]:
        (TARGET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    def process_split(img_list, split_name):
        print(f"  Converting {split_name} set ({len(img_list)} images)...")
        for img_path in tqdm(img_list):
            xml_path = ann_dir / (img_path.stem + ".xml")
            if not xml_path.exists():
                continue
                
            shutil.copy(img_path, TARGET_DIR / "images" / split_name / img_path.name)
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image dimensions for normalization
            size_node = root.find("size")
            img_w = float(size_node.find("width").text)
            img_h = float(size_node.find("height").text)
            
            labels = []
            for obj in root.findall("object"):
                name = obj.find("name").text
                if name != "plastic greenhouse":
                    continue
                class_id = 0 # Only one class
                
                robndbox = obj.find("robndbox")
                cx = float(robndbox.find("cx").text)
                cy = float(robndbox.find("cy").text)
                w = float(robndbox.find("w").text)
                h = float(robndbox.find("h").text)
                angle = float(robndbox.find("angle").text)
                
                # Convert (cx, cy, w, h, angle) to 4 corners
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                
                dx1, dy1 = -w/2, -h/2
                dx2, dy2 = w/2, -h/2
                dx3, dy3 = w/2, h/2
                dx4, dy4 = -w/2, h/2
                
                def rotate(dx, dy):
                    return cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a
                    
                x1, y1 = rotate(dx1, dy1)
                x2, y2 = rotate(dx2, dy2)
                x3, y3 = rotate(dx3, dy3)
                x4, y4 = rotate(dx4, dy4)
                
                # Normalize (clip to 0-1)
                def norm(val, max_val):
                    return max(0.0, min(1.0, val / max_val))
                    
                nx1, ny1 = norm(x1, img_w), norm(y1, img_h)
                nx2, ny2 = norm(x2, img_w), norm(y2, img_h)
                nx3, ny3 = norm(x3, img_w), norm(y3, img_h)
                nx4, ny4 = norm(x4, img_w), norm(y4, img_h)
                
                labels.append(f"{class_id} {nx1:.6f} {ny1:.6f} {nx2:.6f} {ny2:.6f} {nx3:.6f} {ny3:.6f} {nx4:.6f} {ny4:.6f}")
                
            label_path = TARGET_DIR / "labels" / split_name / (img_path.stem + ".txt")
            with open(label_path, "w") as f:
                f.write("\n".join(labels))

    process_split(train_imgs, "train")
    process_split(val_imgs, "val")
    
    # Write yaml config
    yaml_content = f"""path: {TARGET_DIR.absolute()}
train: images/train
val: images/val
test: images/val

# Classes
names:
  0: plastic greenhouse
"""
    with open(DATA_YAML, "w") as f:
        f.write(yaml_content)
    print(f"[Dataset] YAML created at {DATA_YAML}")

# â”€â”€ Shared fine-tuning hyperparameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
COMMON_TRAIN_ARGS: dict = dict(
    data=str(DATA_YAML),
    epochs=50,          # 50-epoch budget
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
    raise RuntimeError(
        "SparK-to-YOLO transfer is disabled: the supported SparK encoder is ResNet50, "
        "and this project has no verified architecture-preserving YOLO adapter."
    )
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

    convert_grsd_to_yolo_obb()
    
    if not DATA_YAML.exists():
        print(f"ERROR: Dataset YAML not found at:\n  {DATA_YAML}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    # â”€â”€ Run A: Official ImageNet baseline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    metrics_a = run_experiment("imagenet_baseline", use_spark=False)
    results.append(metrics_a)

    # â”€â”€ Run B: SparK from scratch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    metrics_b = run_experiment("spark_scratch", use_spark=True)
    results.append(metrics_b)

    print("\n" + "=" * 70)
    print("  COMPARISON RESULTS (50 Epochs - GRSD Greenhouse Dataset)")
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
