import pytest
import os
import torch
from pathlib import Path
from downstream_finetune.finetune_resnet50 import (
    YoloFormatSegmentationDataset,
    build_resnet50_segmentation_model,
    train_resnet50_segmentation,
)

def test_yolo_format_dataset_parsing(tmp_path):
    # Mock a YOLO dataset
    data_yaml = tmp_path / "data.yaml"
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    
    data_yaml.write_text("""
train: images
val: images
nc: 2
names:
  0: class_0
  1: class_1
""")
    
    # Create a mock image
    from PIL import Image
    import numpy as np
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img_dir / "test.jpg")
    
    # Create a mock label (class_id, x1, y1, x2, y2, x3, y3...)
    # Normalized coordinates
    (lbl_dir / "test.txt").write_text("1 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6")
    
    ds = YoloFormatSegmentationDataset(str(data_yaml), split="train", imgsz=64)
    
    assert len(ds) == 1
    batch = ds[0]
    assert batch["image"].shape == (3, 64, 64)
    assert batch["mask"].shape == (64, 64)
    assert batch["mask"].max().item() <= 2 # classes are 1-indexed, 0 is bg

def test_build_resnet50_segmentation_model():
    model = build_resnet50_segmentation_model(num_classes=3, pretrained=False)
    # Output should have 3 channels
    x = torch.randn(2, 3, 224, 224)
    y = model(x)["out"]
    assert y.shape == (2, 3, 224, 224)

def test_train_resnet50_mock(tmp_path, monkeypatch):
    # Test the training loop structure without real data
    data_yaml = tmp_path / "data.yaml"
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    
    data_yaml.write_text("""
train: images
val: images
nc: 2
names:
  0: class_0
  1: class_1
""")
    from PIL import Image
    import numpy as np
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(img_dir / "test.jpg")
    (lbl_dir / "test.txt").write_text("1 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6")
    
    run_dir = tmp_path / "runs" / "segment" / "test_run"
    
    train_resnet50_segmentation(
        data_yaml=str(data_yaml),
        model_weights=None, # no pretrained
        output_dir=str(run_dir),
        epochs=1,
        batch_size=1,
        imgsz=64,
        device="cpu"
    )
    
    # Assert artifacts were saved
    assert (run_dir / "weights" / "best.pt").exists()
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "confusion_matrix.png").exists()
    assert (run_dir / "args.yaml").exists()
