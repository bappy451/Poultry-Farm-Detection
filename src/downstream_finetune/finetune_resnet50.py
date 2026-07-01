import os
import yaml
import torch
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from poultry_geo.train_spark_ssl import translate_state_dict

class YoloFormatSegmentationDataset(Dataset):
    def __init__(self, data_yaml_path, split="train", imgsz=640):
        with open(data_yaml_path, 'r') as f:
            self.data_cfg = yaml.safe_load(f)
        
        # Resolve base directory relative to yaml
        base_dir = Path(data_yaml_path).parent
        
        # YOLO typically defines train: path/to/images. If it's absolute, use it directly.
        split_dir = self.data_cfg.get(split)
        if Path(split_dir).is_absolute():
            self.img_dir = Path(split_dir)
        else:
            self.img_dir = base_dir / split_dir
            
        # Labels are typically in a parallel 'labels' directory
        self.lbl_dir = self.img_dir.parent / "labels"
        
        self.imgsz = imgsz
        self.nc = self.data_cfg.get("nc", 2)
        
        # Gather all valid images
        img_files_set = set()
        for ext in ('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'):
            # On Windows, glob is case-insensitive, so *.jpg and *.JPG return the same files.
            # Using a set ensures we don't duplicate them.
            for f in self.img_dir.glob(f"*{ext}"):
                img_files_set.add(f.resolve())
        self.img_files = list(img_files_set)
        
        if not self.img_files:
            raise RuntimeError(f"No images found in {self.img_dir}. Check your data.yaml '{split}' path.")
            
    def __len__(self):
        return len(self.img_files)
        
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Load label (polygon)
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        class_id = int(parts[0])
                        if len(parts) == 5:
                            # YOLO bounding box format: class_id cx cy bw bh
                            cx, cy, bw, bh = [float(x) for x in parts[1:]]
                            x1, y1 = cx - bw / 2, cy - bh / 2
                            x2, y2 = cx + bw / 2, cy + bh / 2
                            # Denormalize
                            x1, x2 = x1 * w, x2 * w
                            y1, y2 = y1 * h, y2 * h
                            coords = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                        else:
                            # YOLO polygon format: class_id x1 y1 x2 y2 ... (normalized)
                            coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                            # Denormalize
                            coords[:, 0] *= w
                            coords[:, 1] *= h
                            coords = coords.astype(np.int32)
                            
                        # Draw polygon on mask. Classes are 1-indexed (0 is background)
                        cv2.fillPoly(mask, [coords], class_id + 1)
                        
        # Resize
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)
        
        # HWC and normalize
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(mask).long() # Long for CrossEntropyLoss
        }

def build_resnet50_segmentation_model(num_classes, pretrained=False):
    """
    Builds a standard FCN with a ResNet50 backbone.
    num_classes includes the background (so YOLO nc + 1)
    """
    if pretrained:
        model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
    else:
        model = fcn_resnet50(weights=None)
        
    # Replace the classifier head for the specific number of classes
    # FCN head uses 512 channels from layer4.
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    # Also replace aux_classifier if present
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
    return model

def compute_confusion_matrix(preds, targets, num_classes):
    """
    Computes a flattened confusion matrix for segmentation.
    preds: [B, H, W]
    targets: [B, H, W]
    """
    mask = (targets >= 0) & (targets < num_classes)
    hist = torch.bincount(
        num_classes * targets[mask] + preds[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist

def train_resnet50_segmentation(data_yaml, model_weights, output_dir, epochs=50, batch_size=16, imgsz=640, device="cuda", pretrained=False):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    
    # Save args
    with open(out_dir / "args.yaml", "w") as f:
        yaml.dump({"data": data_yaml, "model_weights": model_weights, "epochs": epochs, "batch_size": batch_size, "imgsz": imgsz, "pretrained": pretrained}, f)
        
    # Dataset
    train_ds = YoloFormatSegmentationDataset(data_yaml, split="train", imgsz=imgsz)
    val_ds = YoloFormatSegmentationDataset(data_yaml, split="val", imgsz=imgsz)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    num_classes = train_ds.nc + 1 # +1 for background
    
    # Model
    model = build_resnet50_segmentation_model(num_classes, pretrained=pretrained)
    
    # Load SparK weights if provided
    if model_weights and Path(model_weights).exists():
        print(f"Loading backbone weights from {model_weights}")
        state_dict = torch.load(model_weights, map_location='cpu')
        # Handle different saved dict formats
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        translated = translate_state_dict(state_dict, backbone_type='resnet50')
        msg = model.backbone.load_state_dict(translated, strict=False)
        print(f"Loaded SparK weights into ResNet50 backbone. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
        
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # AMP Setup
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    accumulation_steps = max(1, 8 // batch_size) # Aim for effective batch size of 8
    
    results = []
    best_f1 = -1.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_dl):
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                out = model(imgs)["out"]
                loss = criterion(out, masks)
                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dl):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_loss += loss.item() * accumulation_steps
            
        train_loss /= max(1, len(train_dl))
        
        # Validation
        model.eval()
        val_loss = 0.0
        conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
        
        with torch.no_grad():
            for batch in val_dl:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    out = model(imgs)["out"]
                    loss = criterion(out, masks)
                val_loss += loss.item()
                
                preds = torch.argmax(out, dim=1)
                conf_matrix += compute_confusion_matrix(preds, masks, num_classes)
                
        val_loss /= max(1, len(val_dl))
        
        # Compute metrics from confusion matrix
        hist = conf_matrix.cpu().numpy()
        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp
        
        precision = tp / np.maximum(tp + fp, 1e-6)
        recall = tp / np.maximum(tp + fn, 1e-6)
        f1 = 2 * (precision * recall) / np.maximum(precision + recall, 1e-6)
        iou = tp / np.maximum(tp + fp + fn, 1e-6)
        
        # Average over foreground classes (exclude bg which is index 0)
        mean_p = precision[1:].mean()
        mean_r = recall[1:].mean()
        mean_f1 = f1[1:].mean()
        mean_iou = iou[1:].mean()
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - mF1: {mean_f1:.4f} - mIoU: {mean_iou:.4f}")
        
        results.append({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "metrics/precision(M)": mean_p,
            "metrics/recall(M)": mean_r,
            "metrics/mAP50(M)": mean_iou, # approximation for YOLO's mask map50
            "metrics/F1(M)": mean_f1
        })
        
        # Save weights
        torch.save(model.state_dict(), weights_dir / "last.pt")
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            torch.save(model.state_dict(), weights_dir / "best.pt")
            
            # Save confusion matrix plot for the best model
            fig, ax = plt.subplots(figsize=(8, 6))
            cax = ax.matshow(hist, cmap="Blues")
            fig.colorbar(cax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            class_names = ["background"] + [train_ds.data_cfg.get("names", {}).get(i, f"class_{i}") for i in range(train_ds.nc)]
            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(num_classes))
            ax.set_xticklabels(class_names, rotation=45)
            ax.set_yticklabels(class_names)
            for i in range(num_classes):
                for j in range(num_classes):
                    ax.text(j, i, str(int(hist[i, j])), va='center', ha='center')
            plt.savefig(out_dir / "confusion_matrix.png", bbox_inches='tight')
            plt.close()

    pd.DataFrame(results).to_csv(out_dir / "results.csv", index=False)
    print(f"Training complete. Artifacts saved to {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--weights", type=str, default=None, help="Path to SparK ResNet50 weights or ImageNet if not specified")
    parser.add_argument("--project", type=str, default="runs/segment/resnet50_finetune", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision ImageNet pretrained weights")
    args = parser.parse_args()
    
    train_resnet50_segmentation(
        data_yaml=args.data,
        model_weights=args.weights,
        output_dir=args.project,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        pretrained=args.pretrained
    )
