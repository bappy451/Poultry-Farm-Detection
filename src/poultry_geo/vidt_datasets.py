import yaml
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class YoloFormatDetectionDataset(Dataset):
    """
    Dataset that loads YOLO format OBB (or standard) labels and returns them
    as targets compatible with Hungarian Matcher (boxes, labels, and masks).
    """
    def __init__(self, data_yaml_path, split="train", imgsz=640):
        with open(data_yaml_path, 'r') as f:
            self.data_cfg = yaml.safe_load(f) or {}
            
        if "path" in self.data_cfg and self.data_cfg["path"]:
            base_dir = Path(self.data_cfg["path"]).expanduser().resolve()
        else:
            base_dir = Path(data_yaml_path).parent.resolve()
            
        split_dir = self.data_cfg.get(split)
        if Path(split_dir).is_absolute():
            self.img_dir = Path(split_dir)
        else:
            self.img_dir = (base_dir / split_dir).resolve()
            
        self.lbl_dir = self.img_dir.parent / "labels"
        
        self.imgsz = imgsz
        self.nc = self.data_cfg.get("nc", 2)
        
        img_files_set = set()
        for ext in ('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'):
            for f in self.img_dir.glob(f"*{ext}"):
                img_files_set.add(f.resolve())
        self.img_files = list(img_files_set)
        
        if not self.img_files:
            raise RuntimeError(f"No images found in {self.img_dir}. Check your data.yaml.")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        lbl_path = self.lbl_dir / (img_path.stem + ".txt")
        
        boxes = []
        labels = []
        masks = []
        
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        class_id = int(parts[0])
                        if len(parts) == 5:
                            # YOLO bounding box: cx cy bw bh (normalized)
                            cx, cy, bw, bh = [float(x) for x in parts[1:]]
                            boxes.append([cx, cy, bw, bh])
                            
                            # Create a rectangular mask
                            mask = np.zeros((self.imgsz, self.imgsz), dtype=np.uint8)
                            x1, y1 = int((cx - bw/2) * self.imgsz), int((cy - bh/2) * self.imgsz)
                            x2, y2 = int((cx + bw/2) * self.imgsz), int((cy + bh/2) * self.imgsz)
                            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
                            masks.append(mask)
                        else:
                            # YOLO polygon format (OBB)
                            coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                            
                            # Compute bounding box from polygon (normalized)
                            xmin, ymin = coords.min(axis=0)
                            xmax, ymax = coords.max(axis=0)
                            cx = (xmin + xmax) / 2
                            cy = (ymin + ymax) / 2
                            bw = xmax - xmin
                            bh = ymax - ymin
                            boxes.append([cx, cy, bw, bh])
                            
                            # Create polygon mask
                            mask = np.zeros((self.imgsz, self.imgsz), dtype=np.uint8)
                            coords[:, 0] *= self.imgsz
                            coords[:, 1] *= self.imgsz
                            cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
                            masks.append(mask)
                            
                        labels.append(class_id)
        
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # Apply ImageNet normalization (crucial for pretrained models like DINO/ImageNet)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img = (img - mean) / std

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "masks": torch.tensor(np.stack(masks), dtype=torch.float32) if masks else torch.zeros((0, self.imgsz, self.imgsz), dtype=torch.float32),
        }

        return {
            "image": torch.from_numpy(img),
            "target": target
        }
