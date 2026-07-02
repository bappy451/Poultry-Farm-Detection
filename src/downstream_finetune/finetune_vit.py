import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn

from poultry_geo.ssl_models import RemoteSensingSegmentationModel
from .finetune_resnet50 import YoloFormatSegmentationDataset

def load_mim_backbone_into_vit(model, weights_path):
    """
    Loads MIM pretrained backbone weights into a RemoteSensingSegmentationModel.
    """
    print(f"Loading MIM backbone weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    # Handle different saved dict formats (e.g. if saved via torch.save({'model_state_dict': ...}))
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
        
    # We want to load into model.backbone
    # The MIM model has its ViT under self.backbone as well.
    # So keys matching 'backbone.*' should map perfectly.
    translated = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            translated[k] = v
            
    # Resize pos_embed if needed
    if 'backbone.pos_embed' in translated:
        pos_embed = translated['backbone.pos_embed']
        target_pos_embed = model.backbone.pos_embed
        if pos_embed.shape != target_pos_embed.shape:
            print(f"Interpolating pos_embed from {pos_embed.shape} to {target_pos_embed.shape}")
            import torch.nn.functional as F
            num_patches = pos_embed.shape[1]
            orig_size = int(num_patches ** 0.5)
            target_num_patches = target_pos_embed.shape[1]
            target_size = int(target_num_patches ** 0.5)
            
            # reshape to [1, embed_dim, H, W]
            pos_embed = pos_embed.reshape(1, orig_size, orig_size, pos_embed.shape[-1]).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, size=(target_size, target_size), mode="bicubic", align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, target_num_patches, target_pos_embed.shape[-1])
            translated['backbone.pos_embed'] = pos_embed
            
    msg = model.load_state_dict(translated, strict=False)
    print(f"Loaded MIM weights into ViT backbone. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
    return model

def compute_metrics_binary(preds, targets):
    """
    Computes precision, recall, f1, and iou for binary segmentation masks.
    preds: boolean tensor [B, 1, H, W]
    targets: boolean tensor [B, 1, H, W]
    """
    tp = (preds & targets).sum().float()
    fp = (preds & ~targets).sum().float()
    fn = (~preds & targets).sum().float()
    
    precision = tp / torch.clamp(tp + fp, min=1e-6)
    recall = tp / torch.clamp(tp + fn, min=1e-6)
    f1 = 2 * (precision * recall) / torch.clamp(precision + recall, min=1e-6)
    iou = tp / torch.clamp(tp + fp + fn, min=1e-6)
    
    return precision.item(), recall.item(), f1.item(), iou.item()

def train_vit_segmentation(
    data_yaml,
    model_weights=None,
    output_dir="runs/vit_seg",
    epochs=50,
    batch_size=16,
    imgsz=640,
    device="cuda",
    embed_dim=384,
    depth=12,
    num_heads=6,
    patch_size=16,
    pretrained=False,
    num_workers=8
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    
    import logging
    from tqdm import tqdm
    import json
    
    # Set up logging
    logger = logging.getLogger(f"ViT_Finetune_{out_dir.name}")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(out_dir / "training.log")
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.info(f"Starting ViT Fine-tuning. Output dir: {out_dir}")
    logger.info(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, imgsz={imgsz}, num_workers={num_workers}")
    
    with open(out_dir / "args.yaml", "w") as f:
        yaml.dump({
            "data": data_yaml, 
            "model_weights": model_weights, 
            "epochs": epochs, 
            "batch_size": batch_size, 
            "imgsz": imgsz,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "patch_size": patch_size,
            "pretrained": pretrained,
            "num_workers": num_workers
        }, f)
        
    train_ds = YoloFormatSegmentationDataset(data_yaml, split="train", imgsz=imgsz)
    val_ds = YoloFormatSegmentationDataset(data_yaml, split="val", imgsz=imgsz)
    
    logger.info(f"Loaded {len(train_ds)} training images and {len(val_ds)} validation images.")
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Initialize the ViT Segmentation Model
    model = RemoteSensingSegmentationModel(
        in_channels=3,
        image_size=imgsz,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        decoder_hidden_channels=128,
        precision="fp32"
    )
    
    torch.backends.cudnn.benchmark = True
    
    if pretrained:
        logger.info("Loading ImageNet pretrained weights into ViT backbone...")
        model.backbone.load_pretrained_weights()
    elif model_weights and Path(model_weights).exists():
        logger.info(f"Loading MIM weights from {model_weights}")
        model = load_mim_backbone_into_vit(model, model_weights)
    else:
        logger.info("Training from scratch (random initialization).")
        
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # Binary Cross Entropy with Logits
    criterion = nn.BCEWithLogitsLoss()
    
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    # Target an effective batch size of 16
    effective_batch = 16
    accumulation_steps = max(1, effective_batch // batch_size)
    
    results = []
    best_f1 = -1.0
    
    logger.info("Starting training loop...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for i, batch in enumerate(pbar):
            imgs = batch["image"].to(device, non_blocking=True)
            # YoloDataset returns mask.long() where 0 is background, >0 is class
            masks = (batch["mask"].to(device, non_blocking=True) > 0).float().unsqueeze(1)
            
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                out = model(imgs) # output shape is [B, 1, H, W]
                loss = criterion(out, masks)
                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dl):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            step_loss = loss.item() * accumulation_steps
            train_loss += step_loss
            pbar.set_postfix({'loss': f"{step_loss:.4f}"})
            
        train_loss /= max(1, len(train_dl))
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        all_tp, all_fp, all_fn = 0.0, 0.0, 0.0
        
        val_pbar = tqdm(val_dl, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                imgs = batch["image"].to(device)
                masks = (batch["mask"].to(device) > 0).float().unsqueeze(1)
                
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    out = model(imgs)
                    loss = criterion(out, masks)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(out) > 0.5)
                targets_bool = (masks > 0.5)
                
                all_tp += (preds & targets_bool).sum().item()
                all_fp += (preds & ~targets_bool).sum().item()
                all_fn += (~preds & targets_bool).sum().item()
                
        val_loss /= max(1, len(val_dl))
        
        precision = all_tp / max(all_tp + all_fp, 1e-6)
        recall = all_tp / max(all_tp + all_fn, 1e-6)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)
        iou = all_tp / max(all_tp + all_fp + all_fn, 1e-6)
        
        log_msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - F1: {f1:.4f} - IoU: {iou:.4f}"
        logger.info(log_msg)
        
        epoch_results = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "metrics/precision(M)": precision,
            "metrics/recall(M)": recall,
            "metrics/mAP50(M)": iou, # Approximate to YOLO metrics
            "metrics/F1(M)": f1
        }
        results.append(epoch_results)
        
        # Save epoch results to JSON for detailed report writing
        with open(out_dir / "detailed_metrics.json", "w") as f:
            json.dump(results, f, indent=4)
            
        torch.save(model.state_dict(), weights_dir / "last.pt")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), weights_dir / "best.pt")
            logger.info(f"New best model saved with F1: {best_f1:.4f}")
            
    pd.DataFrame(results).to_csv(out_dir / "results.csv", index=False)
    logger.info(f"Training complete. Artifacts saved to {out_dir}")
    
    # Remove handlers to avoid duplicate logs in loops
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()
