"""
ViDT+ Standalone Training Pipeline for Poultry House Dataset
===================================================
A clean, streamlined training script to fine-tune a ViDT+ model
exclusively on the Poultry House dataset using DINO pretrained weights.
"""
from __future__ import annotations

import os
import time
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from poultry_geo.vidt_models import RemoteSensingViDTPlus
from poultry_geo.vidt_datasets import YoloFormatDetectionDataset
from poultry_geo.vidt_loss import HungarianMatcher, SetCriterion
from poultry_geo.progress import CommandProgress
from rich.console import Console

def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    targets = [b["target"] for b in batch]
    return {"image": images, "mask": targets}

def resolve_dino_checkpoint(path_str: str | Path) -> Path:
    target_path = Path(path_str).expanduser().resolve()
    if target_path.is_file():
        return target_path
    if target_path.is_dir():
        backbone_files = list(target_path.glob("backbone_dino_ep*.pt"))
        if backbone_files:
            def _extract_ep(p: Path) -> int:
                try:
                    return int(p.stem.replace("backbone_dino_ep", ""))
                except ValueError:
                    return -1
            backbone_files.sort(key=_extract_ep, reverse=True)
            return backbone_files[0]
        for candidate in ("last.pt", "best.pt"):
            cand_path = target_path / candidate
            if cand_path.exists():
                return cand_path
        unified_files = list(target_path.glob("dino_unified_ep*.pt"))
        if unified_files:
            def _extract_ep_uni(p: Path) -> int:
                try:
                    return int(p.stem.replace("dino_unified_ep", ""))
                except ValueError:
                    return -1
            unified_files.sort(key=_extract_ep_uni, reverse=True)
            return unified_files[0]
    raise FileNotFoundError(f"Could not resolve a valid DINO checkpoint in: {target_path}")

def load_dino_backbone_into_vit(model: RemoteSensingViDTPlus, checkpoint_path: str | Path) -> RemoteSensingViDTPlus:
    ckpt_file = resolve_dino_checkpoint(checkpoint_path)
    print(f"[DINO Loader] Loading checkpoint from {ckpt_file}...")
    state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    if isinstance(state_dict, dict):
        if "student_backbone" in state_dict:
            state_dict = state_dict["student_backbone"]
        elif "teacher_backbone" in state_dict:
            state_dict = state_dict["teacher_backbone"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

    translated = {}
    num_body_blocks = len(model.body_blocks)
    for k, v in state_dict.items():
        key = k
        if key.startswith("backbone."):
            key = key[len("backbone."):]
        if key.startswith("module."):
            key = key[len("module."):]
        if key in ("cls_token", "dist_token") or key.startswith("head."):
            continue

        if key.startswith("blocks."):
            parts = key.split(".")
            block_idx = int(parts[1])
            if block_idx < num_body_blocks:
                new_key = f"body_blocks.{block_idx}." + ".".join(parts[2:])
            else:
                ram_idx = block_idx - num_body_blocks
                new_key = f"ram_blocks.{ram_idx}." + ".".join(parts[2:])
            translated[new_key] = v
        else:
            translated[f"backbone.{key}"] = v

    if "backbone.pos_embed" in translated:
        pos_embed = translated["backbone.pos_embed"]
        target_pos_embed = model.backbone.pos_embed
        if pos_embed.shape != target_pos_embed.shape:
            print(f"[DINO Loader] Interpolating pos_embed from {pos_embed.shape} to {target_pos_embed.shape}")
            num_patches = pos_embed.shape[1]
            orig_size = int(math.sqrt(num_patches))
            target_num_patches = target_pos_embed.shape[1]
            target_size = int(math.sqrt(target_num_patches))

            pos_embed_2d = pos_embed.reshape(1, orig_size, orig_size, pos_embed.shape[-1]).permute(0, 3, 1, 2)
            pos_embed_2d = F.interpolate(pos_embed_2d, size=(target_size, target_size), mode="bicubic", align_corners=False)
            pos_embed = pos_embed_2d.permute(0, 2, 3, 1).reshape(1, target_num_patches, target_pos_embed.shape[-1])
            translated["backbone.pos_embed"] = pos_embed

    msg = model.load_state_dict(translated, strict=False)
    print(f"[DINO Loader] Weights loaded into ViDT+. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
    return model


def run_training(config_path: str | Path, dry_run: bool = False):
    if os.name == "nt":
        os.system("color")
    console = Console(force_terminal=True)
    cfg_file = Path(config_path).expanduser().resolve()
    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths_cfg = config.get("paths", {})
    training_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})

    dino_checkpoint_dir = paths_cfg.get("dino_checkpoint_dir", "runs/dino_pretrain")
    output_dir = Path(paths_cfg.get("output_dir", "runs/vidt_poultry_standalone"))
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    epochs = 1 if dry_run else int(training_cfg.get("epochs", 300))
    batch_size = int(training_cfg.get("batch", 8))
    accum_steps = int(training_cfg.get("accum_steps", 2))
    imgsz = int(training_cfg.get("imgsz", 640))
    lr = float(training_cfg.get("lr", 5e-5))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    num_workers = int(training_cfg.get("num_workers", 4))
    device_str = training_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    num_det_tokens = int(model_cfg.get("num_det_tokens", 300))

    ds_name = dataset_cfg.get("name", "poultry_houses")
    ds_yaml = dataset_cfg.get("yaml", "configs/poultry_obb.yaml")
    if Path(ds_yaml).exists():
        ds_yaml_path = Path(ds_yaml).resolve()
    elif (cfg_file.parent / ds_yaml).exists():
        ds_yaml_path = (cfg_file.parent / ds_yaml).resolve()
    else:
        raise FileNotFoundError(f"Could not find dataset config file: {ds_yaml}")

    console.print("\n[bold bright_cyan]========================================================================[/]")
    console.print(f"[bold bright_cyan]          Standalone ViDT+ Fine-Tuning: {ds_name}          [/]")
    console.print("[bold bright_cyan]========================================================================[/]")
    console.print(f"[dim]Output Directory:[/] [yellow]{output_dir}[/]")
    console.print(f"[dim]Device:[/] [green]{device}[/] | [dim]Epochs:[/] [green]{epochs}[/]{' [bold magenta](DRY RUN)[/]' if dry_run else ''}")

    progress = CommandProgress("vidt-standalone", verbose=True)
    progress.start()

    best_f1 = -1.0
    best_iou = -1.0
    start_epoch = 1

    last_ckpt_path = weights_dir / "last.pt"
    ckpt_data = None
    if last_ckpt_path.exists() and not dry_run:
        try:
            ckpt_data = torch.load(last_ckpt_path, map_location=device, weights_only=False)
            if isinstance(ckpt_data, dict) and "epoch" in ckpt_data:
                saved_epoch = int(ckpt_data["epoch"])
                if saved_epoch >= epochs:
                    console.print(f"[bold bright_green]  Already completed {saved_epoch}/{epochs} epochs. Exiting![/]")
                    return
        except Exception as e:
            console.print(f"[yellow]  Could not check {last_ckpt_path} ({e}).[/]")
            ckpt_data = None

    # Prepare dataloaders
    train_ds = YoloFormatDetectionDataset(str(ds_yaml_path), split="train", imgsz=imgsz)
    val_ds = YoloFormatDetectionDataset(str(ds_yaml_path), split="val", imgsz=imgsz)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=device.type == "cuda", drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=device.type == "cuda")

    model = RemoteSensingViDTPlus(
        in_channels=3,
        image_size=imgsz,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=train_ds.nc,
        num_det_tokens=num_det_tokens,
        ram_depth=2,
    )

    if model_cfg.get("init_strategy") == "dino":
        model = load_dino_backbone_into_vit(model, dino_checkpoint_dir)
    else:
        console.print(f"[dim]Initializing with random weights from scratch.[/]")

    model = model.to(device)

    # Split learning rates: tiny LR for pretrained backbone, normal LR for DETR decoder
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n or "body_blocks" in n or "ram_blocks" in n]
    head_params = [p for n, p in model.named_parameters() if not ("backbone" in n or "body_blocks" in n or "ram_blocks" in n)]
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr}
    ], weight_decay=weight_decay)

    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0, 'loss_mask': 1.0, 'loss_dice': 1.0}
    criterion = SetCriterion(num_classes=train_ds.nc, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=['labels', 'boxes', 'masks'])
    criterion = criterion.to(device)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if ckpt_data and isinstance(ckpt_data, dict) and "epoch" in ckpt_data:
        try:
            saved_epoch = int(ckpt_data["epoch"])
            model.load_state_dict(ckpt_data["model_state"])
            optimizer.load_state_dict(ckpt_data["optimizer_state"])
            if "scaler_state" in ckpt_data:
                scaler.load_state_dict(ckpt_data["scaler_state"])
            start_epoch = saved_epoch + 1
            best_f1 = float(ckpt_data.get("best_f1", -1.0))
            best_iou = float(ckpt_data.get("best_iou", -1.0))
            console.print(f"[bold bright_green]  Smart Resume: continuing from epoch {start_epoch}/{epochs} (Best F1: {best_f1:.2%})[/]")
        except Exception as e:
            console.print(f"[yellow]  Could not resume weights ({e}). Starting from epoch 1.[/]")

    total_batches_per_epoch = 1 if dry_run else max(1, len(train_dl))
    total_exp_steps = epochs * total_batches_per_epoch
    completed_exp_steps = (start_epoch - 1) * total_batches_per_epoch

    progress.update(
        completed=completed_exp_steps,
        total=total_exp_steps,
        description=f"{ds_name}",
        detail=f"Starting Epoch {start_epoch}/{epochs}...",
    )

    try:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            for i, batch in enumerate(train_dl):
                imgs = batch["image"].to(device, non_blocking=True)
                masks = []
                for m in batch["mask"]:
                    masks.append({
                        "boxes": m["boxes"].to(device, non_blocking=True),
                        "labels": m["labels"].to(device, non_blocking=True),
                        "masks": m["masks"].to(device, non_blocking=True)
                    })

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    out = model(imgs)
                    loss_dict = criterion(out, masks)
                    loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
                    scaled_loss = loss / accum_steps

                scaler.scale(scaled_loss).backward()

                if (i + 1) % accum_steps == 0 or (i + 1) == len(train_dl):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item()
                completed_exp_steps = (epoch - 1) * total_batches_per_epoch + i + 1
                detail_str = (
                    f"E{epoch}/{epochs} B{i+1}/{total_batches_per_epoch} "
                    f"Loss:{loss.item():.4f} (Avg:{train_loss/max(1, i+1):.4f}) "
                    f"BestF1:{max(best_f1, 0.0):.1%} IoU:{max(best_iou, 0.0):.1%}"
                )
                progress.update(
                    completed=completed_exp_steps,
                    total=total_exp_steps,
                    description=f"{ds_name}",
                    detail=detail_str,
                )
                if dry_run:
                    break

            train_loss /= max(1, 1 if dry_run else len(train_dl))

            # Validation loop
            model.eval()
            val_loss = 0.0
            all_tp, all_fp, all_fn = 0.0, 0.0, 0.0

            with torch.no_grad():
                for batch in val_dl:
                    imgs = batch["image"].to(device, non_blocking=True)
                    masks = []
                    for m in batch["mask"]:
                        masks.append({
                            "boxes": m["boxes"].to(device, non_blocking=True),
                            "labels": m["labels"].to(device, non_blocking=True),
                            "masks": m["masks"].to(device, non_blocking=True)
                        })

                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        out = model(imgs)
                        loss_dict = criterion(out, masks)
                        v_loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)

                    val_loss += v_loss.item()
                    
                    pred_logits = out['pred_logits']
                    pred_masks = out['pred_masks']
                    
                    for b in range(imgs.shape[0]):
                        probs = pred_logits[b].softmax(-1)
                        scores, labels = probs[:, :-1].max(-1)
                        keep = scores > 0.5
                        
                        if keep.any():
                            pm = pred_masks[b, keep].unsqueeze(1)
                            import torch.nn.functional as F
                            pm = F.interpolate(pm, size=(imgs.shape[2], imgs.shape[3]), mode='bilinear', align_corners=False)
                            p_masks = (pm.squeeze(1).sigmoid() > 0.5).any(dim=0)
                        else:
                            p_masks = torch.zeros((imgs.shape[2], imgs.shape[3]), dtype=torch.bool, device=device)
                        
                        t_masks = (masks[b]['masks'] > 0.5).any(dim=0).to(device) if len(masks[b]['masks']) > 0 else torch.zeros_like(p_masks)
                        
                        all_tp += (p_masks & t_masks).sum().item()
                        all_fp += (p_masks & ~t_masks).sum().item()
                        all_fn += (~p_masks & t_masks).sum().item()

                    if dry_run:
                        break

            val_loss /= max(1, 1 if dry_run else len(val_dl))
            precision = all_tp / max(all_tp + all_fp, 1e-6)
            recall = all_tp / max(all_tp + all_fn, 1e-6)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)
            iou = all_tp / max(all_tp + all_fp + all_fn, 1e-6)

            ckpt_dict = {
                "epoch": epoch,
                "epochs": epochs,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_f1": best_f1,
                "best_iou": best_iou,
            }

            if f1 > best_f1:
                best_f1 = f1
                best_iou = iou
                ckpt_dict.update({"best_f1": best_f1, "best_iou": best_iou})
                if not dry_run:
                    torch.save(ckpt_dict, weights_dir / "best.pt")

            if not dry_run:
                torch.save(ckpt_dict, weights_dir / "last.pt")

            detail_str = (
                f"E{epoch}/{epochs} Done | "
                f"ValLoss:{val_loss:.4f} F1:{f1:.1%} IoU:{iou:.1%} BestF1:{max(best_f1, 0.0):.1%}"
            )
            progress.update(
                completed=epoch * total_batches_per_epoch,
                total=total_exp_steps,
                description=f"{ds_name}",
                detail=detail_str,
            )

        progress.finish(success=True)
    except BaseException as e:
        progress.finish(success=False)
        import traceback
        traceback.print_exc()
        raise

    console.print(f"\n[bold bright_green]Training completed successfully![/] [dim]Results saved to:[/] [yellow]{output_dir}[/]")
