from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from poultry_geo.checkpoint_manager import StateManager, normalize_training_name
from poultry_geo.grid_search import (
    build_segmentation_dataloader,
    expand_grid_search_configs,
    load_grid_search_config,
    resolve_runtime_device,
)
from poultry_geo.models import SUPPORTED_PRECISIONS, UnifiedRSModel
from poultry_geo.optimizers import build_optimizer


def _run_single_experiment(
    experiment: dict[str, Any],
    training_name: str,
    root_dir: Path,
    total_epochs: int,
    experiment_index: int,
    total_experiments: int,
    device_name: str,
    progress_callback_direct: Any | None = None,
) -> dict[str, Any]:
    grid = experiment["grid"]
    experiment_id = _build_experiment_id(experiment)
    record = {
        "experiment_id": experiment_id,
        "training_name": training_name,
        "backbone": grid["backbone"],
        "optimizer": grid["optimizer"],
        "precision": grid["precision"],
        "learning_rate": grid["learning_rate"],
        "weight_decay": grid["weight_decay"],
        "freeze_backbone": grid["freeze_backbone"],
        "use_lora": grid["use_lora"],
        "status": "pending",
        "reason": "",
        "last_epoch": 0,
        "final_loss": np.nan,
        "best_loss": np.nan,
        "run_dir": str(root_dir / training_name / experiment_id),
        "resolved_device": "",
        "resolved_precision": "",
    }

    try:
        if str(grid["backbone"]) == "resnet50" and bool(grid["use_lora"]):
            record["status"] = "skipped"
            record["reason"] = "LoRA is only supported for ViT backbones."
            return record

        resolved_device = resolve_runtime_device(device_name)
        precision = str(grid["precision"]).strip().lower()
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(f"precision must be one of: {', '.join(SUPPORTED_PRECISIONS)}")
        if precision == "fp16" and resolved_device.type != "cuda":
            raise ValueError("fp16 training requires a CUDA device.")

        model = UnifiedRSModel(
            n_channels=int(experiment["data"]["n_channels"]),
            backbone_name=str(grid["backbone"]),
            image_size=int(experiment["data"]["image_size"]),
            precision=precision,
            pretrained=bool(experiment["runtime"]["pretrained"]),
            freeze_backbone=bool(grid["freeze_backbone"]),
            use_lora=bool(grid["use_lora"]),
        ).to(resolved_device)

        optimizer = build_optimizer(
            str(grid["optimizer"]),
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(grid["learning_rate"]),
            weight_decay=float(grid["weight_decay"]),
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=resolved_device.type == "cuda" and precision == "fp16",
        )

        loader = build_segmentation_dataloader(experiment["data"])
        manager = StateManager(
            config=experiment,
            training_name=training_name,
            experiment_id=experiment_id,
            root_dir=root_dir,
        )
        run_state = manager.prepare(model, optimizer, scaler)
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch_index in range(run_state.start_epoch, total_epochs):
            metrics = _train_one_epoch(
                model,
                loader,
                optimizer,
                criterion,
                device=resolved_device,
                precision=precision,
                scaler=scaler,
                epoch_index=epoch_index,
                progress_callback=progress_callback_direct,
                experiment_index=experiment_index,
                total_experiments=total_experiments,
                total_epochs=total_epochs,
            )
            manager.append_metrics(
                epoch=epoch_index + 1,
                loss=metrics["loss"],
                accuracy=metrics["accuracy"],
                evaluation_matrix=metrics["evaluation_matrix"],
            )
            manager.save_checkpoint(model, optimizer, scaler, epoch=epoch_index)

        metrics_frame = pd.read_csv(manager.metrics_path)
        record["status"] = "ok"
        record["last_epoch"] = int(metrics_frame["epoch"].iloc[-1])
        record["final_loss"] = float(metrics_frame["loss"].iloc[-1])
        record["best_loss"] = float(metrics_frame["loss"].min())
        record["resolved_device"] = str(resolved_device)
        record["resolved_precision"] = precision
    except Exception as exc:
        record["status"] = "error"
        record["reason"] = str(exc)

    return record


def run_stateful_grid_search(
    config_path: str | Path = "config.yaml",
    training_name: str = "default-training",
    *,
    epochs: int | None = None,
    root_dir: str | Path = "trained_models",
    progress_callback: Any | None = None,
) -> pd.DataFrame:
    config = load_grid_search_config(config_path)
    experiments = expand_grid_search_configs(config)
    total_epochs = int(epochs if epochs is not None else config["runtime"].get("epochs", 1))
    if total_epochs <= 0:
        raise ValueError("epochs must be a positive integer.")

    normalized_training_name = normalize_training_name(training_name)
    campaign_dir = Path(root_dir) / normalized_training_name
    campaign_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    # Sequential execution
    for experiment_index, experiment in enumerate(experiments):
        record = _run_single_experiment(
            experiment,
            normalized_training_name,
            Path(root_dir),
            total_epochs,
            experiment_index,
            len(experiments),
            str(experiment["runtime"]["device"]),
            progress_callback_direct=progress_callback,
        )
        records.append(record)

    results = pd.DataFrame.from_records(records)
    results.to_csv(campaign_dir / "grid_search_results.csv", index=False)
    return results


def _train_one_epoch(
    model: UnifiedRSModel,
    loader,
    optimizer,
    criterion,
    *,
    device: torch.device,
    precision: str,
    scaler,
    epoch_index: int,
    progress_callback: Any | None = None,
    experiment_index: int = 0,
    total_experiments: int = 1,
    total_epochs: int = 1,
) -> dict[str, Any]:
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch_index)

    model.train()
    total_loss = 0.0
    total_batches = 0
    correct_pixels = 0
    total_pixels = 0
    tn = fp = fn = tp = 0
    num_batches = len(loader)
    last_loss = 0.0
    last_acc = 0.0

    for batch_index, batch in enumerate(loader, start=1):
        if progress_callback is not None:
            num_batches_safe = max(num_batches, 1)
            total_work = total_experiments * total_epochs * num_batches_safe
            completed_work = (experiment_index * total_epochs * num_batches_safe) + (epoch_index * num_batches_safe) + batch_index - 1

            avg_loss = total_loss / max(total_batches, 1)
            avg_acc = correct_pixels / max(total_pixels, 1)
            lr = optimizer.param_groups[0]["lr"]

            detail = (
                f"exp {experiment_index+1}/{total_experiments} | "
                f"ep {epoch_index+1}/{total_epochs} | "
                f"batch {batch_index}/{num_batches_safe} | "
                f"loss {last_loss:.4f} (avg {avg_loss:.4f}) | "
                f"acc {last_acc:.4f} (avg {avg_acc:.4f}) | "
                f"lr {lr:.6f}"
            )
            progress_callback(completed_work, total_work, detail=detail)

        images = batch["image"].to(device=device, non_blocking=True)
        masks = batch["mask"].to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, precision):
            logits = model(images)
            loss = criterion(logits, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError("Encountered a non-finite segmentation loss.")

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _sanitize_gradients(model.parameters())
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            _sanitize_gradients(model.parameters())
            optimizer.step()

        current_loss = float(loss.detach().cpu())
        total_loss += current_loss
        total_batches += 1
        predictions = torch.sigmoid(logits) >= 0.5
        targets = masks >= 0.5
        batch_correct = int((predictions == targets).sum().item())
        batch_pixels = int(targets.numel())
        correct_pixels += batch_correct
        total_pixels += batch_pixels
        
        last_loss = current_loss
        last_acc = batch_correct / batch_pixels

        tn += int((~predictions & ~targets).sum().item())
        fp += int((predictions & ~targets).sum().item())
        fn += int((~predictions & targets).sum().item())
        tp += int((predictions & targets).sum().item())

        if batch_index % 10 == 0:
            print(f"    batch {batch_index}/{num_batches} - Loss: {current_loss:.6f}")

    return {
        "loss": total_loss / max(total_batches, 1),
        "accuracy": correct_pixels / max(total_pixels, 1),
        "evaluation_matrix": [[tn, fp], [fn, tp]],
    }


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp16":
        return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)
    if precision == "bf16":
        enabled = device.type in {"cpu", "cuda"}
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)
    return nullcontext()


def _sanitize_gradients(parameters) -> None:
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            parameter.grad = torch.nan_to_num(parameter.grad, nan=0.0, posinf=0.0, neginf=0.0)


def _build_experiment_id(experiment: dict[str, Any]) -> str:
    signature = json.dumps(experiment, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run stateful resume-aware segmentation training.")
    parser.add_argument("training_name", help="Run identifier or 'training_name=<name>'.")
    parser.add_argument("--config", default="config.yaml", help="Path to the grid-search YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override runtime.epochs from the config.")
    parser.add_argument("--root-dir", default="trained_models", help="Root directory for saved runs.")
    args = parser.parse_args(argv)

    results = run_stateful_grid_search(
        config_path=args.config,
        training_name=args.training_name,
        epochs=args.epochs,
        root_dir=args.root_dir,
    )
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
