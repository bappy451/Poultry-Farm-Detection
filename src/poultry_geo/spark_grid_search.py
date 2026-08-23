from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

import poultry_geo.models as models
import poultry_geo.naip_data as naip_data
import poultry_geo.spark as spark
import poultry_geo.spark_models as spark_models
import poultry_geo.train_spark_ssl as train_spark_ssl
from poultry_geo.checkpoint_manager import StateManager, normalize_training_name
from poultry_geo.grid_search import resolve_runtime_device
from poultry_geo.optimizers import build_optimizer
from poultry_geo.ssl_grid_search import (
    build_scheduler,
    expand_ssl_grid_search_configs,
    get_system_info,
    load_ssl_grid_search_config,
)


def _run_single_experiment(
    experiment: dict[str, Any], normalized_training_name: str, root_dir: Path,
    total_epochs: int, experiment_index: int, total_experiments: int,
    device_name: str, progress_callback_direct: Any | None = None,
) -> dict[str, Any]:
    grid = experiment["grid"]
    backbone_type = str(grid["model_name"])
    signature = json.dumps(experiment, sort_keys=True, separators=(",", ":"))
    experiment_id = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    precision = str(grid["precision"]).strip().lower()
    folder_name = f"{experiment_id}_{backbone_type}_{precision}"
    record = {
        "experiment_id": experiment_id, "training_name": normalized_training_name,
        "backbone_type": backbone_type, "mask_ratio": grid["mask_ratio"],
        "optimizer": grid["optimizer"], "precision": grid["precision"],
        "learning_rate": grid["learning_rate"], "weight_decay": grid["weight_decay"],
        "status": "pending", "reason": "", "last_epoch": 0,
        "final_loss": np.nan, "best_loss": np.nan,
        "run_dir": str(root_dir / normalized_training_name / folder_name),
        "resolved_device": "", "resolved_precision": "",
    }
    try:
        resolved_device = resolve_runtime_device(device_name)
        in_channels = int(experiment["data"].get("n_channels", 3))
        transformer = spark.SparseTransformer(
            models.BandAdapter(in_channels=in_channels, out_channels=3),
            patch_size=int(grid.get("patch_size", 16)), mask_ratio=float(grid["mask_ratio"]),
        ).to(resolved_device)
        model = spark_models.SparKNetwork(
            backbone_type=backbone_type, in_channels=3, pretrained=bool(grid.get("pretrained", False)),
        ).to(resolved_device)
        optimizer = build_optimizer(
            str(grid["optimizer"]),
            [p for p in list(model.parameters()) + list(transformer.parameters()) if p.requires_grad],
            lr=float(grid["learning_rate"]), weight_decay=float(grid.get("weight_decay", 0.05)),
        )
        scheduler = build_scheduler(optimizer, total_epochs=total_epochs,
                                    warmup_epochs=int(grid.get("warmup_epochs", 0)))
        scaler = torch.amp.GradScaler(device="cuda", enabled=resolved_device.type == "cuda" and precision == "fp16")
        manager = StateManager(config=experiment, training_name=normalized_training_name,
                               experiment_id=folder_name, root_dir=root_dir)
        run_state = manager.prepare(model, optimizer, scaler)
        if run_state.scheduler_state:
            scheduler.load_state_dict(run_state.scheduler_state)
        if run_state.status == "resume":
            transformer_state = manager._load_payload().get("transformer_state_dict")
            if transformer_state is None:
                raise RuntimeError("Grid checkpoint lacks transformer state and cannot be resumed safely.")
            transformer.load_state_dict(transformer_state)
        spark_config = train_spark_ssl.SparKConfig(
            mask_ratio=float(grid["mask_ratio"]), backbone_type=backbone_type,
            precision=precision.upper(), lr=float(grid["learning_rate"]),
            batch_size=int(experiment["data"].get("batch_size", 8)),
            patch_size=int(grid.get("patch_size", 16)), device_type=resolved_device.type,
            weight_decay=float(grid.get("weight_decay", 0.05)),
        )
        catalog_path_str = experiment["data"].get("catalog_path")
        if catalog_path_str and not Path(catalog_path_str).exists():
            naip_data.create_dataset_catalog(experiment["data"]["image_dir"], Path(catalog_path_str))
        train_loader, _ = naip_data.get_dataloaders(
            experiment["data"]["image_dir"], batch_size=int(experiment["data"]["batch_size"]),
            crop_size=int(experiment["data"]["image_size"]), channels=in_channels,
            precision=precision, num_workers=int(experiment["data"]["num_workers"]),
            catalog_path=catalog_path_str,
        )
        best_loss = float("inf")
        if manager.metrics_path.exists():
            prior_metrics = pd.read_csv(manager.metrics_path)
            if not prior_metrics.empty:
                best_loss = float(prior_metrics["loss"].min())
        for epoch_index in range(run_state.start_epoch, total_epochs):
            started = time.time()
            avg_loss = _train_one_epoch(
                model, train_loader, optimizer, transformer, spark_config, device=resolved_device,
                scaler=scaler, epoch_index=epoch_index, total_epochs=total_epochs,
                max_batches=grid.get("epoch_size"), progress_callback=progress_callback_direct,
                experiment_index=experiment_index, total_experiments=total_experiments,
            )
            scheduler.step()
            manager.append_metrics(epoch=epoch_index + 1, loss=avg_loss,
                                   lr=optimizer.param_groups[0]["lr"], duration=time.time() - started)
            manager.save_checkpoint(model, optimizer, scaler, epoch=epoch_index, payload={
                "scheduler": scheduler.state_dict(), "transformer_state_dict": transformer.state_dict(),
                "format_version": 2, "backbone_type": backbone_type,
            })

            train_spark_ssl.save_spark_checkpoint(
                model, optimizer, scaler, spark_config, str(manager.run_dir), epoch_index,
                transformer=transformer, schedule_total_epochs=total_epochs,
                schedule_warmup_epochs=int(grid.get("warmup_epochs", 0)),
            )
            if avg_loss <= best_loss:
                best_loss = avg_loss
                shutil.copyfile(manager.run_dir / "spark_unified_latest.pt", manager.run_dir / "spark_unified_best.pt")
                shutil.copyfile(manager.run_dir / "sparse_encoder_latest.pt", manager.run_dir / "sparse_encoder_best.pt")
        metrics = pd.read_csv(manager.metrics_path)
        record.update(status="ok", last_epoch=int(metrics["epoch"].iloc[-1]),
                      final_loss=float(metrics["loss"].iloc[-1]), best_loss=float(metrics["loss"].min()),
                      run_dir=str(manager.run_dir), resolved_device=str(resolved_device),
                      resolved_precision=precision)
    except Exception as exc:
        record.update(status="error", reason=str(exc))
        failure_dir = Path(record["run_dir"])
        failure_dir.mkdir(parents=True, exist_ok=True)
        (failure_dir / "FAILED.json").write_text(json.dumps({
            "status": "error", "reason": str(exc), "experiment_id": experiment_id,
        }, indent=2), encoding="utf-8")
    return record


def run_stateful_spark_grid_search(config_path: str | Path, training_name: str,
                                    epochs: int | None = None, root_dir: str | Path = "trained_models",
                                    progress_callback: Any | None = None) -> pd.DataFrame:
    config = load_ssl_grid_search_config(config_path)
    experiments = expand_ssl_grid_search_configs(config)
    total_epochs = int(epochs if epochs is not None else config["runtime"].get("epochs", 1))
    normalized_training_name = normalize_training_name(training_name)
    campaign_dir = Path(root_dir) / normalized_training_name
    campaign_dir.mkdir(parents=True, exist_ok=True)
    (campaign_dir / "campaign_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (campaign_dir / "system_info.json").write_text(json.dumps(get_system_info(), indent=2), encoding="utf-8")
    records = [
        _run_single_experiment(exp, normalized_training_name, Path(root_dir), total_epochs, index,
                               len(experiments), str(exp["runtime"].get("device", "auto")), progress_callback)
        for index, exp in enumerate(experiments)
    ]
    results = pd.DataFrame.from_records(records)
    results.to_csv(campaign_dir / "grid_search_results.csv", index=False)
    return results


def _train_one_epoch(model: torch.nn.Module, train_loader, optimizer, transformer: torch.nn.Module,
                     spark_config: Any, *, device: torch.device, scaler: torch.amp.GradScaler,
                     epoch_index: int, total_epochs: int, max_batches: int | None = None,
                     progress_callback: Any | None = None, experiment_index: int = 0,
                     total_experiments: int = 1) -> float:
    model.train()
    transformer.train()
    num_batches = min(len(train_loader), int(max_batches)) if max_batches else len(train_loader)
    if num_batches <= 0:
        raise ValueError("SparK training dataloader is empty.")
    loss_sum = 0.0
    for batch_index, batch in enumerate(train_loader, start=1):
        if batch_index > num_batches:
            break
        if progress_callback:
            total_work = total_experiments * total_epochs * num_batches
            completed = experiment_index * total_epochs * num_batches + epoch_index * num_batches + batch_index - 1
            progress_callback(completed, total_work, detail=f"E{epoch_index + 1}/{total_epochs} B{batch_index}/{num_batches} L:{loss_sum / max(batch_index - 1, 1):.4f}")
        image = batch["image"].to(device, non_blocking=True)
        loss_sum += train_spark_ssl.train_one_step(model, optimizer, scaler, transformer, image, spark_config)
    return loss_sum / num_batches



