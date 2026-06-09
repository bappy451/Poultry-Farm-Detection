import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from poultry_geo.checkpoint_manager import StateManager, normalize_training_name
from poultry_geo.ssl_grid_search import load_ssl_grid_search_config, expand_ssl_grid_search_configs, build_scheduler, get_system_info
from poultry_geo.grid_search import resolve_runtime_device
from poultry_geo.optimizers import build_optimizer
import poultry_geo.naip_data as naip_data
import poultry_geo.spark as spark
import poultry_geo.spark_models as spark_models
import poultry_geo.train_spark_ssl as train_spark_ssl
import poultry_geo.models as models


def run_stateful_spark_grid_search(
    config_path: str | Path,
    training_name: str,
    epochs: int | None = None,
    root_dir: str | Path = "trained_models",
    progress_callback: Any | None = None,
) -> pd.DataFrame:
    config = load_ssl_grid_search_config(config_path)
    experiments = expand_ssl_grid_search_configs(config)
    total_epochs = int(epochs if epochs is not None else config["runtime"].get("epochs", 1))

    normalized_training_name = normalize_training_name(training_name)
    campaign_dir = Path(root_dir) / normalized_training_name
    campaign_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original config
    with open(campaign_dir / "campaign_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    
    # Log system info
    system_info = get_system_info()
    with open(campaign_dir / "system_info.json", "w", encoding="utf-8") as f:
        json.dump(system_info, f, indent=2)

    records: list[dict[str, Any]] = []
    
    for experiment_index, experiment in enumerate(experiments):
        grid = experiment["grid"]
        signature = json.dumps(experiment, sort_keys=True, separators=(",", ":"))
        experiment_id = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
        backbone_type = str(grid["model_name"])
        precision_str = str(grid["precision"]).strip().lower()
        folder_name = f"{experiment_id}_{backbone_type}_{precision_str}"
        
        record = {
            "experiment_id": experiment_id,
            "training_name": normalized_training_name,
            "backbone_type": backbone_type,
            "embed_dim": grid.get("embed_dim", 0),
            "depth": grid.get("depth", 0),
            "num_heads": grid.get("num_heads", 0),
            "mask_ratio": grid["mask_ratio"],
            "optimizer": grid["optimizer"],
            "precision": grid["precision"],
            "learning_rate": grid["learning_rate"],
            "weight_decay": grid["weight_decay"],
            "status": "pending",
            "reason": "",
            "last_epoch": 0,
            "final_loss": np.nan,
            "best_loss": np.nan,
            "run_dir": str(campaign_dir / folder_name),
            "resolved_device": "",
            "resolved_precision": "",
        }

        try:
            resolved_device = resolve_runtime_device(str(experiment["runtime"].get("device", "auto")))
            precision = precision_str

            # Model setup
            in_channels = int(experiment["data"].get("n_channels", 3))
            adapter = models.BandAdapter(in_channels=in_channels, out_channels=3)
            transformer = spark.SparseTransformer(
                adapter=adapter, 
                patch_size=int(grid.get("patch_size", 16)), 
                mask_ratio=float(grid["mask_ratio"])
            ).to(resolved_device)
            
            model = spark_models.SparKNetwork(
                backbone_type=backbone_type, 
                in_channels=3
            ).to(resolved_device)

            optimizer = build_optimizer(
                str(grid["optimizer"]),
                [p for p in model.parameters() if p.requires_grad],
                lr=float(grid["learning_rate"]),
                weight_decay=float(grid.get("weight_decay", 0.05)),
            )

            scheduler = build_scheduler(
                optimizer, 
                total_epochs=total_epochs, 
                warmup_epochs=int(grid.get("warmup_epochs", 0))
            )

            scaler = torch.amp.GradScaler(
                device="cuda",
                enabled=resolved_device.type == "cuda" and precision == "fp16",
            )

            manager = StateManager(
                config=experiment,
                training_name=normalized_training_name,
                experiment_id=folder_name,
                root_dir=root_dir,
            )
            run_state = manager.prepare(model, optimizer, scaler)
            if hasattr(run_state, "scheduler_state") and run_state.scheduler_state:
                scheduler.load_state_dict(run_state.scheduler_state)

            spark_config = train_spark_ssl.SparKConfig(
                mask_ratio=float(grid["mask_ratio"]),
                backbone_type=backbone_type,
                precision=precision.upper(),
                lr=float(grid["learning_rate"]),
                batch_size=int(experiment["data"].get("batch_size", 8))
            )

            print(f"\n--- Starting SparK Experiment: {experiment_id} ---")
            print(f"{folder_name}")
            print(f"Parameters: backbone={backbone_type}, mask={grid['mask_ratio']}, lr={grid['learning_rate']}, precision={precision}")
            print(f"Directory: {manager.run_dir}")

            train_loader, _ = naip_data.get_dataloaders(
                experiment["data"]["image_dir"],
                batch_size=int(experiment["data"].get("batch_size", 8)),
                crop_size=int(experiment["data"].get("image_size", 224)),
                channels=in_channels,
                precision=precision,
                num_workers=int(experiment["data"].get("num_workers", 4)),
            )

            for epoch_index in range(run_state.start_epoch, total_epochs):
                start_time = time.time()
                epoch_loss = 0.0
                total_batches = 0
                num_batches = len(train_loader)
                
                model.train()
                for batch_index, batch in enumerate(train_loader, start=1):
                    img = batch["image"].to(resolved_device, non_blocking=True)
                    loss_val = train_spark_ssl.train_one_step(
                        model, optimizer, scaler, transformer, img, spark_config
                    )
                    epoch_loss += loss_val
                    total_batches += 1

                    if progress_callback is not None:
                        total_work = len(experiments) * total_epochs * num_batches
                        completed_work = (experiment_index * total_epochs * num_batches) + \
                                         (epoch_index * num_batches) + batch_index
                        detail = f"exp {experiment_index+1}/{len(experiments)} ep {epoch_index+1}/{total_epochs} batch {batch_index}/{num_batches} loss={loss_val:.4f}"
                        progress_callback(completed_work, total_work, detail)
                    
                    if batch_index % 10 == 0:
                        print(f"    batch {batch_index}/{num_batches} - Loss: {loss_val:.6f}")

                avg_loss = epoch_loss / max(total_batches, 1)
                scheduler.step()
                duration = time.time() - start_time
                
                manager.append_metrics(
                    epoch=epoch_index + 1,
                    loss=avg_loss,
                    lr=optimizer.param_groups[0]["lr"],
                    duration=duration,
                )
                
                # Store scheduler state in checkpoint
                payload = manager._load_payload() if manager.checkpoint_path.exists() else {}
                payload["scheduler"] = scheduler.state_dict()
                manager.save_checkpoint(model, optimizer, scaler, epoch=epoch_index, payload=payload)
                
                # Save SparK-specific dense backbone for convenience
                train_spark_ssl.save_spark_checkpoint(model, spark_config, str(manager.run_dir), epoch_index)
                
                print(f"  Epoch {epoch_index + 1}/{total_epochs} - Loss: {avg_loss:.6f} - LR: {optimizer.param_groups[0]['lr']:.2e}")

            metrics_frame = pd.read_csv(manager.metrics_path)
            record["status"] = "ok"
            record["last_epoch"] = int(metrics_frame["epoch"].iloc[-1])
            record["final_loss"] = float(metrics_frame["loss"].iloc[-1])
            record["best_loss"] = float(metrics_frame["loss"].min())
            record["run_dir"] = str(manager.run_dir)
            record["resolved_device"] = str(resolved_device)
            record["resolved_precision"] = precision
            print(f"Experiment {experiment_id} finished. Best Loss: {record['best_loss']:.6f}\n")
            
        except Exception as exc:
            record["status"] = "error"
            record["reason"] = str(exc)
            import traceback
            traceback.print_exc()
            print(f"Experiment {experiment_id} failed: {exc}\n")
            
        records.append(record)

    results = pd.DataFrame.from_records(records)
    results.to_csv(campaign_dir / "grid_search_results.csv", index=False)
    return results
