from __future__ import annotations

import argparse
import hashlib
import json
import itertools
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

import platform
import time
from poultry_geo.checkpoint_manager import StateManager, normalize_training_name
from poultry_geo.grid_search import resolve_runtime_device
from poultry_geo.naip_data import get_dataloaders
from poultry_geo.ssl_models import SUPPORTED_PRECISIONS, RemoteSensingMIMModel, VIT_CONFIGS
from poultry_geo.optimizers import build_optimizer


SSL_GRID_KEYS = (
    "model_name",
    "use_lora",
    "freeze_backbone",
    "mask_ratio",
    "drop_path_rate",
    "optimizer",
    "precision",
    "learning_rate",
    "weight_decay",
    "warmup_epochs",
)


def get_system_info() -> dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def build_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int):
    """
    Builds a Linear Warmup + Cosine Decay scheduler.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_ssl_grid_search_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"SSL Config file does not exist: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("ssl_config.yaml must define a mapping at the top level.")

    for section in ("data", "runtime", "grid_search"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"ssl_config.yaml must define a '{section}' mapping.")

    data = config["data"]
    runtime = config["runtime"]
    grid = config["grid_search"]
    
    for key in ("image_dir", "image_size", "n_channels", "batch_size", "num_workers"):
        if key not in data:
            raise ValueError(f"ssl_config.yaml is missing data.{key}")
            
    for key in ("device", "seed", "output_dir"):
        if key not in runtime:
            raise ValueError(f"ssl_config.yaml is missing runtime.{key}")

    missing_grid = [key for key in SSL_GRID_KEYS if key not in grid]
    if missing_grid:
        raise ValueError(f"ssl_config.yaml is missing grid_search keys: {', '.join(missing_grid)}")
        
    for key in SSL_GRID_KEYS:
        if not isinstance(grid[key], list) or len(grid[key]) == 0:
            raise ValueError(f"grid_search.{key} must be a non-empty list.")
            
    return config


def expand_ssl_grid_search_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    data = dict(config["data"])
    runtime = dict(config["runtime"])
    grid = config["grid_search"]
    experiments: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    
    for values in itertools.product(*(grid[key] for key in SSL_GRID_KEYS)):
        grid_values = dict(zip(SSL_GRID_KEYS, values, strict=True))
        experiment = {
            "data": dict(data),
            "runtime": dict(runtime),
            "grid": grid_values,
        }
        signature = json.dumps(experiment, sort_keys=True, separators=(",", ":"))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        experiments.append(experiment)
    return experiments


def run_stateful_ssl_grid_search(
    config_path: str | Path = "ssl_config.yaml",
    training_name: str = "default-ssl",
    *,
    epochs: int | None = None,
    root_dir: str | Path = "trained_models",
    progress_callback: Any | None = None,
) -> pd.DataFrame:
    config = load_ssl_grid_search_config(config_path)
    experiments = expand_ssl_grid_search_configs(config)
    total_epochs = int(epochs if epochs is not None else config["runtime"].get("epochs", 1))
    if total_epochs <= 0:
        raise ValueError("epochs must be a positive integer.")

    normalized_training_name = normalize_training_name(training_name)
    campaign_dir = Path(root_dir) / normalized_training_name
    campaign_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the original config for reference
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
        model_name = str(grid["model_name"]).upper()
        precision_str = str(grid["precision"]).strip().lower()
        folder_name = f"{experiment_id}_{model_name}_{precision_str}"
        
        record = {
            "experiment_id": experiment_id,
            "training_name": normalized_training_name,
            "model_name": model_name,
            "use_lora": grid["use_lora"],
            "freeze_backbone": grid["freeze_backbone"],
            "mask_ratio": grid["mask_ratio"],
            "drop_path_rate": grid["drop_path_rate"],
            "optimizer": grid["optimizer"],
            "precision": grid["precision"],
            "learning_rate": grid["learning_rate"],
            "weight_decay": grid["weight_decay"],
            "warmup_epochs": grid["warmup_epochs"],
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
            resolved_device = resolve_runtime_device(str(experiment["runtime"]["device"]))    
            precision = str(grid["precision"]).strip().lower()
            if precision not in SUPPORTED_PRECISIONS:
                raise ValueError(f"precision must be one of: {', '.join(SUPPORTED_PRECISIONS)}")

            vit_params = VIT_CONFIGS.get(model_name, VIT_CONFIGS["B"])
            model = RemoteSensingMIMModel(
                in_channels=int(experiment["data"]["n_channels"]),
                image_size=int(experiment["data"]["image_size"]),
                patch_size=16,
                embed_dim=vit_params["embed_dim"],
                depth=vit_params["depth"],
                num_heads=vit_params["num_heads"],
                precision=precision,
                mask_ratio=float(grid["mask_ratio"]),
                use_lora=bool(grid["use_lora"]),
                freeze_backbone=bool(grid["freeze_backbone"]),
                drop_path_rate=float(grid["drop_path_rate"]),
            ).to(resolved_device)
            
            optimizer = build_optimizer(
                str(grid["optimizer"]),
                [parameter for parameter in model.parameters() if parameter.requires_grad],   
                lr=float(grid["learning_rate"]),
                weight_decay=float(grid["weight_decay"]),
            )
            
            scheduler = build_scheduler(
                optimizer, 
                total_epochs=total_epochs, 
                warmup_epochs=int(grid["warmup_epochs"])
            )

            scaler = torch.cuda.amp.GradScaler(
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

            print(f"\n--- Starting Experiment: {experiment_id} ---")
            print(f"{folder_name}")
            print(f"Parameters: model={model_name}, lora={grid['use_lora']}, freeze={grid['freeze_backbone']}, dpr={grid['drop_path_rate']}, lr={grid['learning_rate']}")
            print(f"Directory: {manager.run_dir}")

            train_loader, _ = get_dataloaders(
                experiment["data"]["image_dir"],
                batch_size=int(experiment["data"]["batch_size"]),
                val_fraction=0.01,
                seed=int(experiment["runtime"].get("seed", 27)),
                crop_size=int(experiment["data"]["image_size"]),
                channels=int(experiment["data"]["n_channels"]),
                precision=precision,
                num_workers=int(experiment["data"]["num_workers"]),
            )

            for epoch_index in range(run_state.start_epoch, total_epochs):
                start_time = time.time()
                loss = _train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device=resolved_device,
                    precision=precision,
                    scaler=scaler,
                    epoch_index=epoch_index,
                    progress_callback=progress_callback,
                    experiment_index=experiment_index,
                    total_experiments=len(experiments),
                    total_epochs=total_epochs,
                )
                scheduler.step()
                duration = time.time() - start_time
                
                manager.append_metrics(
                    epoch=epoch_index + 1,
                    loss=loss,
                    accuracy=0.0,
                    evaluation_matrix=[[0,0], [0,0]],
                    lr=optimizer.param_groups[0]["lr"],
                    duration=duration,
                )
                # Store scheduler state in checkpoint
                payload = manager._load_payload() if manager.checkpoint_path.exists() else {}
                payload["scheduler"] = scheduler.state_dict()
                manager.save_checkpoint(model, optimizer, scaler, epoch=epoch_index, payload=payload)
                print(f"  Epoch {epoch_index + 1}/{total_epochs} - Loss: {loss:.6f} - LR: {optimizer.param_groups[0]['lr']:.2e}")

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
    
    successful_runs = results[results["status"] == "ok"]
    if not successful_runs.empty:
        best_run = successful_runs.loc[successful_runs["best_loss"].idxmin()]
        print("\n=======================================================")
        print(" BEST MODEL FOUND ")
        print("=======================================================")
        print(f"Experiment ID : {best_run['experiment_id']}")
        print(f"Run Directory : {best_run['run_dir']}")
        print(f"Best Loss     : {best_run['best_loss']:.6f}")
        print("-------------------------------------------------------")
        print("Winning Parameters:")
        print(f"  model_name    = {best_run['model_name']}")
        print(f"  use_lora      = {best_run['use_lora']}")
        print(f"  freeze        = {best_run['freeze_backbone']}")
        print(f"  optimizer     = {best_run['optimizer']}")
        print(f"  learning_rate = {best_run['learning_rate']}")
        print(f"  weight_decay  = {best_run['weight_decay']}")
        print("=======================================================\n")
    else:
        print("\nNo experiments completed successfully to determine a best model.\n")

    return results


def _train_one_epoch(
    model: RemoteSensingMIMModel,
    loader,
    optimizer,
    *,
    device: torch.device,
    precision: str,
    scaler,
    epoch_index: int,
    progress_callback: Any | None = None,
    experiment_index: int = 0,
    total_experiments: int = 1,
    total_epochs: int = 1,
) -> float:
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch_index)

    model.train()
    total_loss = 0.0
    total_batches = 0
    num_batches = len(loader)

    if precision == "fp16":
        ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)
    elif precision == "bf16":
        ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in {"cpu", "cuda"})
    else:
        ctx = nullcontext()

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        with ctx:
            loss = model(images)
            
        if not torch.isfinite(loss):
            raise FloatingPointError("Encountered a non-finite SSL loss.")

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    p.grad = torch.nan_to_num(p.grad, nan=0.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    p.grad = torch.nan_to_num(p.grad, nan=0.0)
            optimizer.step()

        current_loss = float(loss.detach().cpu())
        total_loss += current_loss
        total_batches += 1

        if progress_callback is not None:
            # overall progress calculation
            # experiments -> epochs -> batches
            total_work = total_experiments * total_epochs * num_batches
            completed_work = (experiment_index * total_epochs * num_batches) + \
                             (epoch_index * num_batches) + batch_index
            detail = f"exp {experiment_index+1}/{total_experiments} ep {epoch_index+1}/{total_epochs} batch {batch_index}/{num_batches} loss={current_loss:.4f}"
            progress_callback(completed_work, total_work, detail)
        
        if batch_index % 10 == 0:
            print(f"    batch {batch_index}/{num_batches} - Loss: {current_loss:.6f}")

    return total_loss / max(total_batches, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stateful resume-aware SSL grid search.")
    parser.add_argument("training_name", help="Run identifier.")
    parser.add_argument("--config", default="ssl_config.yaml", help="Path to the grid-search YAML config.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--root-dir", default="trained_models")
    args = parser.parse_args()

    results = run_stateful_ssl_grid_search(
        config_path=args.config,
        training_name=args.training_name,
        epochs=args.epochs,
        root_dir=args.root_dir,
    )
    print(results.to_string(index=False))
