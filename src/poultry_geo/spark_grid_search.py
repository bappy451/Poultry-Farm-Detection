import os
import time
import json
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, Optional, List

from poultry_geo.grid_search import load_grid_search_config, expand_grid_search_configs
import poultry_geo.naip_data as naip_data
import poultry_geo.spark as spark
import poultry_geo.spark_models as spark_models
import poultry_geo.train_spark_ssl as train_spark_ssl
import poultry_geo.models as models

def run_stateful_spark_grid_search(
    config_path: str | Path,
    training_name: str,
    epochs: Optional[int] = None,
    root_dir: str = "trained_models",
    progress_callback = None
) -> pd.DataFrame:
    config = load_grid_search_config(config_path)
    experiments = expand_grid_search_configs(config)
    
    campaign_dir = Path(root_dir) / training_name
    campaign_dir.mkdir(parents=True, exist_ok=True)
    
    results_list = []
    
    for i, experiment in enumerate(experiments):
        grid = experiment["grid"]
        run_name = f"run_{i:03d}_{grid['backbone']}_lr{grid['learning_rate']}_mask{grid['mask_ratio']}"
        output_dir = campaign_dir / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[SparK Grid] Starting experiment {i+1}/{len(experiments)}: {run_name}")
        
        # Override epochs if provided
        num_epochs = epochs if epochs is not None else config["runtime"].get("epochs", 10)
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            device = torch.device("mps")

        # Data
        train_loader, _ = naip_data.get_dataloaders(
            config["data"]["root"],
            batch_size=int(grid["batch_size"]),
            crop_size=config["data"].get("crop_size", 224),
            channels=config["data"].get("channels", 3),
            num_workers=config["runtime"].get("num_workers", 4)
        )
        
        # Model
        in_channels = config["data"].get("channels", 3)
        adapter = models.BandAdapter(in_channels=in_channels, out_channels=3)
        transformer = spark.SparseTransformer(
            adapter=adapter, 
            patch_size=grid.get("patch_size", 16), 
            mask_ratio=float(grid["mask_ratio"])
        )
        model = spark_models.SparKNetwork(
            backbone_type=str(grid["backbone"]), 
            in_channels=3
        ).to(device)

        # Optimizer
        weight_decay = float(grid.get("weight_decay", 0.05))
        lr = float(grid["learning_rate"])
        if str(grid.get("optimizer", "adamw")).lower() == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        scaler = torch.cuda.amp.GradScaler(enabled=(str(grid["precision"]).lower() != "fp32"))
        
        spark_config = train_spark_ssl.SparKConfig(
            mask_ratio=float(grid["mask_ratio"]),
            backbone_type=str(grid["backbone"]),
            precision=str(grid["precision"]).upper(),
            lr=float(grid["learning_rate"]),
            batch_size=int(grid["batch_size"])
        )

        start_time = time.time()
        final_loss = 0.0
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                img = batch["image"].to(device)
                loss = train_spark_ssl.train_one_step(
                    model, optimizer, scaler, transformer, img, spark_config
                )
                epoch_loss += loss
            
            final_loss = epoch_loss / len(train_loader)
            print(f"  Epoch {epoch}/{num_epochs} | Avg Loss: {final_loss:.4f}")
            
            # Save checkpoint
            train_spark_ssl.save_spark_checkpoint(model, spark_config, str(output_dir), epoch)
            
            if progress_callback:
                progress_callback(i * num_epochs + epoch, len(experiments) * num_epochs)

        duration = time.time() - start_time
        
        res = {
            "run_name": run_name,
            "backbone": grid["backbone"],
            "learning_rate": grid["learning_rate"],
            "mask_ratio": grid["mask_ratio"],
            "batch_size": grid["batch_size"],
            "precision": grid["precision"],
            "optimizer": grid.get("optimizer", "adamw"),
            "weight_decay": grid.get("weight_decay", 0.05),
            "final_loss": final_loss,
            "duration": duration,
            "output_dir": str(output_dir)
        }
        results_list.append(res)
        
        # Save intermediate results
        pd.DataFrame(results_list).to_csv(campaign_dir / "grid_search_results.csv", index=False)

    return pd.DataFrame(results_list)
