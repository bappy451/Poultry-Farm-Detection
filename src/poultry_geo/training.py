from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from poultry_geo.ultralytics_runtime import load_ultralytics_model


def select_torch_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_yolo_models(
    models: list[str],
    data_path: str | Path,
    project_dir: str | Path,
    epochs: int = 200,
    batch_size: int = 8,
    image_size: int = 640,
    optimizer: str = "SGD",
    seed: int = 7,
    patience: int = 100,
    device: str | None = None,
    wandb_project: str | None = None,
    extra_train_kwargs: dict[str, Any] | None = None,
    family: str = "auto",
) -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Training requires torch. Install requirements-ml.txt.") from exc

    device = device or select_torch_device()
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    extra_train_kwargs = extra_train_kwargs or {}
    runs: list[dict[str, Any]] = []

    wandb = None
    if wandb_project:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise RuntimeError("wandb logging requested but wandb is not installed.") from exc
        wandb = wandb_module

    for model_name in models:
        if device == "cuda":
            torch.cuda.empty_cache()

        model = load_ultralytics_model(model_name, family=family).to(device)
        run_name = f"{Path(model_name).stem}-bs{batch_size}-ep{epochs}"
        results = model.train(
            data=str(data_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            device=device,
            seed=seed,
            optimizer=optimizer,
            patience=patience,
            project=str(project_dir),
            name=run_name,
            save=True,
            verbose=True,
            **extra_train_kwargs,
        )

        metrics_csv = project_dir / run_name / "results.csv"
        if wandb and metrics_csv.exists():
            wandb.init(project=wandb_project, name=run_name)
            metrics = pd.read_csv(metrics_csv)
            for _, row in metrics.iterrows():
                payload = {key: float(value) for key, value in row.items() if key != "epoch"}
                payload["epoch"] = int(row["epoch"])
                wandb.log(payload)
            wandb.finish()

        runs.append(
            {
                "model_name": model_name,
                "run_name": run_name,
                "project_dir": str(project_dir),
                "metrics_csv": str(metrics_csv),
                "results": results,
            }
        )

    return runs
