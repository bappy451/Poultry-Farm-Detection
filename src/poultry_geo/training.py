from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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


@dataclass(slots=True)
class SSLTrainingSummary:
    output_dir: str
    epochs: int
    train_batches: int
    val_batches: int
    final_train_loss: float | None
    final_val_loss: float | None


def _move_ssl_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    moved = dict(batch)
    image = moved.get("image")
    if hasattr(image, "to"):
        moved["image"] = image.to(device=device, non_blocking=True)
    return moved


class SSLTrainer:
    def __init__(
        self,
        model,
        train_loader,
        optimizer,
        *,
        val_loader=None,
        device=None,
        precision: str = "fp32",
        scheduler=None,
        grad_clip_norm: float | None = None,
        log_every: int = 10,
        progress_callback: Any | None = None,
    ) -> None:
        import torch

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm
        self.log_every = max(1, int(log_every))
        self.requested_precision = precision
        self.progress_callback = progress_callback
        self.device = torch.device(device) if device is not None else torch.device(select_torch_device())
        self.model = self.model.to(self.device)
        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and precision == "fp16"
        )

    def _autocast_context(self):
        import contextlib
        import torch

        if self.requested_precision == "fp16":
            enabled = self.device.type in {"cuda", "mps"}
            return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=enabled)
        if self.requested_precision in {"bf16", "nf4"}:
            enabled = self.device.type in {"cpu", "cuda"}
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=enabled)
        return contextlib.nullcontext()

    def _trainable_parameters(self):
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def _sanitize_gradients(self) -> None:
        import torch

        for parameter in self._trainable_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                parameter.grad = torch.nan_to_num(parameter.grad, nan=0.0, posinf=0.0, neginf=0.0)

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        import torch

        self.model.train()
        moved_batch = _move_ssl_batch_to_device(batch, self.device)
        images = moved_batch["image"]
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast_context():
            loss = self.model(images)
        if not torch.isfinite(loss):
            raise FloatingPointError("Encountered a non-finite SSL loss.")

        if self.grad_scaler.is_enabled():
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            self._sanitize_gradients()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), self.grad_clip_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self._sanitize_gradients()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), self.grad_clip_norm)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {"loss": float(loss.detach().cpu())}

    def train_epoch(self, epoch_index: int, total_epochs: int = 1) -> dict[str, float]:
        sampler = getattr(self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch_index)

        losses: list[float] = []
        num_batches = len(self.train_loader)
        for step_index, batch in enumerate(self.train_loader, start=1):
            metrics = self.train_step(batch)
            losses.append(metrics["loss"])
            
            if self.progress_callback is not None:
                total_work = total_epochs * num_batches
                completed_work = (epoch_index * num_batches) + step_index
                detail = f"ep {epoch_index+1}/{total_epochs} batch {step_index}/{num_batches} loss={metrics['loss']:.4f}"
                self.progress_callback(completed_work, total_work, detail)

            if step_index % self.log_every == 0:
                print(
                    f"train-mim epoch={epoch_index + 1} step={step_index}/{num_batches} "
                    f"loss={metrics['loss']:.6f}"
                )

        return {
            "loss": float(sum(losses) / len(losses)) if losses else float("nan"),
            "batches": len(losses),
        }

    def evaluate(self, epoch_index: int) -> dict[str, float]:
        import torch

        if self.val_loader is None:
            return {"loss": float("nan"), "batches": 0}

        self.model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch in self.val_loader:
                moved_batch = _move_ssl_batch_to_device(batch, self.device)
                with self._autocast_context():
                    loss = self.model(moved_batch["image"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Encountered a non-finite SSL validation loss.")
                losses.append(float(loss.detach().cpu()))

        mean_loss = float(sum(losses) / len(losses)) if losses else float("nan")
        print(f"train-mim epoch={epoch_index + 1} val_loss={mean_loss:.6f}")
        return {"loss": mean_loss, "batches": len(losses)}

    def fit(self, epochs: int, output_dir: str | Path) -> SSLTrainingSummary:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, Any]] = []
        best_metric = float("inf")
        final_train_loss: float | None = None
        final_val_loss: float | None = None
        total_epochs = int(epochs)

        for epoch_index in range(total_epochs):
            train_metrics = self.train_epoch(epoch_index, total_epochs=total_epochs)
            val_metrics = self.evaluate(epoch_index) if self.val_loader is not None else {"loss": None, "batches": 0}
            final_train_loss = train_metrics["loss"]
            final_val_loss = val_metrics["loss"]
            metric = final_val_loss if final_val_loss is not None else final_train_loss
            if metric is None:
                metric = final_train_loss

            epoch_record = {
                "epoch": epoch_index + 1,
                "train_loss": final_train_loss,
                "val_loss": final_val_loss,
            }
            history.append(epoch_record)
            checkpoint = {
                "epoch": epoch_index + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": None if self.scheduler is None else self.scheduler.state_dict(),
                "history": history,
            }
            torch = __import__("torch")
            torch.save(checkpoint, output_dir / "last.pt")
            if metric is not None and metric < best_metric:
                best_metric = metric
                torch.save(checkpoint, output_dir / "best.pt")

        model_metadata = getattr(self.model, "module", self.model)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "history": history,
                    "requested_precision": getattr(model_metadata, "requested_precision", self.requested_precision),
                    "resolved_precision": getattr(model_metadata, "resolved_precision", self.requested_precision),
                    "summary": asdict(
                        SSLTrainingSummary(
                            output_dir=str(output_dir),
                            epochs=total_epochs,
                            train_batches=len(self.train_loader),
                            val_batches=0 if self.val_loader is None else len(self.val_loader),
                            final_train_loss=final_train_loss,
                            final_val_loss=final_val_loss,
                        )
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return SSLTrainingSummary(
            output_dir=str(output_dir),
            epochs=total_epochs,
            train_batches=len(self.train_loader),
            val_batches=0 if self.val_loader is None else len(self.val_loader),
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
        )


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
