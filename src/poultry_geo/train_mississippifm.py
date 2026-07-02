from __future__ import annotations

from contextlib import nullcontext
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from poultry_geo.evaluate_probes import evaluate_mississippi_probes
from poultry_geo.mississippi_data import build_overlap_aware_dataloaders
from poultry_geo.mississippi_models import (
    MississippiFMEncoder,
    compute_gram_anchoring_loss,
    compute_overlap_alignment_loss,
)
from poultry_geo.training import select_torch_device


@dataclass(slots=True)
class MississippiFMTrainingConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    image_size: int = 256
    model_size: str = "vit_b"
    precision: str = "fp32"
    backbone_impl: str = "custom"
    backbone_name: str | None = None
    val_fraction: float = 0.2
    num_workers: int = 0
    patch_size: int = 16
    epoch_size: int = 128
    stage2_fraction: float = 0.1
    lambda_patch: float = 1.0
    lambda_overlap: float = 1.0
    lambda_gram: float = 1.0
    probe_config_path: Path | None = None
    output_dir: Path = Path("trained_models/mississippifm/default")


@dataclass(slots=True)
class MississippiFMTrainingSummary:
    output_dir: str
    epochs: int
    train_batches: int
    val_batches: int
    final_total_loss: float | None
    final_val_total_loss: float | None


def load_mississippifm_config(path: str | Path) -> MississippiFMTrainingConfig:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("MississippiFM config must contain a top-level mapping.")
    defaults = MississippiFMTrainingConfig()
    values = {}
    for field_name in defaults.__dataclass_fields__:
        if field_name in {"output_dir", "probe_config_path"}:
            continue
        if field_name in payload:
            values[field_name] = payload[field_name]
    if "output_dir" in payload:
        values["output_dir"] = Path(payload["output_dir"])
    if "probe_config_path" in payload and payload["probe_config_path"] is not None:
        values["probe_config_path"] = Path(payload["probe_config_path"])
    return replace(defaults, **values)


def train_mississippifm(
    config: MississippiFMTrainingConfig,
    *,
    data_root: str | Path,
    adjacency_csv: str | Path,
    output_dir: str | Path | None = None,
    resume: bool = False,
    device: str | None = None,
) -> MississippiFMTrainingSummary:
    run_dir = Path(output_dir or config.output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = build_overlap_aware_dataloaders(
        data_root,
        adjacency_csv,
        batch_size=config.batch_size,
        val_fraction=config.val_fraction,
        seed=27,
        precision=config.precision,
        augment=True,
        high_res=False,
        low_res_sizes=(config.image_size,),
        patch_size=config.patch_size,
        epoch_size=config.epoch_size,
        num_workers=config.num_workers,
    )
    model = MississippiFMEncoder(
        image_size=config.image_size,
        patch_size=config.patch_size,
        model_size=config.model_size,
        precision=config.precision,
        projection_dim=max(64, config.patch_size * 4),
        backbone_impl=config.backbone_impl,
        backbone_name=config.backbone_name,
    )
    torch_device = torch.device(device or select_torch_device())
    model = model.to(torch_device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    scaler = _build_grad_scaler(device=torch_device, precision=config.precision)

    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_val = float("inf")
    best_probe_score = float("-inf")
    if resume and (run_dir / "last.pt").exists():
        checkpoint = torch.load(run_dir / "last.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val = float(checkpoint.get("best_val_loss", best_val))
        best_probe_score = float(checkpoint.get("best_probe_score", best_probe_score))

    stage2_start_epoch = max(0, config.epochs - max(1, round(config.epochs * config.stage2_fraction)))
    final_train_loss: float | None = None
    final_val_loss: float | None = None

    writer = _build_summary_writer(run_dir)
    try:
        for epoch_index in range(start_epoch, config.epochs):
            high_res_now = epoch_index >= stage2_start_epoch
            if train_loader.dataset.high_res != high_res_now:
                train_loader.dataset.high_res = high_res_now
                val_loader.dataset.high_res = high_res_now

            train_metrics = _run_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=torch_device,
                lambda_patch=config.lambda_patch,
                lambda_overlap=config.lambda_overlap,
                lambda_gram=config.lambda_gram,
                precision=config.precision,
                scaler=scaler,
                train=True,
            )
            val_metrics = _run_epoch(
                model,
                val_loader,
                optimizer=optimizer,
                scheduler=None,
                device=torch_device,
                lambda_patch=config.lambda_patch,
                lambda_overlap=config.lambda_overlap,
                lambda_gram=config.lambda_gram,
                precision=config.precision,
                scaler=None,
                train=False,
            )
            final_train_loss = train_metrics["total_loss"]
            final_val_loss = val_metrics["total_loss"]
            probe_score = None
            if config.probe_config_path is not None:
                probe_summary = evaluate_mississippi_probes(
                    config.probe_config_path,
                    output_dir=run_dir / "probe_eval" / f"epoch_{epoch_index + 1}",
                    encoder=model,
                    device=torch_device,
                )
                probe_score = float(probe_summary["aggregate_primary_score"])
            history.append(
                {
                    "epoch": epoch_index + 1,
                    "train_total_loss": final_train_loss,
                    "train_global_loss": train_metrics["global_loss"],
                    "train_patch_loss": train_metrics["patch_loss"],
                    "train_overlap_loss": train_metrics["overlap_loss"],
                    "train_gram_loss": train_metrics["gram_loss"],
                    "val_total_loss": final_val_loss,
                    "val_global_loss": val_metrics["global_loss"],
                    "val_patch_loss": val_metrics["patch_loss"],
                    "val_overlap_loss": val_metrics["overlap_loss"],
                    "val_gram_loss": val_metrics["gram_loss"],
                    "probe_aggregate_primary_score": probe_score,
                    "high_res": high_res_now,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            _log_writer(writer, epoch_index, history[-1])
            checkpoint = {
                "epoch": epoch_index + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "history": history,
                "best_val_loss": best_val,
                "best_probe_score": best_probe_score,
                "config": _json_safe_config(config),
            }
            torch.save(checkpoint, run_dir / "last.pt")
            is_better = False
            if probe_score is not None:
                if probe_score >= best_probe_score:
                    best_probe_score = float(probe_score)
                    checkpoint["best_probe_score"] = best_probe_score
                    is_better = True
            elif final_val_loss is not None and final_val_loss <= best_val:
                best_val = float(final_val_loss)
                checkpoint["best_val_loss"] = best_val
                is_better = True
            if is_better:
                torch.save(checkpoint, run_dir / "best.pt")
    finally:
        if writer is not None:
            writer.close()

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "config": _json_safe_config(config),
                "history": history,
                "summary": asdict(
                    MississippiFMTrainingSummary(
                        output_dir=str(run_dir),
                        epochs=config.epochs,
                        train_batches=len(train_loader),
                        val_batches=len(val_loader),
                        final_total_loss=final_train_loss,
                        final_val_total_loss=final_val_loss,
                    )
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return MississippiFMTrainingSummary(
        output_dir=str(run_dir),
        epochs=config.epochs,
        train_batches=len(train_loader),
        val_batches=len(val_loader),
        final_total_loss=final_train_loss,
        final_val_total_loss=final_val_loss,
    )


def _run_epoch(
    model: MississippiFMEncoder,
    loader,
    *,
    optimizer,
    scheduler,
    device: torch.device,
    lambda_patch: float,
    lambda_overlap: float,
    lambda_gram: float,
    precision: str,
    scaler,
    train: bool,
) -> dict[str, float]:
    totals = {"total_loss": 0.0, "global_loss": 0.0, "patch_loss": 0.0, "overlap_loss": 0.0, "gram_loss": 0.0}
    count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    model.train(mode=train)
    with context:
        for batch in loader:
            images_a = batch["image_a"].to(device=device, non_blocking=True)
            images_b = batch["image_b"].to(device=device, non_blocking=True)
            with _autocast_context(device=device, precision=precision):
                outputs_a = model.forward_views(global_views=images_a)
                outputs_b = model.forward_views(global_views=images_b)

                global_loss = 1.0 - (
                    outputs_a["global_distillation"] * outputs_b["global_distillation"]
                ).sum(dim=-1).mean()

                paired_patch_count = min(outputs_a["patch_predictions"].shape[1], outputs_b["patch_predictions"].shape[1])
                patch_loss = F.mse_loss(
                    outputs_a["patch_predictions"][:, :paired_patch_count].float(),
                    outputs_b["patch_predictions"][:, :paired_patch_count].detach().float(),
                )

                overlap_loss = compute_overlap_alignment_loss(
                    outputs_a["overlap_projection"],
                    outputs_b["overlap_projection"],
                    batch["patch_indices_a"],
                    batch["patch_indices_b"],
                )
                gram_loss = (
                    compute_gram_anchoring_loss(outputs_a["global_patches"], model.anchor_patch_embeddings(images_a))
                    + compute_gram_anchoring_loss(outputs_b["global_patches"], model.anchor_patch_embeddings(images_b))
                ) / 2.0

                total_loss = global_loss + (lambda_patch * patch_loss) + (lambda_overlap * overlap_loss) + (lambda_gram * gram_loss)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()
            totals["total_loss"] += float(total_loss.detach().cpu())
            totals["global_loss"] += float(global_loss.detach().cpu())
            totals["patch_loss"] += float(patch_loss.detach().cpu())
            totals["overlap_loss"] += float(overlap_loss.detach().cpu())
            totals["gram_loss"] += float(gram_loss.detach().cpu())
            count += 1
    if train and scheduler is not None:
        scheduler.step()
    if count == 0:
        return {key: float("nan") for key in totals}
    return {key: value / count for key, value in totals.items()}


def _build_summary_writer(run_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return None
    return SummaryWriter(log_dir=str(run_dir / "tensorboard"))


def _autocast_context(*, device: torch.device, precision: str):
    if precision == "fp16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision in {"bf16", "nf4"}:
        dtype = torch.bfloat16
        if device.type in {"cpu", "cuda"}:
            return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def _build_grad_scaler(*, device: torch.device, precision: str):
    if precision == "fp16" and device.type == "cuda":
        return torch.cuda.amp.GradScaler()
    return None


def _log_writer(writer, epoch_index: int, metrics: dict[str, Any]) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(key, value, epoch_index + 1)


def _json_safe_config(config: MississippiFMTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["probe_config_path"] = None if config.probe_config_path is None else str(config.probe_config_path)
    return payload
