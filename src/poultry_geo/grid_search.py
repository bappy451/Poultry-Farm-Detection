from __future__ import annotations

import hashlib
import itertools
import json
import time
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from poultry_geo.models import SUPPORTED_PRECISIONS, UnifiedRSModel
from poultry_geo.optimizers import build_optimizer


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".npy")
SUPPORTED_MASK_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
GRID_KEYS = (
    "backbone",
    "optimizer",
    "precision",
    "learning_rate",
    "weight_decay",
    "freeze_backbone",
    "use_lora",
)


def resolve_runtime_device(device: str | None = None) -> torch.device:
    if device is None or str(device).strip().lower() == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    normalized = str(device).strip().lower()
    if normalized == "gpu":
        if not torch.cuda.is_available():
            raise ValueError("Requested device 'gpu', but CUDA is not available.")
        return torch.device("cuda")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested device 'cuda', but CUDA is not available.")
        return torch.device("cuda")
    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(f"Requested device '{device}', but CUDA is not available.")
        return torch.device(normalized)
    if normalized == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Requested device 'mps', but Apple Metal is not available.")
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")
    return torch.device(device)


def load_grid_search_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.yaml must define a mapping at the top level.")

    for section in ("data", "runtime", "grid_search"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"config.yaml must define a '{section}' mapping.")

    data = config["data"]
    runtime = config["runtime"]
    grid = config["grid_search"]
    for key in ("image_dir", "mask_dir", "image_size", "n_channels", "batch_size", "num_workers"):
        if key not in data:
            raise ValueError(f"config.yaml is missing data.{key}")
    for key in ("device", "seed", "output_dir"):
        if key not in runtime:
            raise ValueError(f"config.yaml is missing runtime.{key}")
    runtime.setdefault("pretrained", False)

    missing_grid = [key for key in GRID_KEYS if key not in grid]
    if missing_grid:
        raise ValueError(f"config.yaml is missing grid_search keys: {', '.join(missing_grid)}")
    for key in GRID_KEYS:
        if not isinstance(grid[key], list) or len(grid[key]) == 0:
            raise ValueError(f"grid_search.{key} must be a non-empty list.")
    return config


def expand_grid_search_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    data = dict(config["data"])
    runtime = dict(config["runtime"])
    grid = config["grid_search"]
    experiments: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    for values in itertools.product(*(grid[key] for key in GRID_KEYS)):
        grid_values = dict(zip(GRID_KEYS, values, strict=True))
        experiment = {
            "data": dict(data),
            "runtime": dict(runtime),
            "grid": grid_values,
        }
        signature = _experiment_signature(experiment)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        experiments.append(experiment)
    return experiments


class PairedSegmentationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        *,
        image_size: int,
        n_channels: int,
    ) -> None:
        self.image_dir = Path(image_dir).expanduser().resolve()
        self.mask_dir = Path(mask_dir).expanduser().resolve()
        if not self.image_dir.is_dir():
            raise ValueError(f"image_dir must be a directory: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise ValueError(f"mask_dir must be a directory: {self.mask_dir}")
        if image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
        if n_channels <= 0:
            raise ValueError("n_channels must be a positive integer.")
        self.image_size = int(image_size)
        self.n_channels = int(n_channels)
        self.pairs = self._discover_pairs()

    def _discover_pairs(self) -> list[tuple[Path, Path]]:
        image_paths = sorted(
            path
            for path in self.image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"No supported images found under {self.image_dir}")

        pairs: list[tuple[Path, Path]] = []
        for image_path in image_paths:
            relative = image_path.relative_to(self.image_dir)
            mask_path = self._resolve_mask_path(relative)
            pairs.append((image_path, mask_path))
        return pairs

    def _resolve_mask_path(self, relative_image_path: Path) -> Path:
        exact = self.mask_dir / relative_image_path
        if exact.exists():
            return exact
        stem = relative_image_path.with_suffix("")
        for extension in SUPPORTED_MASK_EXTENSIONS:
            candidate = self.mask_dir / Path(f"{stem}{extension}")
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No matching mask found for image '{relative_image_path}'.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path = self.pairs[index]
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        return {
            "image": image,
            "mask": mask,
            "path": str(image_path),
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            array = np.load(path, allow_pickle=False)
            chw = _array_to_chw(array, expected_channels=self.n_channels, path=path)
        elif suffix in {".tif", ".tiff"}:
            chw = _load_tiff(path, expected_channels=self.n_channels)
        else:
            with Image.open(path) as source:
                if self.n_channels == 1:
                    image = source.convert("L")
                elif self.n_channels == 4:
                    image = source.convert("RGBA")
                else:
                    image = source.convert("RGB")
                chw = np.moveaxis(np.asarray(image).copy(), -1, 0)
        tensor = _normalize_image_array(chw, path=path)
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        if tensor.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels for '{path}', found {tensor.shape[0]}.")
        return tensor.to(dtype=torch.float32)

    def _load_mask(self, path: Path) -> torch.Tensor:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            array = np.load(path, allow_pickle=False)
            if array.ndim == 3:
                array = array[..., 0]
            mask = torch.from_numpy(array.astype(np.float32))
        else:
            with Image.open(path) as source:
                mask = torch.from_numpy(np.asarray(source.convert("L"), dtype=np.float32))
        mask = (mask > 0).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=(self.image_size, self.image_size), mode="nearest")
        return mask.squeeze(0)


def build_segmentation_dataloader(data_config: dict[str, Any]) -> DataLoader:
    dataset = PairedSegmentationDataset(
        data_config["image_dir"],
        data_config["mask_dir"],
        image_size=int(data_config["image_size"]),
        n_channels=int(data_config["n_channels"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=int(data_config["num_workers"]),
        pin_memory=True,
    )


def profile_single_step(
    model: UnifiedRSModel,
    batch: dict[str, torch.Tensor],
    optimizer,
    *,
    device: str | torch.device,
    precision: str,
) -> dict[str, Any]:
    resolved_device = resolve_runtime_device(str(device))
    normalized_precision = precision.strip().lower()
    if normalized_precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"precision must be one of: {', '.join(SUPPORTED_PRECISIONS)}")

    model = model.to(resolved_device)
    images = batch["image"].to(device=resolved_device, non_blocking=True)
    masks = batch["mask"].to(device=resolved_device, non_blocking=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=resolved_device.type == "cuda" and normalized_precision == "fp16",
    )

    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)

    optimizer.zero_grad(set_to_none=True)
    autocast_context = _autocast_context(resolved_device, normalized_precision)
    started = time.perf_counter()
    with autocast_context:
        logits = model(images)
        loss = criterion(logits, masks)
    if not torch.isfinite(loss):
        raise FloatingPointError("Encountered a non-finite loss during profiling.")

    if grad_scaler.is_enabled():
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        _sanitize_gradients(model.parameters())
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        loss.backward()
        _sanitize_gradients(model.parameters())
        optimizer.step()
    elapsed = time.perf_counter() - started

    peak_memory_mb = 0.0
    if resolved_device.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(resolved_device) / (1024 ** 2))

    return {
        "loss": float(loss.detach().cpu()),
        "step_time_seconds": elapsed,
        "peak_memory_mb": peak_memory_mb,
        "used_grad_scaler": bool(grad_scaler.is_enabled()),
        "resolved_device": str(resolved_device),
        "resolved_precision": normalized_precision,
    }


def run_grid_search(config_path: str | Path = "config.yaml") -> pd.DataFrame:
    config = load_grid_search_config(config_path)
    experiments = expand_grid_search_configs(config)
    output_dir = Path(config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(config["runtime"]["seed"]))
    loader = build_segmentation_dataloader(config["data"])
    batch = next(iter(loader))
    records: list[dict[str, Any]] = []

    for experiment in experiments:
        grid = experiment["grid"]
        experiment_id = hashlib.sha1(_experiment_signature(experiment).encode("utf-8")).hexdigest()[:12]
        record = {
            "experiment_id": experiment_id,
            "backbone": grid["backbone"],
            "optimizer": grid["optimizer"],
            "precision": grid["precision"],
            "learning_rate": grid["learning_rate"],
            "weight_decay": grid["weight_decay"],
            "freeze_backbone": grid["freeze_backbone"],
            "use_lora": grid["use_lora"],
            "pretrained": experiment["runtime"]["pretrained"],
            "status": "pending",
            "reason": "",
            "loss": np.nan,
            "step_time_seconds": np.nan,
            "peak_memory_mb": np.nan,
            "trainable_params": 0,
            "total_params": 0,
            "resolved_device": "",
            "resolved_precision": "",
        }

        model: UnifiedRSModel | None = None
        try:
            if grid["backbone"] == "resnet50" and bool(grid["use_lora"]):
                record["status"] = "skipped"
                record["reason"] = "LoRA is only supported for ViT backbones."
                records.append(record)
                continue

            model = UnifiedRSModel(
                n_channels=int(experiment["data"]["n_channels"]),
                backbone_name=str(grid["backbone"]),
                image_size=int(experiment["data"]["image_size"]),
                precision=str(grid["precision"]),
                pretrained=bool(experiment["runtime"]["pretrained"]),
                freeze_backbone=bool(grid["freeze_backbone"]),
                use_lora=bool(grid["use_lora"]),
            )
            record["trainable_params"] = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            record["total_params"] = sum(parameter.numel() for parameter in model.parameters())
            optimizer = build_optimizer(
                str(grid["optimizer"]),
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=float(grid["learning_rate"]),
                weight_decay=float(grid["weight_decay"]),
            )
            metrics = profile_single_step(
                model,
                batch,
                optimizer,
                device=experiment["runtime"]["device"],
                precision=str(grid["precision"]),
            )
            record.update(metrics)
            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "error"
            record["reason"] = str(exc)
            if model is not None:
                record["trainable_params"] = sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                )
                record["total_params"] = sum(parameter.numel() for parameter in model.parameters())
        records.append(record)

    results = pd.DataFrame.from_records(records)
    results.to_csv(output_dir / "grid_search_results.csv", index=False)
    return results


def _experiment_signature(experiment: dict[str, Any]) -> str:
    return json.dumps(experiment, sort_keys=True, separators=(",", ":"))


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp16":
        enabled = device.type in {"cuda", "mps"}
        return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)
    if precision == "bf16":
        enabled = device.type in {"cpu", "cuda"}
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)
    return nullcontext()


def _sanitize_gradients(parameters: Iterable[torch.nn.Parameter]) -> None:
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            parameter.grad = torch.nan_to_num(parameter.grad, nan=0.0, posinf=0.0, neginf=0.0)


def _load_tiff(path: Path, *, expected_channels: int) -> np.ndarray:
    try:
        import rasterio
    except ImportError:
        with Image.open(path) as source:
            image = source.convert("RGB" if expected_channels != 1 else "L")
            if expected_channels == 1:
                return np.asarray(image)[np.newaxis, ...]
            return np.moveaxis(np.asarray(image), -1, 0)
    with rasterio.open(path) as src:
        return src.read()


def _array_to_chw(array: np.ndarray, *, expected_channels: int, path: Path) -> np.ndarray:
    if array.ndim == 2:
        chw = array[np.newaxis, ...]
    elif array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D array for '{path}', found shape {array.shape}.")
    elif array.shape[0] == expected_channels:
        chw = array
    elif array.shape[-1] == expected_channels:
        chw = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(f"Could not infer channel layout for '{path}' with expected_channels={expected_channels}.")
    return chw


def _normalize_image_array(chw: np.ndarray, *, path: Path) -> torch.Tensor:
    if chw.ndim != 3:
        raise ValueError(f"Expected CHW image array for '{path}', found shape {chw.shape}.")
    if np.issubdtype(chw.dtype, np.integer):
        max_value = np.iinfo(chw.dtype).max
        tensor = torch.from_numpy(chw.astype(np.float32) / float(max_value))
    else:
        tensor = torch.from_numpy(chw.astype(np.float32))
        if float(tensor.max()) > 1.0 or float(tensor.min()) < 0.0:
            tensor = tensor.clamp(0.0, 1.0)
    return tensor


__all__ = [
    "PairedSegmentationDataset",
    "build_segmentation_dataloader",
    "expand_grid_search_configs",
    "load_grid_search_config",
    "profile_single_step",
    "run_grid_search",
]
