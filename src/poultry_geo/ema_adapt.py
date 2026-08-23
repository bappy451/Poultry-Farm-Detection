"""EMA teacher/student domain adaptation for Ultralytics OBB detectors.

The student starts from a COCO-pretrained detector and keeps the native
Ultralytics OBB loss on labeled poultry images.  A frozen EMA copy supplies a
stable target for detector-backbone feature consistency on unlabeled NAIP
crops.  The EMA copy is the same model Ultralytics already checkpoints.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from poultry_geo.naip_data import NAIPDataset

try:
    from ultralytics.models.yolo.obb.train import OBBModel, OBBTrainer
    from ultralytics.utils import LOGGER, RANK
    from ultralytics.utils.torch_utils import unwrap_model
except ImportError as exc:  # pragma: no cover
    raise ImportError("Ultralytics is required for EMA YOLO adaptation.") from exc


FEATURE_INDICES = (4, 6, 10)  # YOLO11 P3/P4/P5 backbone outputs


def _rgb(image: torch.Tensor, path: str) -> torch.Tensor:
    """Return a detector-compatible RGB tensor in [0, 1]."""
    if image.ndim != 3:
        raise ValueError(f"Expected CHW imagery for '{path}', got shape {tuple(image.shape)}.")
    channels = image.shape[0]
    if channels == 1:
        image = image.repeat(3, 1, 1)
    elif channels >= 3:
        image = image[:3]
    else:
        raise ValueError(f"Expected at least one image channel for '{path}', got {channels}.")
    return image.to(dtype=torch.float32).clamp_(0.0, 1.0).contiguous()


def _color_view(image: torch.Tensor, *, strong: bool) -> torch.Tensor:
    """Apply bounded photometric noise without changing object geometry."""
    if strong:
        brightness = float(torch.empty(()).uniform_(0.65, 1.35).item())
        contrast = float(torch.empty(()).uniform_(0.65, 1.35).item())
        noise = torch.randn_like(image) * 0.025
    else:
        brightness = float(torch.empty(()).uniform_(0.90, 1.10).item())
        contrast = float(torch.empty(()).uniform_(0.90, 1.10).item())
        noise = torch.randn_like(image) * 0.008
    mean = image.mean(dim=(1, 2), keepdim=True)
    view = (image * brightness - mean) * contrast + mean + noise
    return view.clamp_(0.0, 1.0)


class PairedNAIPDataset(Dataset[dict[str, torch.Tensor]]):
    """Produce synchronized weak/strong views of unlabeled NAIP crops."""

    def __init__(
        self,
        root: str | Path,
        crop_size: int,
        catalog_path: str | Path | None = None,
        max_files: int | None = None,
        subset_seed: int = 0,
    ) -> None:
        self.base = NAIPDataset(
            root,
            crop_size=int(crop_size),
            channels=None,
            precision="fp32",
            augment=False,
            catalog_path=catalog_path,
            max_files=max_files,
            subset_seed=subset_seed,
        )

    @property
    def files(self):
        return self.base.files

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = str(self.base.files[index])
        image = self.base._load_image(path)
        _, height, width = image.shape
        if height < self.base.crop_size or width < self.base.crop_size:
            raise ValueError(
                f"unlabeled crop_size={self.base.crop_size} exceeds '{path}' dimensions {height}x{width}."
            )
        crop = self.base._random_crop(image)
        if bool(torch.rand(()) < 0.5):
            crop = torch.flip(crop, dims=(2,))
        if bool(torch.rand(()) < 0.5):
            crop = torch.flip(crop, dims=(1,))
        crop = _rgb(crop, path)
        return {"weak": _color_view(crop, strong=False), "strong": _color_view(crop, strong=True)}


class PairedLoader:
    """Transparent wrapper appending an unlabeled batch to every YOLO batch."""

    def __init__(self, labeled_loader, unlabeled_loader) -> None:
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.dataset = labeled_loader.dataset
        self.sampler = getattr(labeled_loader, "sampler", None)
        self.num_workers = getattr(labeled_loader, "num_workers", 0)

    def __len__(self) -> int:
        return len(self.labeled_loader)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        unlabeled_iter = iter(self.unlabeled_loader)
        for labeled_batch in self.labeled_loader:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(self.unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)
            labeled_batch["unlabeled_img_weak"] = unlabeled_batch["weak"]
            labeled_batch["unlabeled_img_strong"] = unlabeled_batch["strong"]
            yield labeled_batch

    def reset(self) -> None:
        if hasattr(self.labeled_loader, "reset"):
            self.labeled_loader.reset()

    def close(self) -> None:
        for loader in (self.labeled_loader, self.unlabeled_loader):
            if hasattr(loader, "close"):
                loader.close()

    def __getattr__(self, name: str):
        return getattr(self.labeled_loader, name)


def _backbone_features(model: nn.Module, images: torch.Tensor) -> list[torch.Tensor]:
    """Run YOLO11 backbone modules 0..10 and return P3/P4/P5 maps."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"EMA adaptation expects [B,3,H,W] images, got {tuple(images.shape)}.")
    modules = getattr(model, "model", None)
    if modules is None or len(modules) <= max(FEATURE_INDICES):
        raise RuntimeError("The loaded detector does not expose YOLO backbone modules 0..10.")
    current = images
    outputs: dict[int, torch.Tensor] = {}
    for index, module in enumerate(modules[: max(FEATURE_INDICES) + 1]):
        current = module(current)
        if index in FEATURE_INDICES:
            outputs[index] = current
    return [outputs[index] for index in FEATURE_INDICES]


class EMADomainAdaptOBBModel(OBBModel):
    """Native OBB detector with an EMA feature-consistency term."""

    def __init__(self, *args, consistency_weight: float = 0.05, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.consistency_weight = float(consistency_weight)
        # Do not register the teacher as a child module.  Ultralytics' ModelEMA
        # owns and checkpoints the teacher copy; registering it here would double
        # optimizer/checkpoint memory and recursively EMA it.
        object.__setattr__(self, "_ema_teacher", None)

    def consistency_loss(self, weak: torch.Tensor, strong: torch.Tensor) -> torch.Tensor:
        teacher = object.__getattribute__(self, "_ema_teacher")
        if teacher is None:
            return strong.new_zeros(())
        with torch.no_grad():
            teacher_features = _backbone_features(teacher, weak)
        student_features = _backbone_features(self, strong)
        losses = []
        for student, target in zip(student_features, teacher_features):
            if student.shape != target.shape:
                raise RuntimeError(
                    "Teacher/student feature shapes differ: "
                    f"student={tuple(student.shape)} teacher={tuple(target.shape)}."
                )
            student_n = F.normalize(student.float(), dim=1, eps=1e-6)
            target_n = F.normalize(target.float(), dim=1, eps=1e-6)
            losses.append((1.0 - (student_n * target_n).sum(dim=1)).mean())
        return torch.stack(losses).mean() if losses else strong.new_zeros(())

    def loss(self, batch: dict[str, Any], preds=None):
        supervised_loss, loss_items = super().loss(batch, preds)
        consistency = supervised_loss.new_zeros(())
        weak = batch.get("unlabeled_img_weak")
        strong = batch.get("unlabeled_img_strong")
        if weak is not None and strong is not None and self.consistency_weight > 0:
            consistency = self.consistency_loss(weak, strong)
        total_loss = supervised_loss + self.consistency_weight * consistency
        if not torch.is_tensor(loss_items):
            loss_items = torch.as_tensor(loss_items, device=total_loss.device)
        loss_items = torch.cat((loss_items.detach().reshape(-1), consistency.detach().reshape(1)))
        return total_loss, loss_items


class EMADomainAdaptOBBTrainer(OBBTrainer):
    """OBBTrainer with paired unlabeled NAIP data and an EMA teacher."""

    _CUSTOM_OPTIONS = frozenset({
        "unlabeled_root", "unlabeled_catalog", "unlabeled_batch", "unlabeled_imgsz",
        "unlabeled_workers", "unlabeled_limit", "prefetch_factor", "consistency_weight",
        "consistency_rampup", "ema_decay",
    })

    def __init__(self, *args, **kwargs):
        # Ultralytics rejects unknown keys before the trainer is constructed.
        # Remove adaptation-only options here and retain them on the trainer.
        overrides = kwargs.get("overrides")
        self._adapt_options = {}
        if isinstance(overrides, dict):
            for key in self._CUSTOM_OPTIONS:
                if key in overrides:
                    self._adapt_options[key] = overrides.pop(key)
        super().__init__(*args, **kwargs)

    def _opt(self, name: str, default):
        return self._adapt_options.get(name, default)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = EMADomainAdaptOBBModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data.get("channels", 3),
            verbose=verbose and RANK == -1,
            consistency_weight=float(self._opt("consistency_weight", 0.05)),
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = (*self.loss_names, "consistency_loss")
        return validator

    def _build_train_pipeline(self):
        super()._build_train_pipeline()
        root = self._opt("unlabeled_root", None)
        if not root:
            raise ValueError("--unlabeled-root is required for EMA domain adaptation.")
        crop_size = int(self._opt("unlabeled_imgsz", 448))
        if crop_size % 32:
            raise ValueError("--unlabeled-imgsz must be divisible by 32.")
        unlabeled_limit = self._opt("unlabeled_limit", 250_000)
        if unlabeled_limit is not None:
            unlabeled_limit = int(unlabeled_limit)
            if unlabeled_limit <= 0:
                raise ValueError("--unlabeled-limit must be positive or None to disable limiting.")
        unlabeled_dataset = PairedNAIPDataset(
            root,
            crop_size=crop_size,
            catalog_path=self._opt("unlabeled_catalog", None),
            max_files=unlabeled_limit,
            subset_seed=0,
        )
        unlabeled_batch = int(self._opt("unlabeled_batch", self.batch_size))
        if unlabeled_batch <= 0:
            raise ValueError("--unlabeled-batch must be positive.")
        workers = int(self._opt("unlabeled_workers", self.args.workers))
        loader_kwargs: dict[str, Any] = {
            "batch_size": unlabeled_batch,
            "shuffle": True,
            "num_workers": workers,
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": workers > 0,
        }
        if workers > 0:
            loader_kwargs["prefetch_factor"] = max(2, int(self._opt("prefetch_factor", 2)))
        unlabeled_loader = DataLoader(unlabeled_dataset, drop_last=False, **loader_kwargs)
        self.unlabeled_loader = unlabeled_loader
        self.train_loader = PairedLoader(self.train_loader, unlabeled_loader)
        paired_crops = len(self.train_loader) * unlabeled_batch
        LOGGER.info(
            "EMA adaptation: %d labeled batches/epoch; unlabeled pool=%d tiles; %d unlabeled crops paired/epoch (batch=%d, crop=%d, workers=%d).",
            len(self.train_loader), len(unlabeled_dataset), paired_crops, unlabeled_batch, crop_size, workers,
        )

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        for key in ("unlabeled_img_weak", "unlabeled_img_strong"):
            value = batch.get(key)
            if value is not None:
                batch[key] = value.to(self.device, dtype=torch.float32, non_blocking=True)
        return batch

    def _setup_train(self):
        super()._setup_train()
        student = unwrap_model(self.model)
        object.__setattr__(student, "_ema_teacher", self.ema.ema)
        decay = float(self._opt("ema_decay", 0.9998))
        if not 0.0 < decay < 1.0:
            raise ValueError("--ema-decay must be between 0 and 1.")
        self.ema.decay = lambda updates: decay * (1.0 - math.exp(-updates / 2000.0))
        self.add_callback("on_train_epoch_start", self._update_consistency_weight)
        self._update_consistency_weight(self)

    def _update_consistency_weight(self, trainer):
        model = unwrap_model(trainer.model)
        base = float(self._opt("consistency_weight", 0.05))
        rampup = max(float(self._opt("consistency_rampup", 10.0)), 1.0)
        epoch = float(getattr(trainer, "epoch", 0))
        model.consistency_weight = base * min(1.0, (epoch + 1.0) / rampup)
