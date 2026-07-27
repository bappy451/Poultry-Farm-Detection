from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from poultry_geo.naip_data import NAIPDataset, _split_files_by_parent, _color_jitter, _random_bool, _uniform


class DINOMultiCropDataAugmentation:
    """Multi-crop data augmentation for DINO self-supervised learning.
    
    Generates:
      - num_global_crops views at global_size (default 224x224)
      - num_local_crops views at local_size (default 96x96)
    """

    def __init__(
        self,
        global_size: int = 224,
        local_size: int = 96,
        global_scale: tuple[float, float] = (0.32, 1.0),
        local_scale: tuple[float, float] = (0.05, 0.32),
        num_global_crops: int = 2,
        num_local_crops: int = 8,
        normalize_imagenet: bool = True,
    ) -> None:
        self.global_size = int(global_size)
        self.local_size = int(local_size)
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.num_global_crops = int(num_global_crops)
        self.num_local_crops = int(num_local_crops)
        self.normalize_imagenet = normalize_imagenet

        # ImageNet mean and std tensors for RGB normalization
        self.register_mean_std()

    def register_mean_std(self) -> None:
        self.mean_rgb = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std_rgb = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, image: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Apply multi-crop augmentations to input tensor [C, H, W] in [0.0, 1.0].
        
        Returns:
          global_crops: list of num_global_crops tensors [C, global_size, global_size]
          local_crops: list of num_local_crops tensors [C, local_size, local_size]
        """
        global_crops: list[torch.Tensor] = []
        for idx in range(self.num_global_crops):
            crop = self._random_resized_crop(image, self.global_size, self.global_scale)
            if _random_bool():
                crop = torch.flip(crop, dims=(2,))
            if _random_bool():
                crop = torch.flip(crop, dims=(1,))
            # Apply color jitter
            if torch.rand(()) < 0.8:
                crop = _color_jitter(crop)
            # Apply Gaussian blur: crop 0 (100%), crop 1 (10%)
            blur_p = 1.0 if idx == 0 else 0.1
            if torch.rand(()) < blur_p:
                crop = self._gaussian_blur(crop)
            # Apply solarize on 2nd global crop (20%)
            if idx == 1 and torch.rand(()) < 0.2:
                crop = self._solarize(crop)

            global_crops.append(self._normalize(crop))

        local_crops: list[torch.Tensor] = []
        for _ in range(self.num_local_crops):
            crop = self._random_resized_crop(image, self.local_size, self.local_scale)
            if _random_bool():
                crop = torch.flip(crop, dims=(2,))
            if _random_bool():
                crop = torch.flip(crop, dims=(1,))
            if torch.rand(()) < 0.8:
                crop = _color_jitter(crop)
            if torch.rand(()) < 0.5:
                crop = self._gaussian_blur(crop)

            local_crops.append(self._normalize(crop))

        return global_crops, local_crops

    def _random_resized_crop(
        self,
        image: torch.Tensor,
        target_size: int,
        scale_range: tuple[float, float],
    ) -> torch.Tensor:
        _, h, w = image.shape
        area = float(h * w)

        for _ in range(10):
            target_area = _uniform(scale_range[0], scale_range[1]) * area
            aspect_ratio = math.exp(_uniform(math.log(3.0 / 4.0), math.log(4.0 / 3.0)))

            crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < crop_w <= w and 0 < crop_h <= h:
                top = int(torch.randint(0, h - crop_h + 1, (1,)).item())
                left = int(torch.randint(0, w - crop_w + 1, (1,)).item())
                cropped = image[:, top : top + crop_h, left : left + crop_w]
                return F.interpolate(
                    cropped.unsqueeze(0),
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

        # Fallback to center crop if random crop loop fails
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        cropped = image[:, top : top + min_dim, left : left + min_dim]
        return F.interpolate(
            cropped.unsqueeze(0),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _gaussian_blur(self, image: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Apply lightweight Gaussian blur."""
        sigma = _uniform(0.1, 2.0)
        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        kernel1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel1d = kernel1d / kernel1d.sum()
        kernel2d = kernel1d[:, None] * kernel1d[None, :]
        kernel2d = kernel2d.to(image.device).expand(image.shape[0], 1, kernel_size, kernel_size)

        pad = kernel_size // 2
        padded = F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
        blurred = F.conv2d(padded, kernel2d, groups=image.shape[0])
        return blurred.squeeze(0).clamp(0.0, 1.0)

    def _solarize(self, image: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return torch.where(image >= threshold, 1.0 - image, image)

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        if self.normalize_imagenet and image.shape[0] == 3:
            mean = self.mean_rgb.to(image.device)
            std = self.std_rgb.to(image.device)
            return (image - mean) / std
        return image


class DINODataset(NAIPDataset):
    """NAIPDataset subclass yielding DINO global and local crops."""

    def __init__(
        self,
        root: str | Path,
        crop_size: int = 224,
        local_crop_size: int = 96,
        num_global_crops: int = 2,
        num_local_crops: int = 8,
        channels: int | None = None,
        precision: str = "fp32",
        files: Sequence[str | Path] | None = None,
        catalog_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            root=root,
            crop_size=crop_size,
            channels=channels,
            precision=precision,
            augment=False,
            files=files,
            catalog_path=catalog_path,
        )
        self.transform = DINOMultiCropDataAugmentation(
            global_size=crop_size,
            local_size=local_crop_size,
            num_global_crops=num_global_crops,
            num_local_crops=num_local_crops,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = str(self.files[index])
        image = self._load_image(path)  # returns [C, H, W] in [0, 1]
        global_crops, local_crops = self.transform(image)

        # Convert all crops to requested precision dtype
        global_crops = [crop.to(dtype=self.dtype) for crop in global_crops]
        local_crops = [crop.to(dtype=self.dtype) for crop in local_crops]

        return {
            "global_crops": global_crops,
            "local_crops": local_crops,
            "path": str(path),
        }


def get_dino_dataloaders(
    root: str | Path,
    batch_size: int,
    val_fraction: float = 0.2,
    seed: int = 27,
    crop_size: int = 224,
    local_crop_size: int = 96,
    num_global_crops: int = 2,
    num_local_crops: int = 8,
    channels: int | None = None,
    precision: str = "fp32",
    num_workers: int = 4,
    prefetch_factor: int = 2,
    catalog_path: str | Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders yielding DINO multi-crop dictionaries."""
    catalog = DINODataset(
        root,
        crop_size=crop_size,
        local_crop_size=local_crop_size,
        num_global_crops=num_global_crops,
        num_local_crops=num_local_crops,
        channels=channels,
        precision=precision,
        catalog_path=catalog_path,
    )
    train_files, val_files = _split_files_by_parent(
        catalog.files,
        root=catalog.root,
        val_fraction=val_fraction,
        seed=seed,
    )

    train_dataset = DINODataset(
        root,
        files=train_files,
        crop_size=crop_size,
        local_crop_size=local_crop_size,
        num_global_crops=num_global_crops,
        num_local_crops=num_local_crops,
        channels=channels,
        precision=precision,
    )
    val_dataset = DINODataset(
        root,
        files=val_files,
        crop_size=crop_size,
        local_crop_size=local_crop_size,
        num_global_crops=num_global_crops,
        num_local_crops=num_local_crops,
        channels=channels,
        precision=precision,
    )

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    train_sampler = None
    val_sampler = None
    if distributed:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=seed,
        )

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader
