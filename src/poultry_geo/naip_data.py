from __future__ import annotations

import math
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".npy"})
PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class FileIndex(Sequence[Path]):
    """Compact, pickle-friendly path sequence with normal list-like semantics.

    Internally paths stay in a NumPy string buffer so Windows DataLoader workers do
    not pickle millions of Python objects. Public iteration/indexing returns Path
    objects, preserving the original dataset API and useful equality semantics.
    """

    def __init__(self, paths: Sequence[str | Path]) -> None:
        self._paths = np.asarray([str(path) for path in paths], dtype=str)

    def __len__(self) -> int:
        return int(self._paths.shape[0])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [Path(str(path)) for path in self._paths[index]]
        return Path(str(self._paths[index]))

    def __iter__(self):
        for path in self._paths:
            yield Path(str(path))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FileIndex):
            return bool(np.array_equal(self._paths, other._paths))
        if isinstance(other, Sequence):
            return list(self) == list(other)
        return NotImplemented

    def __getstate__(self):
        return self._paths

    def __setstate__(self, state) -> None:
        self._paths = np.asarray(state, dtype=str)


class NAIPDataset(Dataset[dict[str, Any]]):
    """Load nested NAIP tiles and apply synchronized tensor augmentations."""

    def __init__(
        self,
        root: str | Path,
        crop_size: int = 224,
        channels: int | None = None,
        precision: str = "fp32",
        augment: bool = True,
        files: Sequence[str | Path] | None = None,
        catalog_path: str | Path | None = None,
        max_files: int | None = None,
        subset_seed: int = 0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"NAIP imagery root does not exist: {self.root}")
        if not self.root.is_dir():
            raise ValueError(f"NAIP imagery root must be a directory: {self.root}")
        if crop_size <= 0:
            raise ValueError("crop_size must be a positive integer.")
        if channels is not None and channels <= 0:
            raise ValueError("channels must be a positive integer or None.")
        if precision not in PRECISION_DTYPES:
            raise ValueError("precision must be fp32, fp16, or bf16.")

        self.crop_size = int(crop_size)
        self.channels = channels
        self.dtype = PRECISION_DTYPES[precision]
        self.precision = precision
        self.augment = bool(augment)
        self.catalog_path = Path(catalog_path) if catalog_path else None
        if max_files is not None and max_files <= 0:
            raise ValueError("max_files must be a positive integer or None.")
        self.max_files = int(max_files) if max_files is not None else None
        self.subset_seed = int(subset_seed)

        # A raw recursive NAIP tree can contain millions of tiles. Never retain the
        # complete tree in a worker-pickled Dataset: Windows spawn would serialize
        # several gigabytes and commonly fail with ``pickle data was truncated``.
        # When max_files is set, _resolve_files uses bounded reservoir sampling.
        self.files = FileIndex(self._resolve_files(files))

        self._expected_channels = channels or self._inspect_channel_count(str(self.files[0]))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = str(self.files[index])
        image = self._load_image(path)
        actual_channels = int(image.shape[0])
        if self.channels is not None and actual_channels != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels for '{path}', found {actual_channels}."
            )
        if self.channels is None and actual_channels != self._expected_channels:
            raise ValueError(
                "Inconsistent channel count: "
                f"expected {self._expected_channels}, found {actual_channels} in '{path}'."
            )

        _, height, width = image.shape
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(
                f"crop_size={self.crop_size} exceeds image dimensions "
                f"{height}x{width} for '{path}'."
            )

        if self.augment:
            image = self._random_crop(image)
            if _random_bool():
                image = torch.flip(image, dims=(2,))
            if _random_bool():
                image = torch.flip(image, dims=(1,))
            image = _color_jitter(image)
        else:
            image = self._center_crop(image)

        return {
            "image": image.to(dtype=self.dtype),
            "path": str(path),
        }

    def _resolve_files(self, files: Sequence[str | Path] | None) -> list[str]:
        import os
        if files is not None:
            candidates: Iterable[str] = (str(path) for path in files)
        elif self.catalog_path is not None:
            if not self.catalog_path.exists():
                raise FileNotFoundError(f"Catalog file not found: {self.catalog_path}")
            df = pd.read_csv(self.catalog_path)
            if "path" not in df.columns:
                raise ValueError(f"Catalog CSV {self.catalog_path} must have a 'path' column.")
            candidates = (str(path) for path in df["path"].astype(str).tolist())
        else:
            candidates = (
                str(path)
                for path in self.root.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            )

        # Keep memory bounded for very large roots. Reservoir sampling gives every
        # discovered file an equal chance of inclusion without first materializing a
        # 10M-element list. Sorting the final sample makes the selected manifest
        # deterministic for a fixed subset_seed.
        if self.max_files is None:
            resolved = sorted(candidates)
        else:
            rng = random.Random(self.subset_seed)
            reservoir: list[str] = []
            for count, path in enumerate(candidates):
                if count < self.max_files:
                    reservoir.append(path)
                else:
                    replacement = rng.randrange(count + 1)
                    if replacement < self.max_files:
                        reservoir[replacement] = path
            resolved = sorted(reservoir)

        root_str = str(self.root)
        # Ensure all paths are absolute using fast os.path string operations
        resolved = [
            path if os.path.isabs(path) else os.path.join(root_str, path)
            for path in resolved
        ]

        if not resolved:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
            raise ValueError(
                f"No supported imagery found. Check root: {self.root} or catalog: {self.catalog_path}"
            )
        return resolved

    def _inspect_channel_count(self, path: str) -> int:
        import os
        _, ext = os.path.splitext(path)
        suffix = ext.lower()
        if suffix in {".jpg", ".jpeg", ".png"}:
            return 3
        if suffix in {".tif", ".tiff"}:
            try:
                import rasterio
            except ImportError as exc:
                raise RuntimeError(
                    "Loading multi-band TIFF imagery requires rasterio."
                ) from exc
            with rasterio.open(path) as src:
                return int(src.count)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return int(_array_to_chw(array, requested_channels=None, path=path).shape[0])

    def _load_image(self, path: str) -> torch.Tensor:
        import os
        _, ext = os.path.splitext(path)
        suffix = ext.lower()
        if suffix in {".jpg", ".jpeg", ".png"}:
            with Image.open(path) as source:
                image = source.convert("RGB")
                array = np.asarray(image).copy()
            chw = np.moveaxis(array, -1, 0)
        elif suffix in {".tif", ".tiff"}:
            try:
                import rasterio
            except ImportError as exc:
                raise RuntimeError(
                    "Loading multi-band TIFF imagery requires rasterio."
                ) from exc
            with rasterio.open(path) as src:
                chw = src.read()
        elif suffix == ".npy":
            array = np.load(path, allow_pickle=False)
            chw = _array_to_chw(array, requested_channels=self.channels, path=path)
        else:
            raise ValueError(f"Unsupported image extension '{suffix}' for '{path}'.")
        return _normalize_image_array(chw, path=path)

    def _random_crop(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        max_top = height - self.crop_size
        max_left = width - self.crop_size
        top = int(torch.randint(max_top + 1, (1,)).item()) if max_top else 0
        left = int(torch.randint(max_left + 1, (1,)).item()) if max_left else 0
        return image[:, top : top + self.crop_size, left : left + self.crop_size]

    def _center_crop(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        top = (height - self.crop_size) // 2
        left = (width - self.crop_size) // 2
        return image[:, top : top + self.crop_size, left : left + self.crop_size]


def get_dataloaders(
    root: str | Path,
    batch_size: int,
    val_fraction: float = 0.2,
    seed: int = 27,
    crop_size: int = 224,
    channels: int | None = None,
    precision: str = "fp32",
    num_workers: int = 4,
    prefetch_factor: int = 2,
    catalog_path: str | Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build leakage-resistant train and validation loaders for nested NAIP tiles."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be a positive integer.")

    catalog = NAIPDataset(
        root,
        crop_size=crop_size,
        channels=channels,
        precision=precision,
        augment=False,
        catalog_path=catalog_path,
    )
    train_files, val_files = _split_files_by_parent(
        catalog.files,
        root=catalog.root,
        val_fraction=val_fraction,
        seed=seed,
    )
    train_dataset = NAIPDataset(
        root,
        files=train_files,
        crop_size=crop_size,
        channels=channels,
        precision=precision,
        augment=True,
    )
    val_dataset = NAIPDataset(
        root,
        files=val_files,
        crop_size=crop_size,
        channels=channels,
        precision=precision,
        augment=False,
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


def _split_files_by_parent(
    files: Sequence[str],
    *,
    root: Path,
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    import os
    from collections import defaultdict
    grouped = defaultdict(list)
    root_str = str(root)
    
    for raw_path in files:
        path = os.fspath(raw_path)
        parent = os.path.dirname(path)
        group = parent if parent != root_str else path
        grouped[group].append(path)
        
    groups = sorted(grouped.keys())
    if len(groups) < 2:
        raise ValueError(
            "At least two source-image parent groups are required for a train/validation split."
        )

    rng = random.Random(seed)
    rng.shuffle(groups)
    val_count = min(len(groups) - 1, max(1, round(len(groups) * val_fraction)))
    val_groups = set(groups[:val_count])
    
    train_files = []
    val_files = []
    for group, paths in grouped.items():
        if group in val_groups:
            val_files.extend(paths)
        else:
            train_files.extend(paths)
            
    return sorted(train_files), sorted(val_files)


def _array_to_chw(
    array: np.ndarray,
    *,
    requested_channels: int | None,
    path: str,
) -> np.ndarray:
    if array.ndim == 2:
        if requested_channels not in {None, 1}:
            raise ValueError(f"Expected {requested_channels} channels for '{path}', found 1.")
        return array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D NPY array for '{path}', found shape {array.shape}.")

    if requested_channels is not None:
        first_matches = array.shape[0] == requested_channels
        last_matches = array.shape[-1] == requested_channels
        if first_matches and last_matches:
            raise ValueError(
                f"Ambiguous NPY layout for '{path}' with shape {array.shape}; "
                "both first and last axes match channels."
            )
        if first_matches:
            return array
        if last_matches:
            return np.moveaxis(array, -1, 0)
        raise ValueError(
            f"Expected {requested_channels} channels for '{path}', found shape {array.shape}."
        )

    first_plausible = array.shape[0] <= 16 and array.shape[1] > array.shape[0] and array.shape[2] > array.shape[0]
    last_plausible = array.shape[-1] <= 16 and array.shape[0] > array.shape[-1] and array.shape[1] > array.shape[-1]
    if first_plausible == last_plausible:
        raise ValueError(
            f"Ambiguous NPY layout for '{path}' with shape {array.shape}; "
            "pass channels explicitly."
        )
    return array if first_plausible else np.moveaxis(array, -1, 0)


def _normalize_image_array(array: np.ndarray, *, path: str) -> torch.Tensor:
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Image array must be numeric for '{path}', found {array.dtype}.")
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        normalized = array.astype(np.float32) / float(info.max)
    else:
        normalized = array.astype(np.float32, copy=False)
        if not np.isfinite(normalized).all():
            raise ValueError(f"Floating image contains NaN or infinity: '{path}'.")
        minimum = float(normalized.min())
        maximum = float(normalized.max())
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                f"Floating image values must already be normalized to [0, 1] for '{path}'; "
                f"found range [{minimum}, {maximum}]."
            )
    return torch.from_numpy(np.ascontiguousarray(normalized))


def _color_jitter(image: torch.Tensor) -> torch.Tensor:
    brightness = _uniform(0.8, 1.2)
    contrast = _uniform(0.8, 1.2)
    image = image * brightness
    mean = image.mean()
    image = (image - mean) * contrast + mean

    if image.shape[0] == 3:
        saturation = _uniform(0.8, 1.2)
        grayscale = (
            image[0:1] * 0.2989
            + image[1:2] * 0.5870
            + image[2:3] * 0.1140
        )
        image = grayscale + saturation * (image - grayscale)
        image = _adjust_rgb_hue(image, _uniform(-0.05, 0.05))
    return image.clamp_(0.0, 1.0)


def _adjust_rgb_hue(image: torch.Tensor, hue_factor: float) -> torch.Tensor:
    if math.isclose(hue_factor, 0.0, abs_tol=1e-8):
        return image
    hsv = _rgb_to_hsv(image.clamp(0.0, 1.0))
    hsv[0] = torch.remainder(hsv[0] + hue_factor, 1.0)
    return _hsv_to_rgb(hsv)


def _rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    red, green, blue = image.unbind(dim=0)
    maximum, max_indices = image.max(dim=0)
    minimum = image.min(dim=0).values
    delta = maximum - minimum
    saturation = torch.where(maximum > 0, delta / maximum.clamp_min(1e-8), torch.zeros_like(maximum))
    hue = torch.zeros_like(maximum)
    nonzero = delta > 1e-8
    hue = torch.where(
        nonzero & (max_indices == 0),
        torch.remainder((green - blue) / delta.clamp_min(1e-8), 6.0),
        hue,
    )
    hue = torch.where(
        nonzero & (max_indices == 1),
        (blue - red) / delta.clamp_min(1e-8) + 2.0,
        hue,
    )
    hue = torch.where(
        nonzero & (max_indices == 2),
        (red - green) / delta.clamp_min(1e-8) + 4.0,
        hue,
    )
    return torch.stack((hue / 6.0, saturation, maximum))


def _hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    hue, saturation, value = image.unbind(dim=0)
    sector = torch.floor(hue * 6.0).to(torch.int64)
    fraction = hue * 6.0 - sector.to(hue.dtype)
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    sector = torch.remainder(sector, 6)

    red = torch.where(
        sector == 0,
        value,
        torch.where(
            sector == 1,
            q,
            torch.where(sector == 2, p, torch.where(sector == 3, p, torch.where(sector == 4, t, value))),
        ),
    )
    green = torch.where(
        sector == 0,
        t,
        torch.where(
            sector == 1,
            value,
            torch.where(sector == 2, value, torch.where(sector == 3, q, torch.where(sector == 4, p, p))),
        ),
    )
    blue = torch.where(
        sector == 0,
        p,
        torch.where(
            sector == 1,
            p,
            torch.where(sector == 2, t, torch.where(sector == 3, value, torch.where(sector == 4, value, q))),
        ),
    )
    return torch.stack((red, green, blue))


def _random_bool() -> bool:
    return bool(torch.rand(()) < 0.5)


def _uniform(low: float, high: float) -> float:
    return float(torch.empty(()).uniform_(low, high).item())


def create_dataset_catalog(root: str | Path, output_path: str | Path) -> None:
    """Scan a directory for supported imagery and save the paths to a CSV catalog."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f"Catalog root must be a directory: {root_path}")

    print(f"Scanning {root_path} for imagery...")
    # Using a list comprehension for speed, but filtering as we go
    files = []
    for p in root_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            # Save path relative to root to make the catalog portable
            files.append(str(p.relative_to(root_path)))

    files.sort()
    df = pd.DataFrame({"path": files})
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Catalog created with {len(files)} files at {output_path}")
