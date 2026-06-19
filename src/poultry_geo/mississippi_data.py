from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader

from poultry_geo.mississippi_adjacency import parse_tile_filename
from poultry_geo.naip_data import NAIPDataset


@dataclass(frozen=True, slots=True)
class OverlapPairRecord:
    tile_a: Path
    tile_b: Path
    source_raster: str
    row_offset: int
    col_offset: int
    overlap_box_a: tuple[int, int, int, int]


class OverlapAwareNAIPDataset(NAIPDataset):
    """Overlap-aware SSL dataset for MississippiFM training."""

    def __init__(
        self,
        root: str | Path,
        adjacency_csv: str | Path,
        *,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 27,
        channels: int | None = None,
        precision: str = "fp32",
        augment: bool = True,
        high_res: bool = False,
        low_res_sizes: Sequence[int] = (256, 320),
        patch_size: int = 16,
        tile_size: int = 512,
        epoch_size: int | None = None,
    ) -> None:
        self.seed = int(seed)
        self.patch_size = int(patch_size)
        self.tile_size = int(tile_size)
        self.low_res_sizes = tuple(int(size) for size in low_res_sizes)
        self.high_res = bool(high_res)
        self._requested_split = split
        self._val_fraction = float(val_fraction)
        self._user_augment = bool(augment)
        super().__init__(
            root,
            crop_size=self.tile_size,
            channels=channels,
            precision=precision,
            augment=False,
        )

        self.adjacency_csv = Path(adjacency_csv).expanduser().resolve()
        if not self.adjacency_csv.exists():
            raise FileNotFoundError(f"Adjacency CSV does not exist: {self.adjacency_csv}")

        self.tile_metadata = {path: parse_tile_filename(path, tile_size=self.tile_size) for path in self.files}
        self._tiles_by_raster: dict[str, list[Path]] = {}
        for path, metadata in self.tile_metadata.items():
            self._tiles_by_raster.setdefault(metadata.raster_id, []).append(path)
        self._all_raster_ids = sorted(self._tiles_by_raster)
        self._split_map = _split_raster_ids(self._all_raster_ids, val_fraction=self._val_fraction, seed=self.seed)
        self.active_raster_ids: set[str] = set()
        self.active_tiles: list[Path] = []
        self.active_pairs: list[OverlapPairRecord] = []
        self.active_pair_count = 0
        self.set_split(split)
        self.epoch_size = int(epoch_size or max(4, len(self.active_tiles)))

    def preload_tile_metadata(self) -> dict[str, dict[str, int | str]]:
        return {
            path.as_posix(): {
                "raster_id": metadata.raster_id,
                "row_offset": metadata.row_offset,
                "col_offset": metadata.col_offset,
            }
            for path, metadata in self.tile_metadata.items()
        }

    def set_split(self, split: str) -> None:
        normalized = split.strip().lower()
        if normalized not in {"train", "val", "validation"}:
            raise ValueError("split must be 'train' or 'val'.")
        if normalized == "validation":
            normalized = "val"
        self._requested_split = normalized
        self.active_raster_ids = set(self._split_map[normalized])
        self.active_tiles = sorted(
            path
            for path, metadata in self.tile_metadata.items()
            if metadata.raster_id in self.active_raster_ids
        )
        self.active_pairs = _load_active_pairs(
            self.adjacency_csv,
            root=self.root,
            active_raster_ids=self.active_raster_ids,
        )
        self.active_pair_count = len(self.active_pairs)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + index)
        pair_type = self._pair_type_for_index(index)
        if pair_type == "overlap" and self.active_pairs:
            sample = self._sample_overlap_pair(index, rng)
        elif pair_type == "same_tile":
            sample = self._sample_same_tile_pair(index, rng)
        else:
            sample = self._sample_random_pair(index, rng)
        sample["pair_type"] = pair_type if pair_type != "overlap" or self.active_pairs else "random"
        return sample

    def _pair_type_for_index(self, index: int) -> str:
        schedule_index = index % 4
        if schedule_index in {0, 1} and self.active_pairs:
            return "overlap"
        if schedule_index == 2:
            return "same_tile"
        return "random"

    def _sample_overlap_pair(self, index: int, rng: random.Random) -> dict[str, Any]:
        record = self.active_pairs[index % len(self.active_pairs)]
        image_a = self._load_image(record.tile_a)
        image_b = self._load_image(record.tile_b)
        if self.high_res:
            view_a = image_a
            view_b = image_b
            crop_origin_a = (0, 0)
            crop_origin_b = (0, 0)
            overlap_box_a = record.overlap_box_a
            overlap_box_b = _shift_overlap_box(
                record.overlap_box_a,
                row_delta=-record.row_offset,
                col_delta=-record.col_offset,
            )
        else:
            crop_size = rng.choice(self.low_res_sizes)
            point_a = _sample_shared_world_point(record.overlap_box_a, rng=rng)
            top_a, left_a = _sample_crop_origin_covering_point(
                point_a,
                crop_size=crop_size,
                tile_size=self.tile_size,
                rng=rng,
            )
            point_b = (point_a[0] - record.col_offset, point_a[1] - record.row_offset)
            top_b, left_b = _sample_crop_origin_covering_point(
                point_b,
                crop_size=crop_size,
                tile_size=self.tile_size,
                rng=rng,
            )
            view_a = image_a[:, top_a : top_a + crop_size, left_a : left_a + crop_size]
            view_b = image_b[:, top_b : top_b + crop_size, left_b : left_b + crop_size]
            crop_origin_a = (top_a, left_a)
            crop_origin_b = (top_b, left_b)
            overlap_box_a = _clip_box_to_crop(record.overlap_box_a, crop_origin_a, crop_size)
            overlap_box_b = _clip_box_to_crop(
                _shift_overlap_box(record.overlap_box_a, row_delta=-record.row_offset, col_delta=-record.col_offset),
                crop_origin_b,
                crop_size,
            )
        view_a, view_b, overlap_box_a, overlap_box_b = self._apply_pair_augmentations(
            view_a,
            view_b,
            overlap_box_a,
            overlap_box_b,
            rng=rng,
        )
        patch_indices_a = compute_patch_indices_for_overlap(
            overlap_box_a,
            image_size=(int(view_a.shape[1]), int(view_a.shape[2])),
            patch_size=self.patch_size,
        )
        patch_indices_b = compute_patch_indices_for_overlap(
            overlap_box_b,
            image_size=(int(view_b.shape[1]), int(view_b.shape[2])),
            patch_size=self.patch_size,
        )
        return {
            "image_a": view_a.to(dtype=self.dtype),
            "image_b": view_b.to(dtype=self.dtype),
            "overlap_box_a": overlap_box_a,
            "overlap_box_b": overlap_box_b,
            "patch_indices_a": patch_indices_a,
            "patch_indices_b": patch_indices_b,
            "crop_origin_a": crop_origin_a,
            "crop_origin_b": crop_origin_b,
            "path_a": str(record.tile_a),
            "path_b": str(record.tile_b),
        }

    def _sample_same_tile_pair(self, index: int, rng: random.Random) -> dict[str, Any]:
        tile = self.active_tiles[index % len(self.active_tiles)]
        image = self._load_image(tile)
        view_a, crop_origin_a = self._make_single_view(image, rng)
        view_b, crop_origin_b = self._make_single_view(image, rng)
        return _empty_overlap_payload(
            image_a=view_a.to(dtype=self.dtype),
            image_b=view_b.to(dtype=self.dtype),
            crop_origin_a=crop_origin_a,
            crop_origin_b=crop_origin_b,
            path_a=str(tile),
            path_b=str(tile),
        )

    def _sample_random_pair(self, index: int, rng: random.Random) -> dict[str, Any]:
        if len(self.active_tiles) == 1:
            return self._sample_same_tile_pair(index, rng)
        tile_a = self.active_tiles[index % len(self.active_tiles)]
        tile_b = self.active_tiles[(index + 1) % len(self.active_tiles)]
        image_a = self._load_image(tile_a)
        image_b = self._load_image(tile_b)
        view_a, crop_origin_a = self._make_single_view(image_a, rng)
        view_b, crop_origin_b = self._make_single_view(image_b, rng)
        return _empty_overlap_payload(
            image_a=view_a.to(dtype=self.dtype),
            image_b=view_b.to(dtype=self.dtype),
            crop_origin_a=crop_origin_a,
            crop_origin_b=crop_origin_b,
            path_a=str(tile_a),
            path_b=str(tile_b),
        )

    def _make_single_view(self, image: torch.Tensor, rng: random.Random) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.high_res:
            view = image
            origin = (0, 0)
        else:
            crop_size = rng.choice(self.low_res_sizes)
            _, height, width = image.shape
            top = rng.randint(0, height - crop_size)
            left = rng.randint(0, width - crop_size)
            view = image[:, top : top + crop_size, left : left + crop_size]
            origin = (top, left)
        if self._user_augment:
            if rng.random() < 0.5:
                view = torch.flip(view, dims=(2,))
            if rng.random() < 0.5:
                view = torch.flip(view, dims=(1,))
        return view, origin

    def _apply_pair_augmentations(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
        overlap_box_a: tuple[int, int, int, int],
        overlap_box_b: tuple[int, int, int, int],
        *,
        rng: random.Random,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int], tuple[int, int, int, int]]:
        if not self._user_augment:
            return view_a, view_b, overlap_box_a, overlap_box_b

        if rng.random() < 0.5:
            view_a = torch.flip(view_a, dims=(2,))
            view_b = torch.flip(view_b, dims=(2,))
            overlap_box_a = _flip_box_horizontally(overlap_box_a, width=int(view_a.shape[2]))
            overlap_box_b = _flip_box_horizontally(overlap_box_b, width=int(view_b.shape[2]))
        if rng.random() < 0.5:
            view_a = torch.flip(view_a, dims=(1,))
            view_b = torch.flip(view_b, dims=(1,))
            overlap_box_a = _flip_box_vertically(overlap_box_a, height=int(view_a.shape[1]))
            overlap_box_b = _flip_box_vertically(overlap_box_b, height=int(view_b.shape[1]))
        brightness = 0.9 + 0.2 * rng.random()
        view_a = torch.clamp(view_a * brightness, 0.0, 1.0)
        view_b = torch.clamp(view_b * brightness, 0.0, 1.0)
        return view_a, view_b, overlap_box_a, overlap_box_b


def compute_patch_indices_for_overlap(
    overlap_box: tuple[int, int, int, int],
    *,
    image_size: int | tuple[int, int],
    patch_size: int,
) -> torch.Tensor:
    if isinstance(image_size, int):
        height = width = int(image_size)
    else:
        height, width = int(image_size[0]), int(image_size[1])
    x1, y1, x2, y2 = (int(value) for value in overlap_box)
    if x2 <= x1 or y2 <= y1:
        return torch.empty(0, dtype=torch.long)
    grid_height = height // patch_size
    grid_width = width // patch_size
    start_col = max(0, x1 // patch_size)
    end_col = min(grid_width, (x2 + patch_size - 1) // patch_size)
    start_row = max(0, y1 // patch_size)
    end_row = min(grid_height, (y2 + patch_size - 1) // patch_size)
    indices = [
        row_index * grid_width + col_index
        for row_index in range(start_row, end_row)
        for col_index in range(start_col, end_col)
    ]
    return torch.tensor(indices, dtype=torch.long)


def collate_overlap_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image_a": torch.stack([item["image_a"] for item in batch], dim=0),
        "image_b": torch.stack([item["image_b"] for item in batch], dim=0),
        "pair_type": [item["pair_type"] for item in batch],
        "patch_indices_a": [item["patch_indices_a"] for item in batch],
        "patch_indices_b": [item["patch_indices_b"] for item in batch],
        "overlap_box_a": [item["overlap_box_a"] for item in batch],
        "overlap_box_b": [item["overlap_box_b"] for item in batch],
        "crop_origin_a": [item["crop_origin_a"] for item in batch],
        "crop_origin_b": [item["crop_origin_b"] for item in batch],
        "path_a": [item["path_a"] for item in batch],
        "path_b": [item["path_b"] for item in batch],
    }


def build_overlap_aware_dataloaders(
    root: str | Path,
    adjacency_csv: str | Path,
    *,
    batch_size: int,
    val_fraction: float = 0.2,
    seed: int = 27,
    channels: int | None = None,
    precision: str = "fp32",
    augment: bool = True,
    high_res: bool = False,
    low_res_sizes: Sequence[int] = (256,),
    patch_size: int = 16,
    tile_size: int = 512,
    epoch_size: int | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = OverlapAwareNAIPDataset(
        root,
        adjacency_csv,
        split="train",
        val_fraction=val_fraction,
        seed=seed,
        channels=channels,
        precision=precision,
        augment=augment,
        high_res=high_res,
        low_res_sizes=low_res_sizes,
        patch_size=patch_size,
        tile_size=tile_size,
        epoch_size=epoch_size,
    )
    val_dataset = OverlapAwareNAIPDataset(
        root,
        adjacency_csv,
        split="val",
        val_fraction=val_fraction,
        seed=seed,
        channels=channels,
        precision=precision,
        augment=False,
        high_res=high_res,
        low_res_sizes=low_res_sizes,
        patch_size=patch_size,
        tile_size=tile_size,
        epoch_size=epoch_size,
    )
    loader_kwargs = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "collate_fn": collate_overlap_batch,
        "shuffle": False,
    }
    return (
        DataLoader(train_dataset, **loader_kwargs),
        DataLoader(val_dataset, **loader_kwargs),
    )


def _split_raster_ids(raster_ids: Sequence[str], *, val_fraction: float, seed: int) -> dict[str, list[str]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    ids = list(sorted(raster_ids))
    if len(ids) < 2:
        raise ValueError("At least two raster groups are required for train/validation splitting.")
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_count = min(len(ids) - 1, max(1, round(len(ids) * val_fraction)))
    val_ids = sorted(ids[:val_count])
    train_ids = sorted(ids[val_count:])
    return {"train": train_ids, "val": val_ids}


def _load_active_pairs(
    adjacency_csv: Path,
    *,
    root: Path,
    active_raster_ids: set[str],
) -> list[OverlapPairRecord]:
    frame = pd.read_csv(adjacency_csv)
    records: list[OverlapPairRecord] = []
    for row in frame.itertuples(index=False):
        if str(row.source_raster) not in active_raster_ids:
            continue
        records.append(
            OverlapPairRecord(
                tile_a=(root / str(row.tile_a)).resolve(),
                tile_b=(root / str(row.tile_b)).resolve(),
                source_raster=str(row.source_raster),
                row_offset=int(row.row_offset),
                col_offset=int(row.col_offset),
                overlap_box_a=(
                    int(row.overlap_x1),
                    int(row.overlap_y1),
                    int(row.overlap_x2),
                    int(row.overlap_y2),
                ),
            )
        )
    return records


def _sample_world_aligned_crop(
    overlap_box_a: tuple[int, int, int, int],
    *,
    crop_size: int,
    tile_size: int,
    rng: random.Random,
) -> tuple[int, int]:
    x1, y1, x2, y2 = overlap_box_a
    min_top = max(0, y2 - crop_size)
    max_top = min(y1, tile_size - crop_size)
    min_left = max(0, x2 - crop_size)
    max_left = min(x1, tile_size - crop_size)
    if max_top < min_top or max_left < min_left:
        raise ValueError("Crop size is larger than the shared overlap extent.")
    top = rng.randint(min_top, max_top)
    left = rng.randint(min_left, max_left)
    return top, left


def _sample_shared_world_point(
    overlap_box: tuple[int, int, int, int],
    *,
    rng: random.Random,
) -> tuple[int, int]:
    x1, y1, x2, y2 = overlap_box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Overlap box must have positive area.")
    return (
        rng.randint(x1, x2 - 1),
        rng.randint(y1, y2 - 1),
    )


def _sample_crop_origin_covering_point(
    point: tuple[int, int],
    *,
    crop_size: int,
    tile_size: int,
    rng: random.Random,
) -> tuple[int, int]:
    x, y = point
    if crop_size > tile_size:
        raise ValueError("crop_size cannot exceed tile_size.")
    max_origin = tile_size - crop_size
    min_top = max(0, y - crop_size + 1)
    max_top = min(y, max_origin)
    min_left = max(0, x - crop_size + 1)
    max_left = min(x, max_origin)
    if max_top < min_top or max_left < min_left:
        raise ValueError("Could not sample a crop covering the shared point.")
    return (
        rng.randint(min_top, max_top),
        rng.randint(min_left, max_left),
    )


def _clip_box_to_crop(
    box: tuple[int, int, int, int],
    crop_origin: tuple[int, int],
    crop_size: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    top, left = crop_origin
    clipped = (
        max(0, x1 - left),
        max(0, y1 - top),
        min(crop_size, x2 - left),
        min(crop_size, y2 - top),
    )
    return clipped


def _shift_overlap_box(
    box: tuple[int, int, int, int],
    *,
    row_delta: int,
    col_delta: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (x1 + col_delta, y1 + row_delta, x2 + col_delta, y2 + row_delta)


def _flip_box_horizontally(box: tuple[int, int, int, int], *, width: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (width - x2, y1, width - x1, y2)


def _flip_box_vertically(box: tuple[int, int, int, int], *, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (x1, height - y2, x2, height - y1)


def _empty_overlap_payload(
    *,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    crop_origin_a: tuple[int, int],
    crop_origin_b: tuple[int, int],
    path_a: str,
    path_b: str,
) -> dict[str, Any]:
    return {
        "image_a": image_a,
        "image_b": image_b,
        "overlap_box_a": (-1, -1, -1, -1),
        "overlap_box_b": (-1, -1, -1, -1),
        "patch_indices_a": torch.empty(0, dtype=torch.long),
        "patch_indices_b": torch.empty(0, dtype=torch.long),
        "crop_origin_a": crop_origin_a,
        "crop_origin_b": crop_origin_b,
        "path_a": path_a,
        "path_b": path_b,
    }
