from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from os.path import commonpath
from pathlib import Path
from typing import Iterable, Sequence


SUPPORTED_TILE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})

_FILENAME_PATTERNS = (
    re.compile(r"^(?P<raster>.+?)__row_(?P<row>\d+)__col_(?P<col>\d+)$"),
    re.compile(r"^(?P<raster>.+?)__r(?P<row>\d+)__c(?P<col>\d+)$"),
    re.compile(r"^(?P<raster>.+?)_y(?P<row>\d+)_x(?P<col>\d+)$"),
)


@dataclass(frozen=True, slots=True)
class TileMetadata:
    path: Path
    raster_id: str
    row_offset: int
    col_offset: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class AdjacencyRecord:
    tile_a: str
    tile_b: str
    source_raster: str
    row_offset: int
    col_offset: int
    overlap_x1: int
    overlap_y1: int
    overlap_x2: int
    overlap_y2: int


def parse_tile_filename(path: str | Path, *, tile_size: int = 512) -> TileMetadata:
    candidate = Path(path)
    stem = candidate.stem
    for pattern in _FILENAME_PATTERNS:
        match = pattern.match(stem)
        if match is None:
            continue
        raster_id = match.group("raster")
        return TileMetadata(
            path=candidate,
            raster_id=raster_id,
            row_offset=int(match.group("row")),
            col_offset=int(match.group("col")),
            height=int(tile_size),
            width=int(tile_size),
        )
    raise ValueError(
        "Could not parse tile offsets from filename. "
        "Expected patterns like 'raster__row_256__col_512.jpg'."
    )


def discover_tiles(root: str | Path, *, tile_size: int = 512) -> list[TileMetadata]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Tile root does not exist: {root_path}")
    if not root_path.is_dir():
        raise ValueError(f"Tile root must be a directory: {root_path}")

    tiles = [
        parse_tile_filename(path, tile_size=tile_size)
        for path in sorted(root_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_TILE_EXTENSIONS
    ]
    if not tiles:
        raise ValueError(f"No supported image tiles found under {root_path}")
    return tiles


def build_adjacency_records(
    root_or_tiles: str | Path | Sequence[TileMetadata],
    *,
    overlap_ratio: float = 0.5,
    tile_size: int = 512,
) -> list[AdjacencyRecord]:
    if not 0.0 < overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be between 0 and 1.")

    if isinstance(root_or_tiles, (str, Path)):
        root_path = Path(root_or_tiles).expanduser().resolve()
        tiles = discover_tiles(root_path, tile_size=tile_size)
        relative_root = root_path
    else:
        tiles = list(root_or_tiles)
        if not tiles:
            return []
        relative_root = _common_root([tile.path for tile in tiles])

    stride = int(round(tile_size * (1.0 - overlap_ratio)))
    if stride <= 0 or stride >= tile_size:
        raise ValueError("overlap_ratio must produce a stride strictly between 0 and tile_size.")

    grouped: dict[str, dict[tuple[int, int], TileMetadata]] = {}
    for tile in tiles:
        grouped.setdefault(tile.raster_id, {})[(tile.row_offset, tile.col_offset)] = tile

    records: list[AdjacencyRecord] = []
    for raster_id in sorted(grouped):
        positions = grouped[raster_id]
        for (row_offset, col_offset), tile_a in sorted(positions.items()):
            for delta_row, delta_col in ((0, stride), (stride, 0), (stride, stride)):
                tile_b = positions.get((row_offset + delta_row, col_offset + delta_col))
                if tile_b is None:
                    continue
                overlap_box = _overlap_box(
                    tile_size=tile_size,
                    row_offset=tile_b.row_offset - tile_a.row_offset,
                    col_offset=tile_b.col_offset - tile_a.col_offset,
                )
                records.append(
                    AdjacencyRecord(
                        tile_a=_relative_or_name(tile_a.path, relative_root),
                        tile_b=_relative_or_name(tile_b.path, relative_root),
                        source_raster=raster_id,
                        row_offset=tile_b.row_offset - tile_a.row_offset,
                        col_offset=tile_b.col_offset - tile_a.col_offset,
                        overlap_x1=overlap_box[0],
                        overlap_y1=overlap_box[1],
                        overlap_x2=overlap_box[2],
                        overlap_y2=overlap_box[3],
                    )
                )
    return records


def write_adjacency_index_csv(
    input_dir: str | Path,
    output_csv: str | Path,
    *,
    overlap_ratio: float = 0.5,
    tile_size: int = 512,
) -> Path:
    records = build_adjacency_records(
        input_dir,
        overlap_ratio=overlap_ratio,
        tile_size=tile_size,
    )
    output_path = Path(output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tile_a",
                "tile_b",
                "source_raster",
                "row_offset",
                "col_offset",
                "overlap_x1",
                "overlap_y1",
                "overlap_x2",
                "overlap_y2",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return output_path


def _overlap_box(*, tile_size: int, row_offset: int, col_offset: int) -> tuple[int, int, int, int]:
    left = max(0, col_offset)
    top = max(0, row_offset)
    right = min(tile_size, tile_size + col_offset)
    bottom = min(tile_size, tile_size + row_offset)
    if left >= right or top >= bottom:
        raise ValueError("Tiles do not overlap.")
    return (left, top, right, bottom)


def _common_root(paths: Iterable[Path]) -> Path:
    resolved = [Path(path).expanduser().resolve() for path in paths]
    if not resolved:
        raise ValueError("paths must contain at least one item.")
    if len(resolved) == 1:
        return resolved[0].parent
    return Path(commonpath([str(path) for path in resolved]))


def _relative_or_name(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name
