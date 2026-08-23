"""Validate explicit tile geometry before enabling cross-tile CTQ supervision."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = ("tile_path", "source_raster", "row_offset", "col_offset", "tile_width", "tile_height")


@dataclass(frozen=True, slots=True)
class TileRecord:
    tile_path: str
    source_raster: str
    row_offset: int
    col_offset: int
    tile_width: int
    tile_height: int

    @property
    def right(self) -> int:
        return self.col_offset + self.tile_width

    @property
    def bottom(self) -> int:
        return self.row_offset + self.tile_height


def load_manifest(path: str | Path, *, root: str | Path | None = None) -> list[TileRecord]:
    """Read an explicit, source-aware tile manifest and validate every row."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    image_root = Path(root).expanduser().resolve() if root is not None else manifest_path.parent
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(REQUIRED_COLUMNS) - set(reader.fieldnames):
            missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames or ()))
            raise ValueError(f"Tile manifest is missing columns: {missing}")
        records = [_parse_record(row, index + 2, image_root) for index, row in enumerate(reader)]
    if not records:
        raise ValueError("Tile manifest contains no records")
    identities = [(record.source_raster, record.row_offset, record.col_offset) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Tile manifest contains duplicate source_raster,row_offset,col_offset rows")
    return records


def _parse_record(row: dict[str, str], line_number: int, image_root: Path) -> TileRecord:
    try:
        tile_path = str(row["tile_path"]).strip()
        source_raster = str(row["source_raster"]).strip()
        record = TileRecord(
            tile_path=tile_path,
            source_raster=source_raster,
            row_offset=int(str(row["row_offset"]).strip()),
            col_offset=int(str(row["col_offset"]).strip()),
            tile_width=int(str(row["tile_width"]).strip()),
            tile_height=int(str(row["tile_height"]).strip()),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid tile-manifest row {line_number}") from exc
    if not record.tile_path or not record.source_raster or record.source_raster.lower() in {"unknown", "__unknown__"}:
        raise ValueError(f"Tile-manifest row {line_number} must include a real source_raster identifier")
    if record.row_offset < 0 or record.col_offset < 0 or record.tile_width <= 0 or record.tile_height <= 0:
        raise ValueError(f"Tile-manifest row {line_number} has invalid offsets or dimensions")
    image_path = Path(record.tile_path)
    if not image_path.is_absolute():
        image_path = image_root / image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"Tile-manifest row {line_number} references missing tile: {image_path}")
    return record


def overlapping_pairs(records: Iterable[TileRecord]) -> list[tuple[TileRecord, TileRecord]]:
    """Return only same-source tile pairs with non-zero pixel overlap."""
    ordered = sorted(records, key=lambda item: (item.source_raster, item.row_offset, item.col_offset, item.tile_path))
    pairs: list[tuple[TileRecord, TileRecord]] = []
    for index, first in enumerate(ordered):
        for second in ordered[index + 1:]:
            if second.source_raster != first.source_raster:
                break
            if second.row_offset >= first.bottom:
                break
            if max(first.col_offset, second.col_offset) < min(first.right, second.right) and max(first.row_offset, second.row_offset) < min(first.bottom, second.bottom):
                pairs.append((first, second))
    return pairs


def validate_cross_tile_manifest(path: str | Path, *, root: str | Path | None = None, min_pairs: int = 1) -> dict[str, int]:
    if min_pairs < 1:
        raise ValueError("min_pairs must be positive")
    records = load_manifest(path, root=root)
    pairs = overlapping_pairs(records)
    if len(pairs) < min_pairs:
        raise ValueError(f"Cross-tile stage requires at least {min_pairs} overlapping same-source tile pairs; found {len(pairs)}")
    return {"tiles": len(records), "source_rasters": len({item.source_raster for item in records}), "overlap_pairs": len(pairs)}
