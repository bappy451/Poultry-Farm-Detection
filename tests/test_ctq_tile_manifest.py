from __future__ import annotations

import csv

import pytest

from scripts.ctq_tile_manifest import load_manifest, validate_cross_tile_manifest


def write_manifest(tmp_path, rows):
    for row in rows:
        (tmp_path / row["tile_path"]).write_bytes(b"image")
    path = tmp_path / "tiles.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tile_path", "source_raster", "row_offset", "col_offset", "tile_width", "tile_height"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_manifest_accepts_same_source_overlap(tmp_path):
    path = write_manifest(tmp_path, [
        {"tile_path": "a.jpg", "source_raster": "raster_a", "row_offset": 0, "col_offset": 0, "tile_width": 512, "tile_height": 512},
        {"tile_path": "b.jpg", "source_raster": "raster_a", "row_offset": 0, "col_offset": 256, "tile_width": 512, "tile_height": 512},
    ])
    assert validate_cross_tile_manifest(path) == {"tiles": 2, "source_rasters": 1, "overlap_pairs": 1}


def test_manifest_rejects_unknown_source_id(tmp_path):
    path = write_manifest(tmp_path, [{"tile_path": "a.jpg", "source_raster": "unknown", "row_offset": 0, "col_offset": 0, "tile_width": 512, "tile_height": 512}])
    with pytest.raises(ValueError, match="source_raster"):
        load_manifest(path)


def test_manifest_rejects_non_overlapping_tiles(tmp_path):
    path = write_manifest(tmp_path, [
        {"tile_path": "a.jpg", "source_raster": "raster_a", "row_offset": 0, "col_offset": 0, "tile_width": 512, "tile_height": 512},
        {"tile_path": "b.jpg", "source_raster": "raster_a", "row_offset": 0, "col_offset": 512, "tile_width": 512, "tile_height": 512},
    ])
    with pytest.raises(ValueError, match="overlapping"):
        validate_cross_tile_manifest(path)
