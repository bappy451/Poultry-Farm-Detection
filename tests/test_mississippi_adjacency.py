from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from poultry_geo.mississippi_adjacency import (
    build_adjacency_records,
    parse_tile_filename,
    write_adjacency_index_csv,
)


def _write_tile(path: Path, *, size: tuple[int, int] = (512, 512), value: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(value, value, value)).save(path)


def test_parse_tile_filename_recovers_raster_and_offsets() -> None:
    metadata = parse_tile_filename(Path("tiles/raster_alpha__row_256__col_512.jpg"))

    assert metadata.raster_id == "raster_alpha"
    assert metadata.row_offset == 256
    assert metadata.col_offset == 512
    assert metadata.height == 512
    assert metadata.width == 512


def test_build_adjacency_records_finds_expected_overlap_pairs(tmp_path: Path) -> None:
    _write_tile(tmp_path / "raster_a__row_0__col_0.jpg")
    _write_tile(tmp_path / "raster_a__row_0__col_256.jpg")
    _write_tile(tmp_path / "raster_a__row_256__col_256.jpg")
    _write_tile(tmp_path / "raster_a__row_800__col_800.jpg")

    records = build_adjacency_records(tmp_path, overlap_ratio=0.5, tile_size=512)

    assert len(records) == 3
    left_right = next(
        record
        for record in records
        if record.tile_a.endswith("raster_a__row_0__col_0.jpg")
        and record.tile_b.endswith("raster_a__row_0__col_256.jpg")
    )
    assert left_right.row_offset == 0
    assert left_right.col_offset == 256
    assert left_right.overlap_x1 == 256
    assert left_right.overlap_y1 == 0
    assert left_right.overlap_x2 == 512
    assert left_right.overlap_y2 == 512


def test_write_adjacency_index_csv_persists_expected_columns(tmp_path: Path) -> None:
    _write_tile(tmp_path / "raster_b__row_0__col_0.jpg")
    _write_tile(tmp_path / "raster_b__row_256__col_256.jpg")

    output_csv = tmp_path / "adjacency.csv"
    written = write_adjacency_index_csv(
        tmp_path,
        output_csv,
        overlap_ratio=0.5,
        tile_size=512,
    )

    assert written == output_csv
    frame = pd.read_csv(output_csv)
    assert frame.columns.tolist() == [
        "tile_a",
        "tile_b",
        "source_raster",
        "row_offset",
        "col_offset",
        "overlap_x1",
        "overlap_y1",
        "overlap_x2",
        "overlap_y2",
    ]
    assert frame.shape == (1, 9)
