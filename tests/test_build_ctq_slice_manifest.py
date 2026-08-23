from __future__ import annotations

from PIL import Image

from scripts.build_ctq_slice_manifest import build_manifest
from scripts.ctq_tile_manifest import load_manifest, validate_cross_tile_manifest


def test_build_manifest_recovers_source_and_pixel_offsets(tmp_path):
    source = tmp_path / "source_a"
    source.mkdir()
    Image.new("RGB", (512, 512)).save(source / "slice_0_0.jpg")
    Image.new("RGB", (512, 512)).save(source / "slice_0_1.jpg")
    output = build_manifest(tmp_path, tmp_path / "manifest.csv", stride=256)
    records = load_manifest(output, root=tmp_path)
    assert [(row.source_raster, row.row_offset, row.col_offset) for row in records] == [("source_a", 0, 0), ("source_a", 0, 256)]
    assert validate_cross_tile_manifest(output, root=tmp_path)["overlap_pairs"] == 1
