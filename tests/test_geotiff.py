from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from poultry_geo.geotiff import (
    compute_tile_starts,
    iter_tile_windows,
    slice_geotiff_collection_to_files,
    slice_geotiff_to_files,
    slice_geotiff_to_memory,
    stitch_geotiff_tiles,
)


def _write_test_raster(path: Path, data: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs="EPSG:26916",
        transform=from_origin(100.0, 200.0, 1.0, 1.0),
    ) as dst:
        dst.write(data)


def test_compute_tile_starts_aligns_last_tile_to_image_edge() -> None:
    assert compute_tile_starts(length=7, tile_size=4, stride=4) == [0, 3]
    assert compute_tile_starts(length=7, tile_size=4, stride=3) == [0, 3]
    assert compute_tile_starts(length=4, tile_size=4, stride=4) == [0]


def test_iter_tile_windows_covers_full_image_without_trailing_sliver() -> None:
    windows = list(iter_tile_windows(width=7, height=5, tile_size=4, stride=4))

    assert windows == [
        (0, 0, 0, 0, 4, 4),
        (0, 1, 3, 0, 7, 4),
        (1, 0, 0, 1, 4, 5),
        (1, 1, 3, 1, 7, 5),
    ]


def test_iter_tile_windows_rejects_stride_larger_than_tile_size() -> None:
    with pytest.raises(ValueError, match="stride must be less than or equal to tile_size"):
        list(iter_tile_windows(width=7, height=5, tile_size=4, stride=5))


def test_slice_geotiff_to_memory_aligns_edge_tiles(tmp_path: Path) -> None:
    raster_path = tmp_path / "input.tif"
    data = np.arange(3 * 5 * 7, dtype=np.uint8).reshape(3, 5, 7)
    _write_test_raster(raster_path, data)

    tiles = slice_geotiff_to_memory(raster_path, tile_size=4)

    assert len(tiles) == 4
    assert tiles[0].image_array.shape == (3, 4, 4)
    assert tiles[-1].image_array.shape == (3, 4, 4)
    assert tiles[0].valid_shape == (4, 4)
    assert tiles[-1].valid_shape == (4, 4)
    np.testing.assert_array_equal(tiles[0].image_array[:, :4, :4], data[:, :4, :4])
    np.testing.assert_array_equal(tiles[-1].image_array, data[:, 1:5, 3:7])


def test_slice_geotiff_to_memory_keeps_small_image_unpadded(tmp_path: Path) -> None:
    raster_path = tmp_path / "small.tif"
    data = np.arange(3 * 3 * 2, dtype=np.uint8).reshape(3, 3, 2)
    _write_test_raster(raster_path, data)

    tiles = slice_geotiff_to_memory(raster_path, tile_size=4)

    assert len(tiles) == 1
    assert tiles[0].image_array.shape == (3, 3, 2)
    assert tiles[0].valid_shape == (3, 2)
    np.testing.assert_array_equal(tiles[0].image_array, data)


def test_stitch_geotiff_tiles_recreates_original_region(tmp_path: Path) -> None:
    raster_path = tmp_path / "input.tif"
    data = np.arange(3 * 5 * 7, dtype=np.uint8).reshape(3, 5, 7)
    _write_test_raster(raster_path, data)

    slices_dir = tmp_path / "slices"
    slice_geotiff_to_files(raster_path, slices_dir, tile_size=4, to_jpeg=False)
    stitched_path = tmp_path / "stitched.tif"
    stitch_geotiff_tiles(slices_dir, stitched_path)

    with rasterio.open(stitched_path) as src:
        rebuilt = src.read()

    np.testing.assert_array_equal(rebuilt, data)
    assert rebuilt.shape == (3, 5, 7)


def test_stitch_geotiff_tiles_recreates_original_region_with_overlap(tmp_path: Path) -> None:
    raster_path = tmp_path / "input_overlap.tif"
    data = np.arange(3 * 5 * 7, dtype=np.uint8).reshape(3, 5, 7)
    _write_test_raster(raster_path, data)

    slices_dir = tmp_path / "overlap_slices"
    slice_geotiff_to_files(raster_path, slices_dir, tile_size=4, stride=3, to_jpeg=False)
    stitched_path = tmp_path / "stitched_overlap.tif"
    stitch_geotiff_tiles(slices_dir, stitched_path)

    with rasterio.open(stitched_path) as src:
        rebuilt = src.read()

    np.testing.assert_array_equal(rebuilt, data)
    assert rebuilt.shape == (3, 5, 7)


def test_slice_geotiff_to_files_writes_png_tiles(tmp_path: Path) -> None:
    raster_path = tmp_path / "input_png.tif"
    data = np.arange(3 * 5 * 7, dtype=np.uint8).reshape(3, 5, 7)
    _write_test_raster(raster_path, data)

    slices_dir = tmp_path / "png_slices"
    written = slice_geotiff_to_files(
        raster_path,
        slices_dir,
        tile_size=4,
        output_format="png",
    )

    assert len(written) == 4
    assert all(path.suffix.lower() == ".png" for path in written)
    first_image = Image.open(written[0])
    assert first_image.size == (4, 4)


def test_slice_geotiff_collection_to_files_handles_directory_input(tmp_path: Path) -> None:
    raster_dir = tmp_path / "geotiffs"
    raster_dir.mkdir()
    nested_dir = raster_dir / "nested"
    nested_dir.mkdir()

    _write_test_raster(raster_dir / "a.tif", np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4))
    _write_test_raster(nested_dir / "b.tif", np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4))

    output_dir = tmp_path / "tiles"
    written = slice_geotiff_collection_to_files(
        raster_dir,
        output_dir,
        tile_size=4,
        output_format="jpg",
    )

    assert len(written) == 2
    assert (output_dir / "a" / "slice_0_0.jpg").exists()
    assert (output_dir / "nested" / "b" / "slice_0_0.jpg").exists()


def test_slice_collection_reports_file_progress(tmp_path: Path) -> None:
    raster_dir = tmp_path / "geotiffs"
    raster_dir.mkdir()
    _write_test_raster(raster_dir / "a.tif", np.zeros((3, 4, 4), dtype=np.uint8))
    _write_test_raster(raster_dir / "b.tif", np.zeros((3, 4, 4), dtype=np.uint8))
    updates: list[tuple[int, int, str]] = []

    slice_geotiff_collection_to_files(
        raster_dir,
        tmp_path / "tiles",
        tile_size=4,
        output_format="png",
        workers=1,
        progress_callback=lambda completed, total, detail: updates.append((completed, total, detail)),
    )

    assert updates == [(1, 2, "a.tif"), (2, 2, "b.tif")]


def test_slice_geotiff_collection_to_files_supports_multiple_workers(tmp_path: Path) -> None:
    raster_dir = tmp_path / "geotiffs"
    raster_dir.mkdir()

    _write_test_raster(raster_dir / "a.tif", np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4))
    _write_test_raster(raster_dir / "b.tif", np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4))

    output_dir = tmp_path / "tiles"
    written = slice_geotiff_collection_to_files(
        raster_dir,
        output_dir,
        tile_size=4,
        output_format="png",
        workers=2,
    )

    assert len(written) == 2
    assert (output_dir / "a" / "slice_0_0.png").exists()
    assert (output_dir / "b" / "slice_0_0.png").exists()


def test_slice_geotiff_collection_to_files_includes_path_on_worker_failure(tmp_path: Path) -> None:
    raster_dir = tmp_path / "geotiffs"
    raster_dir.mkdir()

    _write_test_raster(raster_dir / "good.tif", np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4))
    (raster_dir / "bad.tif").write_text("not a real geotiff", encoding="utf-8")

    with pytest.raises(RuntimeError, match="bad.tif"):
        slice_geotiff_collection_to_files(
            raster_dir,
            tmp_path / "tiles",
            tile_size=4,
            output_format="png",
            workers=2,
        )
