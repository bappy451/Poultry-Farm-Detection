from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from poultry_geo.conversion import convert_geotiff_to_png_or_jpg


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


def test_convert_geotiff_to_png_or_jpg_handles_single_band_and_nested_dirs(tmp_path: Path) -> None:
    input_dir = tmp_path / "geotiffs"
    nested_dir = input_dir / "nested"
    nested_dir.mkdir(parents=True)

    single_band = np.arange(1 * 5 * 7, dtype=np.uint8).reshape(1, 5, 7)
    rgb = np.arange(3 * 5 * 7, dtype=np.uint8).reshape(3, 5, 7)
    _write_test_raster(input_dir / "one_band.tif", single_band)
    _write_test_raster(nested_dir / "rgb.tif", rgb)

    output_dir = tmp_path / "converted_png"
    written = convert_geotiff_to_png_or_jpg(input_dir, output_dir, out_ext="png")

    assert len(written) == 2
    assert (output_dir / "one_band.png").exists()
    assert (output_dir / "nested" / "rgb.png").exists()

    image = Image.open(output_dir / "one_band.png")
    assert image.mode == "RGB"
    assert image.size == (7, 5)


def test_convert_geotiff_to_png_or_jpg_writes_jpeg_for_single_file_input(tmp_path: Path) -> None:
    raster_path = tmp_path / "input.tif"
    rgb = np.arange(3 * 4 * 6, dtype=np.uint8).reshape(3, 4, 6)
    _write_test_raster(raster_path, rgb)

    output_dir = tmp_path / "jpg_output"
    written = convert_geotiff_to_png_or_jpg(
        raster_path,
        output_dir,
        out_ext="jpg",
        jpeg_quality=90,
    )

    assert len(written) == 1
    assert written[0].suffix.lower() == ".jpg"
    image = Image.open(written[0])
    assert image.size == (6, 4)
