from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import fiona
from rasterio.transform import from_origin

from poultry_geo.gis import (
    create_boundary_shapefiles_from_geotiffs,
    process_shapefile_with_bbox_dimensions,
    save_bbox_shapefile_from_csv,
)


def test_create_boundary_shapefiles_from_geotiffs(tmp_path: Path) -> None:
    raster_path = tmp_path / "input.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=4,
        width=5,
        count=1,
        dtype=np.uint8,
        crs="EPSG:26916",
        transform=from_origin(100.0, 200.0, 1.0, 1.0),
    ) as dst:
        dst.write(np.ones((1, 4, 5), dtype=np.uint8))

    output_dir = tmp_path / "shapes"
    written = create_boundary_shapefiles_from_geotiffs(tmp_path, output_dir)

    assert len(written) == 1
    assert written[0].exists()


def test_save_bbox_shapefile_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "detections.csv"
    pd.DataFrame(
        [
            {
                "bbox_lon_x1": 100.0,
                "bbox_lat_y1": 190.0,
                "bbox_lon_x2": 110.0,
                "bbox_lat_y2": 200.0,
                "latitude": 195.0,
                "longitude": 105.0,
                "class": 0,
                "confidence": 0.9,
                "tile_name": "tile_a.tif",
            },
        ]
    ).to_csv(csv_path, index=False)

    output = save_bbox_shapefile_from_csv(csv_path, tmp_path / "bbox_out")

    assert output.exists()
    with fiona.open(output, "r") as src:
        assert len(src) == 1


def test_process_shapefile_with_bbox_dimensions_writes_output(tmp_path: Path) -> None:
    raster_path = tmp_path / "poly.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=4,
        width=5,
        count=1,
        dtype=np.uint8,
        crs="EPSG:26916",
        transform=from_origin(100.0, 200.0, 1.0, 1.0),
    ) as dst:
        dst.write(np.ones((1, 4, 5), dtype=np.uint8))

    boundaries = create_boundary_shapefiles_from_geotiffs(tmp_path, tmp_path / "boundaries")
    output = process_shapefile_with_bbox_dimensions(
        boundaries[0],
        tmp_path / "processed" / "dims.shp",
    )

    assert output.exists()
    with fiona.open(output, "r") as src:
        row = next(iter(src))
        assert "area_gis" in row["properties"]
