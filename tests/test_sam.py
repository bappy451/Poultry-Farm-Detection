from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from poultry_geo.sam import lonlat_to_pixel, run_sam_geo_csv_batch


def test_lonlat_to_pixel_matches_transform() -> None:
    transform = from_origin(100.0, 200.0, 1.0, 1.0)
    assert lonlat_to_pixel(101.0, 199.0, transform) == (1, 1)


def test_run_sam_geo_csv_batch_with_fake_predictor(tmp_path: Path, monkeypatch) -> None:
    class FakePredictor:
        def set_image(self, image):
            self.last_image = image

        def predict(self, **kwargs):
            roi_h = self.last_image.shape[0]
            roi_w = self.last_image.shape[1]
            return np.array([np.ones((roi_h, roi_w), dtype=np.uint8)]), np.array([0.9]), None

    class FakeCV2:
        INTER_NEAREST = 0
        FONT_HERSHEY_SIMPLEX = 0
        COLOR_RGB2BGR = 0

        @staticmethod
        def resize(image, size, interpolation=0):
            width, height = size
            return np.ones((height, width), dtype=image.dtype)

        @staticmethod
        def addWeighted(image_a, alpha, image_b, beta, gamma):
            return image_a

        @staticmethod
        def rectangle(image, pt1, pt2, color, thickness):
            return image

        @staticmethod
        def circle(image, center, radius, color, thickness):
            return image

        @staticmethod
        def putText(image, text, origin, font, scale, color, thickness):
            return image

        @staticmethod
        def cvtColor(image, code):
            return image

        @staticmethod
        def imwrite(path, image):
            Path(path).write_bytes(b"fake-jpg")
            return True

    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    monkeypatch.setattr("poultry_geo.sam.build_sam_predictor", lambda checkpoint_path, model_type="vit_h", device=None: FakePredictor())

    geotiff_dir = tmp_path / "geotiffs"
    geotiff_dir.mkdir()
    geotiff_path = geotiff_dir / "demo.tif"
    with rasterio.open(
        geotiff_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=3,
        dtype="uint8",
        crs="EPSG:26916",
        transform=from_origin(100.0, 200.0, 1.0, 1.0),
    ) as dst:
        dst.write(np.ones((3, 4, 4), dtype=np.uint8))

    csv_path = tmp_path / "detections.csv"
    pd.DataFrame(
        [
            {
                "tile_name": "demo.tif",
                "longitude": 101.0,
                "latitude": 199.0,
                "bbox_lon_x1": 100.0,
                "bbox_lat_y1": 200.0,
                "bbox_lon_x2": 102.0,
                "bbox_lat_y2": 198.0,
                "class": 0,
                "confidence": 0.9,
            }
        ]
    ).to_csv(csv_path, index=False)

    written = run_sam_geo_csv_batch(
        csv_path=csv_path,
        geotiff_dir=geotiff_dir,
        output_dir=tmp_path / "sam_out",
        checkpoint_path="weights/sam_vit_h_4b8939.pth",
    )

    assert len(written) == 1
    assert written[0].exists()
