from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.transform import from_origin

from poultry_geo.inference import _iter_same_shape_batches, yolo_result_to_records


def test_yolo_result_to_records_emits_geospatial_fields() -> None:
    result = SimpleNamespace(
        masks=SimpleNamespace(
            xy=[
                np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=np.float32)
            ]
        ),
        boxes=SimpleNamespace(
            xyxy=[np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32)],
            cls=[np.float32(1)],
            conf=[np.float32(0.95)],
            data=np.ones((1, 6), dtype=np.float32),
        ),
    )

    records = yolo_result_to_records(
        result=result,
        transform=from_origin(100.0, 200.0, 1.0, 1.0),
        tile_name="demo.tif",
        offset_xy=(10, 20),
    )

    assert len(records) == 1
    record = records[0]
    assert record["tile_name"] == "demo.tif"
    assert record["class"] == 1
    assert record["confidence"] == pytest.approx(0.95)
    assert record["global_x"] == 11.0
    assert record["global_y"] == 21.0
    assert json.loads(record["mask_points_gis"])[0] == [100.0, 200.0]


def test_iter_same_shape_batches_keeps_mixed_shapes_separate() -> None:
    tiles = [
        SimpleNamespace(image_array=np.zeros((3, 4, 4), dtype=np.uint8)),
        SimpleNamespace(image_array=np.zeros((3, 4, 4), dtype=np.uint8)),
        SimpleNamespace(image_array=np.zeros((3, 3, 4), dtype=np.uint8)),
        SimpleNamespace(image_array=np.zeros((3, 3, 4), dtype=np.uint8)),
    ]

    batches = list(_iter_same_shape_batches(tiles, batch_size=8))

    assert len(batches) == 2
    assert all(tile.image_array.shape[1:] == (4, 4) for tile in batches[0])
    assert all(tile.image_array.shape[1:] == (3, 4) for tile in batches[1])
