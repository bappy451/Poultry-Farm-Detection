from types import SimpleNamespace

import numpy as np
import torch
from affine import Affine

from poultry_geo.inference import yolo_result_to_records


class FakeBoxes:
    def __init__(self):
        self.data = torch.ones(1, 6)
        self.xyxy = torch.tensor([[1.0, 2.0, 5.0, 6.0]])
        self.cls = torch.tensor([3.0])
        self.conf = torch.tensor([0.9])


def test_detection_boxes_are_exported_without_masks():
    result = SimpleNamespace(boxes=FakeBoxes(), masks=None, obb=None)
    records = yolo_result_to_records(result, Affine.identity(), "tile.tif", (10, 20))
    assert len(records) == 1
    assert records[0]["task"] == "detect"
    assert records[0]["global_x"] == 13.0


def test_segmentation_polygon_is_exported():
    result = SimpleNamespace(
        boxes=FakeBoxes(),
        masks=SimpleNamespace(xy=[np.array([[1, 2], [5, 2], [5, 6], [1, 6]], dtype=np.float32)]),
        obb=None,
    )
    assert yolo_result_to_records(result, Affine.identity(), "tile.tif", (0, 0))[0]["task"] == "segment"
