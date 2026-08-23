import pytest
import torch
import torch.nn as nn

import poultry_geo.spark as spark
from poultry_geo.spark import SparseTransformer, require_spconv
from poultry_geo.spark_models import SparKNetwork


def test_sparse_runtime_is_required(monkeypatch):
    monkeypatch.setattr(spark, "spconv", None)
    with pytest.raises(RuntimeError, match="requires spconv"):
        require_spconv()


def test_transformer_validates_mask_geometry_before_sparse_execution(monkeypatch):
    monkeypatch.setattr(spark, "spconv", object())
    transformer = SparseTransformer(nn.Identity(), patch_size=16, mask_ratio=0.5)
    with pytest.raises(ValueError, match="divisible"):
        transformer.generate_mask(1, 30, 32, torch.device("cpu"))


def test_spark_network_rejects_unverified_yolo_backbone(monkeypatch):
    import poultry_geo.spark_models as spark_models
    monkeypatch.setattr(spark_models, "spconv", object())
    monkeypatch.setattr(spark, "spconv", object())
    with pytest.raises(ValueError, match="Only 'resnet50'"):
        SparKNetwork(backbone_type="yolov26l")
