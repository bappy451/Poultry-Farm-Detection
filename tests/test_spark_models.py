import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import sys

# Mock spconv before importing the module that uses it
class MockSparseConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class MockSubMConv2d(MockSparseConv2d):
    pass

class MockSparseConvTensor:
    def __init__(self, *args, **kwargs):
        # features: [N, C]
        if 'features' in kwargs:
            self.features = kwargs['features']
        elif len(args) > 0:
            self.features = args[0]
        else:
            self.features = torch.randn(1, 64) 
        self.spatial_shape = [1, 224, 224] # [D, H, W]
    def replace_feature(self, features):
        return MockSparseConvTensor(features=features)
    def dense(self):
        return torch.randn(2, 2048, 7, 7) 

# Create a clean mock structure
class MockPytorch:
    SparseConv2d = MockSparseConv2d
    SubMConv2d = MockSubMConv2d
    SparseConvTensor = MockSparseConvTensor

mock_spconv = MagicMock()
mock_spconv.pytorch = MockPytorch

sys.modules["spconv"] = mock_spconv
sys.modules["spconv.pytorch"] = MockPytorch
import spconv.pytorch as spconv

import poultry_geo.spark_models as spark_models
from poultry_geo.spark_models import SparKNetwork
from poultry_geo.spark import SparseTransformer
from poultry_geo.models import BandAdapter

def test_spark_network_resnet_initialization():
    """Verify SparKNetwork can be initialized with a ResNet50 backbone."""
    model = SparKNetwork(backbone_type="resnet50", in_channels=10)
    assert model.encoder is not None
    assert model.decoder is not None
    assert isinstance(model.encoder, nn.Module)

def test_spark_network_yolo_initialization():
    """Verify SparKNetwork can be initialized with a YOLO/CSPDarkNet backbone."""
    model = SparKNetwork(backbone_type="cspdarknet", in_channels=10)
    assert model.encoder is not None
    assert model.decoder is not None

class MockEncoder(nn.Module):
    def __init__(self, B):
        super().__init__()
        self.B = B
    def forward(self, *args, **kwargs):
        m = MagicMock()
        m.dense = lambda: torch.randn(self.B, 2048, 7, 7)
        m.spatial_shape = [1, 224, 224]
        return [torch.randn(self.B, 256, 56, 56), 
                torch.randn(self.B, 512, 28, 28),
                torch.randn(self.B, 1024, 14, 14),
                m]

class MockDecoder(nn.Module):
    def __init__(self, B, H, W):
        super().__init__()
        self.B, self.H, self.W = B, H, W
    def forward(self, *args, **kwargs):
        return torch.randn(self.B, 3, self.H, self.W)

def test_spark_network_forward_pass():
    """Verify forward pass processes sparse tensor and returns dense reconstructed image."""
    B, C, H, W = 2, 10, 224, 224
    model = SparKNetwork(backbone_type="resnet50", in_channels=C)
    
    # Use nn.Module subclasses for mocking
    model.encoder = MockEncoder(B)
    model.decoder = MockDecoder(B, H, W)
    
    # Mocking input pipeline: SparseTransformer -> SparKNetwork
    adapter = BandAdapter(in_channels=C, out_channels=3)
    transformer = SparseTransformer(adapter=adapter, patch_size=16, mask_ratio=0.75)
    
    input_tensor = torch.randn(B, C, H, W)
    sparse_tensor, mask = transformer(input_tensor)
    
    # Forward pass through SparKNetwork
    reconstructed = model(sparse_tensor, mask)
    
    # Verify output resolution: (B, 3, H, W)
    assert reconstructed.shape == (B, 3, H, W)

def test_sparse_layer_replacement(monkeypatch):
    """Ensure that the encoder uses sparse convolution layers instead of dense ones."""
    monkeypatch.setattr(spark_models, "spconv", MockPytorch)
    model = SparKNetwork(backbone_type="resnet50", in_channels=3)
    
    # Check if some layers in the encoder are spconv types
    found_sparse = False
    for name, module in model.encoder.named_modules():
        if isinstance(module, (MockSparseConv2d, MockSubMConv2d)):
            found_sparse = True
            break
    
    assert found_sparse
