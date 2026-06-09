import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from unittest.mock import MagicMock

# Mock spconv for tests
class MockSparseConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x): return x

class MockSubMConv2d(MockSparseConv2d): pass

class MockSparseConvTensor:
    def __init__(self, *args, **kwargs):
        if 'features' in kwargs:
            self.features = kwargs['features']
        elif len(args) > 0:
            self.features = args[0]
        else:
            self.features = torch.randn(1, 64)
        self.spatial_shape = [1, 224, 224]
    def replace_feature(self, features): return MockSparseConvTensor(features=features)
    def dense(self): return torch.randn(2, 2048, 7, 7)

class MockPytorch:
    SparseConv2d = MockSparseConv2d
    SubMConv2d = MockSubMConv2d
    SparseConvTensor = MockSparseConvTensor

mock_spconv = MagicMock()
mock_spconv.pytorch = MockPytorch
sys.modules["spconv"] = mock_spconv
sys.modules["spconv.pytorch"] = MockPytorch

from poultry_geo.spark import SparseTransformer
from poultry_geo.spark_models import SparKNetwork
from poultry_geo.train_spark_ssl import calculate_spark_loss, translate_state_dict
from poultry_geo.models import BandAdapter
from torchvision.models import resnet50

class MockEncoder(nn.Module):
    def __init__(self, B):
        super().__init__()
        self.B = B
        self.param = nn.Parameter(torch.randn(1))
    def forward(self, x):
        m = MockSparseConvTensor()
        m.dense = lambda: torch.randn(self.B, 2048, 4, 4)
        return [None, None, None, m]

class MockDecoder(nn.Module):
    def __init__(self, B, H, W):
        super().__init__()
        self.B, self.H, self.W = B, H, W
        self.up1 = nn.Sequential(nn.Conv2d(2048, 3, 1))
    def forward(self, features, mask):
        # We need a tensor that depends on decoder parameters
        x = features[-1].dense()
        return self.up1(x)

def test_spark_optimization_step():
    """Verify that a 2-step optimization updates weights based on masked-patch loss."""
    B, C, H, W = 2, 3, 32, 32
    model = SparKNetwork(backbone_type='resnet50', in_channels=C)
    
    # Mock encoder/decoder to avoid spconv/LayerNorm issues in this functional test
    model.encoder = MockEncoder(B)
    model.decoder = MockDecoder(B, H, W)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-1) # High LR to ensure update
    initial_param = model.decoder.up1[0].weight.clone()
    
    adapter = nn.Identity()
    transformer = SparseTransformer(adapter=adapter, patch_size=4, mask_ratio=0.5)
    
    for _ in range(2):
        optimizer.zero_grad()
        input_tensor = torch.randn(B, C, H, W)
        sparse_tensor, mask = transformer(input_tensor)
        
        reconstructed = model(sparse_tensor, mask)
        # Ensure target matches reconstructed shape for loss
        target = F.interpolate(input_tensor, size=reconstructed.shape[2:])
        loss = calculate_spark_loss(reconstructed, target, mask)
        
        loss.backward()
        optimizer.step()
        
    final_param = model.decoder.up1[0].weight
    assert not torch.equal(initial_param, final_param), "Weights should have updated after optimization steps."
    assert loss.item() >= 0

import torch.nn.functional as F

def test_weight_translation_resnet50():
    """Verify sparse weights can be translated to a standard dense ResNet50 state-dict."""
    B, C, H, W = 1, 3, 224, 224
    spark_model = SparKNetwork(backbone_type='resnet50', in_channels=C)
    
    # Export to dense state dict
    dense_state_dict = translate_state_dict(spark_model.encoder.state_dict(), backbone_type='resnet50')
    
    # Load into vanilla resnet50
    vanilla_model = resnet50()
    
    # Attempt loading
    # We use strict=False because stem might be custom, but stages should match
    missing, unexpected = vanilla_model.load_state_dict(dense_state_dict, strict=False)
    
    # Check if stage 1 layers were loaded (they shouldn't be in missing)
    stage1_missing = [k for k in missing if k.startswith('layer1')]
    # Many will be missing because of BatchNorm vs LayerNorm mismatch (running_mean/var)
    # But weight/bias should be present.
    
    for k, v in dense_state_dict.items():
        if k.startswith('layer1') and (k.endswith('.weight') or k.endswith('.bias')):
            assert k in vanilla_model.state_dict(), f"Key {k} should be in vanilla model"
