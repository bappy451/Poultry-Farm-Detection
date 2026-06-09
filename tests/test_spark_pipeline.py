import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import sys
import importlib

# Mock spconv before importing the module that uses it
mock_spconv = MagicMock()
sys.modules["spconv"] = mock_spconv
mock_pytorch = MagicMock()
mock_spconv.pytorch = mock_pytorch
sys.modules["spconv.pytorch"] = mock_pytorch

import poultry_geo.spark
# Force reload to ensure it picks up the mock if it was already imported elsewhere
importlib.reload(poultry_geo.spark)

from poultry_geo.spark import PatchMasker, SparseTransformer
from poultry_geo.models import BandAdapter

def test_patch_masker_ratio():
    """Verify the patch masking generator flags the correct percentage of the image."""
    B, H, W = 4, 224, 224
    patch_size = 16
    mask_ratio = 0.75
    masker = PatchMasker(patch_size=patch_size, mask_ratio=mask_ratio)
    
    mask = masker.generate_mask(B, H, W)
    
    assert mask.shape == (B, 1, H, W)
    # Check if ratio is approximately correct
    actual_ratio = mask.float().mean().item()
    assert abs(actual_ratio - mask_ratio) < 0.05

def test_sparse_transformer_output():
    """Verify SparseTransformer returns a valid Sparse Tensor with correct features."""
    B, C, H, W = 2, 10, 32, 32
    patch_size = 4
    mask_ratio = 0.5 # 50% masked, 50% active
    
    adapter = BandAdapter(in_channels=C, out_channels=3)
    transformer = SparseTransformer(adapter=adapter, patch_size=patch_size, mask_ratio=mask_ratio)
    
    input_tensor = torch.randn(B, C, H, W)
    sparse_tensor, mask = transformer(input_tensor)
    
    # Check mask shape
    assert mask.shape == (B, 1, H, W)
    
    # Verify spconv.SparseConvTensor was instantiated
    assert mock_pytorch.SparseConvTensor.called
    
    # Verify the number of active features
    active_mask_indices = (mask == 0).sum().item()
    
    # Get the arguments passed to SparseConvTensor
    args, kwargs = mock_pytorch.SparseConvTensor.call_args
    features = kwargs.get('features') if 'features' in kwargs else args[0]
    indices = kwargs.get('indices') if 'indices' in kwargs else args[1]
    
    assert features.shape[0] == active_mask_indices
    assert features.shape[1] == 3 # output of BandAdapter
    assert indices.shape[0] == active_mask_indices
    assert indices.shape[1] == 4

def test_no_dense_ops_on_masked_pixels():
    """Verify that features are only extracted from unmasked coordinates."""
    B, C, H, W = 1, 3, 4, 4
    patch_size = 2
    mask_ratio = 0.5 
    
    adapter = nn.Identity() 
    transformer = SparseTransformer(adapter=adapter, patch_size=patch_size, mask_ratio=mask_ratio)
    
    input_tensor = torch.arange(B * C * H * W).reshape(B, C, H, W).float()
    
    sparse_tensor, mask = transformer(input_tensor)
    
    assert mock_pytorch.SparseConvTensor.called
    args, kwargs = mock_pytorch.SparseConvTensor.call_args
    features = kwargs.get('features') if 'features' in kwargs else args[0]
    indices = kwargs.get('indices') if 'indices' in kwargs else args[1]
    
    for i in range(features.shape[0]):
        b, z, y, x = indices[i].tolist()
        expected_val = input_tensor[b, :, y, x]
        assert torch.allclose(features[i], expected_val)
        assert mask[b, 0, y, x] == 0
