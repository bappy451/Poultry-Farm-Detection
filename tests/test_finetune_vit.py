import os
import torch
import yaml
import numpy as np
import pytest
from pathlib import Path

from poultry_geo.ssl_models import RemoteSensingSegmentationModel
from downstream_finetune.finetune_vit import load_mim_backbone_into_vit

@pytest.fixture
def mock_mim_weights(tmp_path):
    # Create a mock RemoteSensingMIMModel state dict
    model = torch.nn.Linear(10, 10) # dummy
    # Just need something that looks like the real weights
    # The real weights have a 'backbone.patch_embed...'
    state_dict = {
        'backbone.pos_embed': torch.randn(1, 16, 384),
        'backbone.patch_embed.weight': torch.randn(384, 3, 16, 16),
        'backbone.patch_embed.bias': torch.randn(384),
        'reconstruction_head.weight': torch.randn(768, 384) # shouldn't be loaded
    }
    
    weights_path = tmp_path / "mock_mim.pt"
    torch.save({'model_state_dict': state_dict}, weights_path)
    return weights_path

def test_load_mim_backbone(mock_mim_weights):
    # Initialize a segmentation model
    seg_model = RemoteSensingSegmentationModel(
        in_channels=3,
        image_size=64,
        patch_size=16,
        embed_dim=384,
        depth=2,
        num_heads=6,
        precision="fp32"
    )
    
    # Store old weight to verify it changes
    old_pos_embed = seg_model.backbone.pos_embed.clone()
    
    # Load weights
    load_mim_backbone_into_vit(seg_model, mock_mim_weights)
    
    # Verify weights changed
    assert not torch.allclose(seg_model.backbone.pos_embed, old_pos_embed)
    
def test_vit_segmentation_forward():
    seg_model = RemoteSensingSegmentationModel(
        in_channels=3,
        image_size=64,
        patch_size=16,
        embed_dim=384,
        depth=2,
        num_heads=6,
        decoder_hidden_channels=64,
        precision="fp32"
    )
    
    dummy_input = torch.randn(2, 3, 64, 64)
    output = seg_model(dummy_input)
    
    # Output should have shape [batch, 1, H, W] for 1-class segmentation (foreground)
    # Wait, YoloFormatSegmentationDataset returns masks with class 1 for foreground, 0 for background.
    # We might need num_classes channels in the head, or 1 channel with BCEWithLogitsLoss.
    assert output.shape == (2, 1, 64, 64)
