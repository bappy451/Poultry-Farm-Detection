import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, Optional
import os

from poultry_geo.spark import SparseTransformer
from poultry_geo.spark_models import SparKNetwork
from poultry_geo.models import BandAdapter

def calculate_spark_loss(reconstructed: torch.Tensor, original: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the Mean Squared Error (MSE) reconstruction loss, restricted 
    exclusively to the masked pixel regions.
    
    In SparK, we only penalize the model for incorrectly predicting the pixels 
    it couldn't see during the forward pass.
    
    Args:
        reconstructed: Output of the decoder [B, 3, H, W].
        original: The ground truth adapted image [B, 3, H, W].
        mask: Binary mask where 1 = masked (hidden).
    """
    # Compute element-wise squared error
    loss = F.mse_loss(reconstructed, original, reduction='none')
    
    # Ensure mask matches spatial resolution if upsampling occurred
    if mask.shape[2:] != reconstructed.shape[2:]:
        mask = F.interpolate(mask, size=reconstructed.shape[2:], mode='nearest')

    # Masked loss only: Average error across all hidden pixels
    masked_loss = (loss * mask).sum() / (mask.sum() + 1e-6)
    
    return masked_loss

def translate_state_dict(sparse_state_dict: Dict[str, torch.Tensor], backbone_type: str = 'resnet50') -> Dict[str, torch.Tensor]:
    """
    Translates weights from a sparse encoder state dict back into the naming 
    conventions used by standard dense backbones (e.g., torchvision ResNet).
    
    This allows us to pretrain sparsely and then load the resulting weights 
    into a standard YOLO or Segmentation model without code modifications.
    """
    dense_state_dict = {}
    for key, value in sparse_state_dict.items():
        new_key = key
        
        # Strip the 'encoder.' prefix added by the SparK wrapper
        if new_key.startswith('encoder.'):
            new_key = new_key.replace('encoder.', '', 1)
            
        # ResNet naming: 'stageX' -> 'layerX'
        if 'stage' in new_key:
            new_key = new_key.replace('stage', 'layer')
            
        # Stem translation: handle SparseConv2d/LayerNorm vs standard Conv2d/BatchNorm
        if new_key.startswith('stem.'):
            # Map our sequential stem indices back to torchvision resnet50 names
            if '0.weight' in new_key: new_key = 'conv1.weight'
            elif '1.ln.weight' in new_key: new_key = 'bn1.weight'
            elif '1.ln.bias' in new_key: new_key = 'bn1.bias'

        # Flatten SparseLayerNorm wrappers back to the base param names
        if '.ln.' in new_key:
            new_key = new_key.replace('.ln.', '.')
            
        dense_state_dict[new_key] = value
        
    return dense_state_dict

class SparKConfig:
    """
    Configuration container for SparK SSL training sessions.
    """
    def __init__(self, 
                 mask_ratio: float = 0.65, 
                 sparse_engine: str = 'spconv', 
                 backbone_type: str = 'resnet50', 
                 precision: str = 'FP32',
                 lr: float = 1e-3,
                 batch_size: int = 32):
        self.mask_ratio = mask_ratio
        self.sparse_engine = sparse_engine
        self.backbone_type = backbone_type
        self.precision = precision
        self.lr = lr
        self.batch_size = batch_size

def train_one_step(model, optimizer, scaler, transformer, batch, config):
    """
    Executes a single optimization step for the SparK objective.
    
    Workflow:
    1. Dense Image -> Sparse Transformer (generates mask and sparse visible features).
    2. Sparse Encoder -> Dense Decoder (reconstructs hidden pixels).
    3. Calculate MSE loss on masked regions.
    4. Backpropagate and update.
    """
    optimizer.zero_grad()
    
    # Map precision string to torch dtypes
    dtype = torch.float32
    if config.precision == 'FP16': dtype = torch.float16
    elif config.precision == 'BF16': dtype = torch.bfloat16
    
    # Use Automatic Mixed Precision (AMP) to speed up training on supported GPUs
    with torch.cuda.amp.autocast(enabled=(config.precision != 'FP32'), dtype=dtype):
        # 1. Transform dense input into a SparseConvTensor (Visible pixels only)
        sparse_tensor, mask = transformer(batch)
        
        # 2. Forward pass through hierarchical encoder and decoder
        reconstructed = model(sparse_tensor, mask)
        
        # 3. Generate the target image (the original adapted 3-band image)
        with torch.no_grad():
            target = transformer.adapter(batch)
            
        # 4. Compute reconstruction loss on masked patches
        loss = calculate_spark_loss(reconstructed, target, mask)
        
    # Standard backprop vs Scaled backprop (for AMP)
    if config.precision == 'FP32':
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    return loss.item()

def save_spark_checkpoint(model, config, path, epoch):
    """
    Saves both the full training checkpoint and a 'translated' dense backbone.
    
    The translated backbone can be used immediately in standard Ultralytics 
    or Timm models as a pretrained weights file.
    """
    # 1. Save the full SparK network (for resuming pretraining)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': vars(config)
    }, os.path.join(path, f'spark_unified_ep{epoch}.pt'))
    
    # 2. Save the extracted dense backbone weights (for downstream tasks)
    dense_backbone_state = translate_state_dict(model.encoder.state_dict(), config.backbone_type)
    torch.save(dense_backbone_state, os.path.join(path, f'backbone_dense_ep{epoch}.pt'))
