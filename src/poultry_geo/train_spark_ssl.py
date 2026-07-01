import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, Optional
import os

from poultry_geo.spark import SparseTransformer
from poultry_geo.spark_models import SparKNetwork
from poultry_geo.models import BandAdapter

def calculate_spark_loss(reconstructed: torch.Tensor, original: torch.Tensor, mask: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Computes the L1 (Mean Absolute Error) reconstruction loss with 
    Patch-wise Normalization, restricted exclusively to the masked pixel regions.
    
    Args:
        reconstructed: Output of the decoder [B, 3, H, W].
        original: The ground truth adapted image [B, 3, H, W].
        mask: Binary mask where 1 = masked (hidden).
        patch_size: Size of the patches used for masking (usually 16 or 32).
    """
    # Ensure mask matches spatial resolution if upsampling occurred
    if mask.shape[2:] != reconstructed.shape[2:]:
        mask = F.interpolate(mask, size=reconstructed.shape[2:], mode='nearest')

    # Unfold both images into patches: [B, C*P*P, L] where L is number of patches
    recon_patches = F.unfold(reconstructed, kernel_size=patch_size, stride=patch_size)
    orig_patches = F.unfold(original, kernel_size=patch_size, stride=patch_size)
    
    B, _, L = orig_patches.shape
    C = original.shape[1]
    P_sq = patch_size * patch_size

    # Reshape to [B, C, P*P, L] to normalize PER CHANNEL
    orig_patches = orig_patches.view(B, C, P_sq, L)
    recon_patches = recon_patches.view(B, C, P_sq, L)
    
    # Patch-wise Normalization on target patches (Per Channel)
    orig_mean = orig_patches.mean(dim=2, keepdim=True)
    orig_var = orig_patches.var(dim=2, unbiased=False, keepdim=True)
    orig_patches_norm = (orig_patches - orig_mean) / (orig_var + 1e-6).sqrt()
    
    # L1 Loss between reconstructed patches and NORMALIZED original patches
    # Average over C and P*P dimensions to get [B, L]
    loss_patches = (recon_patches - orig_patches_norm).abs().mean(dim=(1, 2))
    
    # Pool mask down to patch level: [B, L]
    mask_patches = F.avg_pool2d(mask.float(), kernel_size=patch_size, stride=patch_size).view(mask.shape[0], -1)
    
    # Masked loss only
    masked_loss = (loss_patches * mask_patches).sum() / (mask_patches.sum() + 1e-6)
    return masked_loss

def translate_state_dict(sparse_state_dict: Dict[str, torch.Tensor], backbone_type: str = 'resnet50') -> Dict[str, torch.Tensor]:
    """
    Translates weights from a sparse encoder state dict back into the naming 
    conventions used by standard dense backbones (e.g., torchvision ResNet or Ultralytics YOLO).
    """
    dense_state_dict = {}
    is_yolo = "yolo" in backbone_type.lower()
    
    yolo_mapping = {
        "stem.weight": "model.0.conv.weight",
        "stem.ln.weight": "model.0.bn.weight",
        "stem.ln.bias": "model.0.bn.bias",
        "stage1.0.weight": "model.1.conv.weight",
        "stage1.1.ln.weight": "model.1.bn.weight",
        "stage1.1.ln.bias": "model.1.bn.bias",
        "stage2.0.weight": "model.3.conv.weight",
        "stage2.1.ln.weight": "model.3.bn.weight",
        "stage2.1.ln.bias": "model.3.bn.bias",
        "stage3.0.weight": "model.5.conv.weight",
        "stage3.1.ln.weight": "model.5.bn.weight",
        "stage3.1.ln.bias": "model.5.bn.bias",
        "stage4.0.weight": "model.7.conv.weight",
        "stage4.1.ln.weight": "model.7.bn.weight",
        "stage4.1.ln.bias": "model.7.bn.bias",
    }
    
    for key, value in sparse_state_dict.items():
        new_key = key
        
        if new_key.startswith('encoder.'):
            new_key = new_key.replace('encoder.', '', 1)
            
        if is_yolo:
            if new_key in yolo_mapping:
                dense_state_dict[yolo_mapping[new_key]] = value
            continue
            
        # ResNet naming: 'stageX' -> 'layerX'
        if 'stage' in new_key:
            new_key = new_key.replace('stage', 'layer')
            
        # Stem translation: handle SparseConv2d/LayerNorm vs standard Conv2d/BatchNorm
        if new_key.startswith('stem.'):
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
                 batch_size: int = 32,
                 patch_size: int = 16):
        self.mask_ratio = mask_ratio
        self.sparse_engine = sparse_engine
        self.backbone_type = backbone_type
        self.precision = precision
        self.lr = lr
        self.batch_size = batch_size
        self.patch_size = patch_size

def train_one_step(model, optimizer, scaler, transformer, batch, config):
    """
    Executes an optimized optimization step for the SparK objective.
    """
    optimizer.zero_grad(set_to_none=True)
    
    dtype = torch.float32
    if config.precision == 'FP16': dtype = torch.float16
    elif config.precision == 'BF16': dtype = torch.bfloat16
    
    # Use AMP for speedup
    with torch.amp.autocast(device_type='cuda', enabled=(config.precision != 'FP32'), dtype=dtype):
        # 1. Combined forward: sparse conversion and mask generation
        sparse_tensor, mask = transformer(batch)
        
        # 2. Reconstruct hidden pixels
        reconstructed = model(sparse_tensor, mask)
        
        # 3. Optimized target generation: 
        # SparseTransformer already has access to the adapted input.
        # We can reuse the transformer's adapter if we modify the return signature,
        # but for now, we just ensure it's in the same autocast context.
        with torch.no_grad():
            target = transformer.adapter(batch)
            
        # 4. Masked loss
        loss = calculate_spark_loss(reconstructed, target, mask, patch_size=config.patch_size)
        
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
