import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, Optional
import os

from poultry_geo.spark import SparseTransformer
from poultry_geo.spark_models import SparKNetwork
from poultry_geo.models import BandAdapter
from poultry_geo.checkpoint_manager import StateManager


def calculate_spark_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int = 16,
) -> torch.Tensor:
    """
    Computes the Mean Squared Error (MSE) reconstruction loss with
    Patch-wise Normalization, restricted exclusively to the masked pixel regions.

    Args:
        reconstructed: Output of the decoder [B, 3, H, W].
        original: The ground truth adapted image [B, 3, H, W].
        mask: Binary mask where 1 = masked (hidden), [B, 1, H, W].
        patch_size: Size of the patches used for masking (usually 16 or 32).
    """
    if reconstructed.shape != original.shape:
        raise ValueError(f"Reconstruction shape {tuple(reconstructed.shape)} does not match target {tuple(original.shape)}.")
    if reconstructed.shape[-2] % patch_size or reconstructed.shape[-1] % patch_size:
        raise ValueError("Reconstruction dimensions must be divisible by patch_size.")
    if mask.shape[2:] != reconstructed.shape[2:]:
        raise ValueError("Mask and reconstruction must have the same spatial dimensions.")

    # Unfold both images into patches: [B, C*P*P, L] where L is number of patches.
    # MUST cast to float32 â€” F.unfold does not support bfloat16 (raises RuntimeError).
    recon_patches = F.unfold(reconstructed.float(), kernel_size=patch_size, stride=patch_size)
    orig_patches  = F.unfold(original.float(),      kernel_size=patch_size, stride=patch_size)

    B, _, L = orig_patches.shape
    C = original.shape[1]
    P_sq = patch_size * patch_size

    # Reshape to [B, C, P*P, L] to normalize PER CHANNEL
    orig_patches  = orig_patches.view(B, C, P_sq, L)
    recon_patches = recon_patches.view(B, C, P_sq, L)

    # Patch-wise Normalization on target patches (Per Channel)
    orig_mean = orig_patches.mean(dim=2, keepdim=True)
    orig_var  = orig_patches.var(dim=2, unbiased=False, keepdim=True)
    orig_patches_norm = (orig_patches - orig_mean) / (orig_var + 1e-6).sqrt()

    # MSE (L2) Loss between reconstructed patches and NORMALIZED original patches
    # Average over C and P*P dimensions to get [B, L]
    loss_patches = (recon_patches - orig_patches_norm).pow(2).mean(dim=(1, 2))

    # Pool mask down to patch level: [B, L]
    mask_patches = F.avg_pool2d(mask.float(), kernel_size=patch_size, stride=patch_size).view(mask.shape[0], -1)

    # Masked loss only
    masked_loss = (loss_patches * mask_patches).sum() / (mask_patches.sum() + 1e-6)
    return masked_loss


def export_sparse_encoder_state(model: nn.Module, config, path: str):
    """Save the custom sparse encoder translated back into standard dense weights."""
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise ValueError("SparK model has no encoder to export.")
    
    dense_state_dict = translate_state_dict(encoder.state_dict(), config.backbone_type)
    
    export_path = os.path.join(path, f"{config.backbone_type}_spark_latest.pt")
    torch.save({
        "format_version": 2, # Version 2 expected by downstream loader
        "architecture": config.backbone_type,
        "backbone_type": config.backbone_type,
        "state_dict": dense_state_dict,
        "export_note": "SparK pre-trained weights successfully converted back to dense PyTorch layout for YOLO/ResNet transfer."
    }, export_path)
    return export_path

def translate_state_dict(sparse_state_dict: Dict[str, torch.Tensor], backbone_type: str = "resnet50") -> Dict[str, torch.Tensor]:
    """Convert sparse spconv KRSC weights back to dense OIHW weights for downstream use."""
    from poultry_geo.spark_models import spconv_weight_to_dense_conv
    dense_dict = {}
    for key, value in sparse_state_dict.items():
        if "weight" in key and value.ndim == 4:
            dense_dict[key] = spconv_weight_to_dense_conv(value)
        else:
            dense_dict[key] = value
            
    if backbone_type.startswith("yolo"):
        # Strip 'backbone.' prefix for YOLO so it matches Ultralytics layers (e.g. '0.conv.weight')
        yolo_dict = {}
        for k, v in dense_dict.items():
            if k.startswith("backbone."):
                yolo_dict[k.replace("backbone.", "", 1)] = v
        return yolo_dict
        
    return dense_dict

class SparKConfig:
    """
    Configuration container for SparK SSL training sessions.
    """
    def __init__(
        self,
        mask_ratio: float = 0.65,
        sparse_engine: str = 'spconv',
        backbone_type: str = 'resnet50',
        precision: str = 'FP32',
        lr: float = 1e-3,
        batch_size: int = 32,
        patch_size: int = 16,
        device_type: str = 'cuda',       # FIX Bug 7: dynamic device type for autocast
        weight_decay: float = 0.05,      # FIX Bug 4: configurable weight decay
    ):
        self.mask_ratio = mask_ratio
        self.sparse_engine = sparse_engine
        self.backbone_type = backbone_type
        self.precision = precision
        self.lr = lr
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.device_type = device_type
        self.weight_decay = weight_decay


def train_one_step(model, optimizer, scaler, transformer, batch, config):
    """
    Executes one optimized optimization step for the SparK objective.

    Fixes applied:
    - Bug 7: autocast uses config.device_type instead of hardcoded 'cuda'.
    - DF4: gradient clipping (max_norm=1.0) prevents BF16 gradient spikes.
    """
    optimizer.zero_grad(set_to_none=True)

    dtype = torch.float32
    if config.precision == 'FP16':
        dtype = torch.float16
    elif config.precision == 'BF16':
        dtype = torch.bfloat16

    # AMP is only supported on CUDA and (for bf16) CPU in newer PyTorch.
    amp_enabled = (config.precision != 'FP32') and (config.device_type in ('cuda', 'cpu'))

    with torch.amp.autocast(device_type=config.device_type, enabled=amp_enabled, dtype=dtype):
        # 1. Sparse conversion + mask generation
        sparse_tensor, mask = transformer(batch)

        # 2. Reconstruct hidden pixels
        reconstructed = model(sparse_tensor, mask)

        # 3. Compute target as the adapter's output â€” detached so it acts as a
        #    fixed target for THIS step while still allowing the adapter to update.
        target = transformer.adapter(batch).detach()

        # 4. Masked reconstruction loss
        loss = calculate_spark_loss(reconstructed, target, mask, patch_size=config.patch_size)

    if config.precision == 'FP32' or not amp_enabled:
        loss.backward()
        # DF4 FIX: gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(transformer.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        # DF4 FIX: unscale before clipping so norms are in the original scale
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(transformer.parameters()),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()

    return loss.item()


def save_spark_checkpoint(
    model, optimizer, scaler, config, path, epoch,
    transformer: nn.Module | None = None,
    schedule_total_epochs: int = None,
    schedule_warmup_epochs: int = None,
):
    """
    Saves 'spark_unified_latest.pt' (always) and 'spark_unified_best.pt' (externally
    managed by the training loop).

    DF2 FIX: No per-epoch checkpoint accumulation â€” only latest + best are kept,
    avoiding up to 30 GB of disk usage over 600 epochs.

    LR Schedule Fix: schedule_total_epochs and schedule_warmup_epochs are saved so
    that on resume the LR curve is continuous even if --epochs is changed between runs.
    """
    latest_path = os.path.join(path, 'spark_unified_latest.pt')

    checkpoint_state = {
        'epoch': epoch,
        'format_version': 2,
        'backbone_type': config.backbone_type,
        'model_state_dict': model.state_dict(),
        'transformer_state_dict': transformer.state_dict() if transformer is not None else None,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'rng_state': StateManager._capture_rng_state(),
        'config': vars(config),
        # LR schedule anchors â€” restored on resume to prevent LR spikes.
        'schedule_total_epochs': schedule_total_epochs,
        'schedule_warmup_epochs': schedule_warmup_epochs,
    }
    torch.save(checkpoint_state, latest_path)

    # The sparse encoder has a custom stem and spconv KRSC weights. Export it only
    # in its native format; a dense/YOLO conversion would be misleading and unsafe.
    export_sparse_encoder_state(model, config, path)


def save_spark_visualization(original: torch.Tensor, mask: torch.Tensor, reconstructed: torch.Tensor, output_path: str):
    """
    Saves a debugging visualization grid of Original vs Masked vs Reconstructed.
    """
    import torchvision.utils as vutils

    def norm_tensor(t):
        t = t.clone().float()
        for b in range(t.size(0)):
            t[b] -= t[b].min()
            t[b] /= (t[b].max() + 1e-5)
        return t

    orig_norm  = norm_tensor(original)
    recon_norm = norm_tensor(reconstructed)

    # Expand mask to 3 channels for overlay
    mask_3c = mask.float().repeat(1, 3, 1, 1)

    # Masked input: original with masked regions grayed out
    masked_input = orig_norm.clone()
    masked_input[mask_3c == 1] = 0.5

    # Create grid: [Orig, Masked, Recon]
    combined = torch.cat([orig_norm, masked_input, recon_norm], dim=0)
    vutils.save_image(combined, output_path, nrow=orig_norm.size(0), normalize=False, padding=2)


