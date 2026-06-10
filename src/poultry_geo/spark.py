import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import spconv.pytorch as spconv
except ImportError:
    # Fallback or Mock for environments where spconv is not installed
    # This allows the logic to be defined even if the library is missing
    spconv = None

class PatchMasker:
    """
    Generates random binary masks for patch-based image masking.
    
    This utility is used in Self-Supervised Learning (SSL) objectives like 
    Masked Image Modeling (MIM) to hide portions of the input image, 
    forcing the model to learn spatial context and semantic features 
    to reconstruct the missing regions.
    """
    def __init__(self, patch_size: int = 16, mask_ratio: float = 0.75):
        """
        Args:
            patch_size: The size of each square patch in pixels.
            mask_ratio: Fraction of the image to be masked (0.0 to 1.0).
        """
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def generate_mask(self, B: int, H: int, W: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Generates a binary mask of shape (B, 1, H, W).
        1 indicates a masked pixel, 0 indicates an unmasked (visible) pixel.
        """
        h_p, w_p = H // self.patch_size, W // self.patch_size
        num_patches = h_p * w_p
        num_mask = int(num_patches * self.mask_ratio)
        
        # Optimized: Vectorized mask generation without loops
        noise = torch.rand(B, num_patches, device=device)
        # 1 = masked, 0 = unmasked
        mask = torch.zeros(B, num_patches, device=device)
        mask.scatter_(1, noise.topk(num_mask, dim=1).indices, 1.0)
            
        # Reshape and upsample
        mask = mask.reshape(B, 1, h_p, w_p)
        mask = F.interpolate(mask, size=(H, W), mode='nearest')
        
        return mask

class SparseTransformer(nn.Module):
    """
    Converts masked multi-band remote sensing imagery into native Sparse Tensors.
    
    This module bridges standard dense image processing with Sparse Convolution 
    frameworks (like spconv). It is specifically designed for SparK-style 
    SSL pretraining, where only the unmasked (visible) pixels are processed 
    by the sparse encoder, significantly reducing computation while 
    enabling hierarchical CNNs to learn from masked inputs.
    """
    def __init__(self, adapter: nn.Module, patch_size: int = 16, mask_ratio: float = 0.75):
        """
        Args:
            adapter: A BandAdapter module to project multi-band input into 3-channel space.
            patch_size: Pixel size of the masking patches.
            mask_ratio: Fraction of the image to mask.
        """
        super().__init__()
        self.adapter = adapter
        self.masker = PatchMasker(patch_size=patch_size, mask_ratio=mask_ratio)

    def forward(self, x: torch.Tensor):
        """
        Transforms a dense image batch into a SparseConvTensor containing only visible pixels.
        """
        B, C, H, W = x.shape
        device = x.device
        
        # 1. BandAdapter processing
        x_adapted = self.adapter(x)
        
        # 2. Generate patch-level mask
        h_p, w_p = H // self.masker.patch_size, W // self.masker.patch_size
        num_patches = h_p * w_p
        num_mask = int(num_patches * self.masker.mask_ratio)
        
        noise = torch.rand(B, num_patches, device=device)
        patch_mask = torch.zeros(B, num_patches, device=device)
        patch_mask.scatter_(1, noise.topk(num_mask, dim=1).indices, 1.0)
        
        # Expand to full resolution mask for loss calculation
        mask = patch_mask.reshape(B, 1, h_p, w_p)
        mask = F.interpolate(mask, size=(H, W), mode='nearest')

        # 3. Optimized coordinate extraction (Patch-based expansion)
        # Find indices of visible patches (patch_mask == 0)
        vis_patch_indices = (patch_mask == 0).nonzero(as_tuple=False) # [num_vis_patches, 2] -> (Batch, PatchIdx)
        
        if vis_patch_indices.shape[0] == 0:
            # Fallback for edge case: no visible pixels
            if spconv is not None:
                return spconv.SparseConvTensor(torch.empty(0, 3, device=device), torch.empty(0, 4, device=device), [1, H, W], B), mask
            return (torch.empty(0, 3, device=device), torch.empty(0, 4, device=device), [1, H, W], B), mask

        # Expand patch indices to pixel coordinates
        # Each patch becomes patch_size * patch_size pixels
        p_s = self.masker.patch_size
        num_pixels_per_patch = p_s * p_s
        
        # Local pixel grid
        yy, xx = torch.meshgrid(
            torch.arange(p_s, device=device),
            torch.arange(p_s, device=device),
            indexing='ij'
        )
        yy = yy.reshape(-1) # [p_s*p_s]
        xx = xx.reshape(-1) # [p_s*p_s]
        
        # Patch top-left corners
        b_idx = vis_patch_indices[:, 0]
        p_idx = vis_patch_indices[:, 1]
        p_y = (p_idx // w_p) * p_s
        p_x = (p_idx % w_p) * p_s
        
        # Repeat and offset to get pixel coordinates
        # Using broadcasting to avoid loops
        b_idx_ext = b_idx.repeat_interleave(num_pixels_per_patch)
        y_idx_ext = p_y.repeat_interleave(num_pixels_per_patch) + yy.repeat(vis_patch_indices.shape[0])
        x_idx_ext = p_x.repeat_interleave(num_pixels_per_patch) + xx.repeat(vis_patch_indices.shape[0])
        
        # Prepare indices [N, 4] -> (B, Z, Y, X)
        z_idx_ext = torch.zeros_like(b_idx_ext)
        indices = torch.stack([b_idx_ext.int(), z_idx_ext.int(), y_idx_ext.int(), x_idx_ext.int()], dim=1)
        
        # Gather features
        features = x_adapted[b_idx_ext, :, y_idx_ext, x_idx_ext]
        
        # 4. Instantiate the Sparse Tensor
        if spconv is not None:
            sparse_tensor = spconv.SparseConvTensor(
                features=features,
                indices=indices,
                spatial_shape=[1, H, W],
                batch_size=B
            )
        else:
            sparse_tensor = (features, indices, [1, H, W], B)
            
        return sparse_tensor, mask
