import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List

from .ssl_models import RemoteSensingViT, BandAdapter, _TransformerBlock

class EPFF(nn.Module):
    """
    Efficient Pyramid Feature Fusion (EPFF)
    Fuses multi-scale features extracted from the backbone.
    """
    def __init__(self, embed_dim: int, out_dim: int = 256):
        super().__init__()
        # Simple EPFF: standard FPN-like 1x1 convolutions on 3 scales
        self.proj1 = nn.Conv2d(embed_dim, out_dim, 1)
        self.proj2 = nn.Conv2d(embed_dim, out_dim, 1)
        self.proj3 = nn.Conv2d(embed_dim, out_dim, 1)

        self.fpn_conv1 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.fpn_conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.fpn_conv3 = nn.Conv2d(out_dim, out_dim, 3, padding=1)

    def forward(self, x1, x2, x3):
        # x1, x2, x3 are spatial features from backbone blocks at different depths
        # In ViT they have the same spatial resolution, but we treat them as a pyramid
        
        c3 = self.proj3(x3)
        c2 = self.proj2(x2) + F.interpolate(c3, size=x2.shape[-2:], mode="nearest")
        c1 = self.proj1(x1) + F.interpolate(c2, size=x1.shape[-2:], mode="nearest")

        p3 = self.fpn_conv3(c3)
        p2 = self.fpn_conv2(c2)
        p1 = self.fpn_conv1(c1)
        
        return p1, p2, p3

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class UQR(nn.Module):
    """
    Unified Query Representation (UQR)
    Bridges detection tokens and spatial features for mask generation.
    """
    def __init__(self, embed_dim: int, mask_dim: int = 256):
        super().__init__()
        self.mask_embed = MLP(embed_dim, embed_dim, mask_dim, 3)

    def forward(self, det_tokens, fpff_features):
        """
        det_tokens: [B, N, C]
        fpff_features: list of spatial features [B, mask_dim, H, W]
        Returns: [B, N, H, W] mask logits
        """
        # We use the highest resolution feature (p1)
        p1 = fpff_features[0]
        # B, C, H, W -> B, H, W, C
        p1 = p1.permute(0, 2, 3, 1)
        
        # det_tokens to mask embeddings
        mask_embeds = self.mask_embed(det_tokens) # [B, N, mask_dim]
        
        # Dot product between mask embeddings and spatial features
        # [B, N, C] @ [B, C, H*W] -> [B, N, H*W]
        masks = torch.bmm(mask_embeds, p1.flatten(1, 2).transpose(1, 2))
        
        # Reshape to [B, N, H, W]
        B, N, _ = masks.shape
        _, H, W, _ = p1.shape
        return masks.view(B, N, H, W)

class RemoteSensingViDTPlus(nn.Module):
    """
    ViDT+ implementation for remote sensing imagery.
    Incorporates RAM blocks, EPFF, and UQR for joint box and mask prediction.
    """
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        num_classes: int,
        num_det_tokens: int = 300,
        ram_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_det_tokens = num_det_tokens
        
        self.band_adapter = BandAdapter(in_channels=in_channels, out_channels=3)
        self.backbone = RemoteSensingViT(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        
        # We steal the last `ram_depth` blocks from the backbone to use as RAM blocks
        assert depth > ram_depth
        self.body_blocks = self.backbone.blocks[:-ram_depth]
        self.ram_blocks = self.backbone.blocks[-ram_depth:]
        
        # Replace backbone blocks so it doesn't run them twice
        self.backbone.blocks = nn.ModuleList()

        # Object Queries (Detection Tokens)
        self.det_tokens = nn.Parameter(torch.zeros(1, num_det_tokens, embed_dim))
        self.det_pos = nn.Parameter(torch.randn(1, num_det_tokens, embed_dim))
        nn.init.trunc_normal_(self.det_tokens, std=0.02)
        
        # EPFF & UQR for mask generation
        self.epff = EPFF(embed_dim, out_dim=256)
        self.uqr = UQR(embed_dim, mask_dim=256)

        # Detection Heads
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)

    def forward(self, x):
        # 1. Band Adaptation
        x = self.band_adapter(x)

        # 2. Extract patch tokens
        patch_tokens, current_grid = self.backbone.embed_patches(x)
        positioned_tokens = self.backbone.add_position_embeddings(patch_tokens, current_grid)

        # 3. Backbone Body (extract multi-scale features for EPFF)
        features = []
        out = positioned_tokens
        
        # We'll save features roughly at 1/3, 2/3, and end of body
        milestones = [len(self.body_blocks) // 3, 2 * len(self.body_blocks) // 3, len(self.body_blocks)]
        
        for i, block in enumerate(self.body_blocks):
            out = block(out)
            if (i + 1) in milestones:
                features.append(out)

        # 4. Reconfigured Attention Module (RAM)
        # Concatenate patch tokens and detection tokens
        B = out.shape[0]
        det_tokens = self.det_tokens.expand(B, -1, -1) + self.det_pos.expand(B, -1, -1)
        
        # [B, L + N, D]
        ram_out = torch.cat([out, det_tokens], dim=1)
        
        for block in self.ram_blocks:
            ram_out = block(ram_out)
            
        # Split patches and detection tokens
        patch_tokens_out = ram_out[:, :-self.num_det_tokens]
        det_tokens_out = ram_out[:, -self.num_det_tokens:]
        
        det_tokens_out = self.backbone.norm(det_tokens_out)

        # 5. Prediction Heads
        outputs_class = self.class_embed(det_tokens_out)
        outputs_coord = self.bbox_embed(det_tokens_out).sigmoid()

        # 6. ViDT+ Mask Generation (EPFF + UQR)
        # Reshape features to spatial grids
        spatial_features = []
        for feat in features:
            feat = self.backbone.norm(feat)
            feat = feat.permute(0, 2, 1).reshape(B, self.embed_dim, current_grid[0], current_grid[1])
            spatial_features.append(feat)
            
        p1, p2, p3 = self.epff(spatial_features[0], spatial_features[1], spatial_features[2])
        masks = self.uqr(det_tokens_out, [p1, p2, p3])

        return {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,
            "pred_masks": masks,
        }
