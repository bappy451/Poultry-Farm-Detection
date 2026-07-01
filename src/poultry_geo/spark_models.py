import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List

try:
    import spconv.pytorch as spconv
except ImportError:
    spconv = None

class SparseLayerNorm(nn.Module):
    """
    Standard LayerNorm applied only to valid features in a SparseConvTensor.
    """
    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        if spconv is not None and isinstance(x, spconv.SparseConvTensor):
            # Extract features, normalize, and replace back in the sparse tensor
            x = x.replace_feature(self.ln(x.features))
        return x

class SparseReLU(nn.Module):
    """
    ReLU activation applied to the feature vector of a SparseConvTensor.
    """
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        if spconv is not None and isinstance(x, spconv.SparseConvTensor):
            return x.replace_feature(self.relu(x.features))
        return self.relu(x)

def replace_with_sparse_layers(module: nn.Module):
    """
    Recursively replaces dense Conv2d, BatchNorm2d, and ReLU layers with their 
    sparse equivalents from the spconv library.
    
    This is critical for converting standard hierarchical backbones (like ResNet) 
    into sparse encoders that only compute values for unmasked pixels.
    """
    if spconv is None:
        return module

    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            # Same-resolution convolutions use SubMConv2d (Sub-Manifold) to preserve sparsity pattern.
            # Strided convolutions use SparseConv2d to downsample while maintaining sparse representation.
            if child.stride[0] > 1 or child.stride[1] > 1:
                new_layer = spconv.SparseConv2d(
                    child.in_channels, child.out_channels,
                    kernel_size=child.kernel_size, stride=child.stride,
                    padding=child.padding, bias=(child.bias is not None)
                )
            else:
                new_layer = spconv.SubMConv2d(
                    child.in_channels, child.out_channels,
                    kernel_size=child.kernel_size, stride=child.stride,
                    padding=child.padding, bias=(child.bias is not None)
                )
            # Copy weights from the original dense layer
            with torch.no_grad():
                new_layer.weight.copy_(child.weight)
                if child.bias is not None:
                    new_layer.bias.copy_(child.bias)
            setattr(module, name, new_layer)
        elif isinstance(child, nn.BatchNorm2d):
            # Replace BatchNorm with LayerNorm for sparse processing
            new_layer = SparseLayerNorm(child.num_features)
            # BN to LN mapping is not direct, but we can copy the weights/bias for gain/offset
            with torch.no_grad():
                new_layer.ln.weight.copy_(child.weight)
                new_layer.ln.bias.copy_(child.bias)
            setattr(module, name, new_layer)
        elif isinstance(child, nn.ReLU):
            setattr(module, name, SparseReLU(inplace=child.inplace))
        else:
            replace_with_sparse_layers(child)
    return module

class SparseResNetEncoder(nn.Module):
    """
    A ResNet50-based sparse encoder for SparK SSL pretraining.
    
    This encoder processes only unmasked pixels, allowing the hierarchical CNN 
    to learn features from irregular masking patterns.
    """
    def __init__(self, in_channels=3, pretrained=False):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        
        if pretrained:
            print("Loading pretrained ResNet50 weights (ImageNet-1K v2)...")
            base = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            base = resnet50()
        
        # Initial stem: Conv(s=2) + MaxPool(s=2) = H/4 downsampling.
        # MaxPool is replaced with a strided SparseConv to avoid 'blurring' the sparsity pattern.
        if spconv is not None:
            self.stem = nn.Sequential(
                spconv.SparseConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                SparseLayerNorm(64),
                SparseReLU(inplace=True),
                spconv.SparseConv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                SparseLayerNorm(64),
                SparseReLU(inplace=True)
            )
            # Copy weights from base resnet stem if pretrained
            if pretrained:
                with torch.no_grad():
                    self.stem[0].weight.copy_(base.conv1.weight)
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            if pretrained:
                with torch.no_grad():
                    self.stem[0].weight.copy_(base.conv1.weight)
                    # BatchNorm weights are copied during replace_with_sparse_layers if spconv is None
                    # but here we use base.layer1 etc below.
        
        # ResNet stages converted to sparse operations
        self.stage1 = replace_with_sparse_layers(base.layer1)
        self.stage2 = replace_with_sparse_layers(base.layer2)
        self.stage3 = replace_with_sparse_layers(base.layer3)
        self.stage4 = replace_with_sparse_layers(base.layer4)

    def forward(self, x):
        # Input x is a SparseConvTensor or a tuple (features, indices, spatial_shape, batch_size) if spconv is missing
        if spconv is None and isinstance(x, tuple):
            features, indices, spatial_shape, batch_size = x
            B = batch_size
            C = features.shape[1]
            H, W = spatial_shape[1], spatial_shape[2]
            dense_x = torch.zeros((B, C, H, W), device=features.device, dtype=features.dtype)
            
            # Optimized Vectorization: Replace slow Python loop
            # indices is [N, 4] -> (B, Z, Y, X)
            b = indices[:, 0].long()
            y = indices[:, 2].long()
            x_idx = indices[:, 3].long()
            
            # Using advanced indexing to populate the dense tensor in one GPU operation
            # features is [N, C]
            # dense_x[b, :, y, x_idx] selects N pixels across all C channels
            dense_x[b, :, y, x_idx] = features
            x = dense_x

        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return [x1, x2, x3, x4]

class SparseSiLU(nn.Module):
    """
    SiLU (Swish) activation applied to the feature vector of a SparseConvTensor.
    Commonly used in YOLO-family backbones (CSPDarkNet).
    """
    def __init__(self, inplace=False):
        super().__init__()
        self.silu = nn.SiLU(inplace=inplace)

    def forward(self, x):
        if spconv is not None and isinstance(x, spconv.SparseConvTensor):
            return x.replace_feature(self.silu(x.features))
        return self.silu(x)

class SparseCSPDarkNetEncoder(nn.Module):
    """
    A simplified CSPDarkNet sparse encoder (YOLOv8/v11 style) that matches 
    the exact downsampling channel dimensions of Ultralytics YOLO models 
    so that pretrained weights can be loaded.
    """
    def __init__(self, in_channels=3, backbone_type="yolo11l"):
        super().__init__()
        
        # YOLOv11 standard channel configurations (p1, p2, p3, p4, p5)
        if backbone_type == "yolo11l":
            ch = [64, 128, 256, 512, 512]
        elif backbone_type == "yolo11m":
            ch = [48, 96, 192, 384, 576]
        else: # n / s
            ch = [32, 64, 128, 256, 512]

        self.encoder_channels = [ch[1], ch[2], ch[3], ch[4]]

        if spconv is not None:
            self.stem = nn.Sequential(
                spconv.SparseConv2d(in_channels, ch[0], kernel_size=3, stride=2, padding=1),
                SparseLayerNorm(ch[0]),
                SparseSiLU(inplace=True)
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, ch[0], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(ch[0]),
                nn.SiLU(inplace=True)
            )
        
        def make_stage(in_c, out_c, next_in_c):
            if spconv is not None:
                layers = [
                    spconv.SparseConv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    SparseLayerNorm(out_c),
                    SparseSiLU(inplace=True)
                ]
                if out_c != next_in_c:
                    layers.extend([
                        spconv.SubMConv2d(out_c, next_in_c, kernel_size=1),
                        SparseLayerNorm(next_in_c),
                        SparseSiLU(inplace=True)
                    ])
            else:
                layers = [
                    nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.SiLU(inplace=True)
                ]
                if out_c != next_in_c:
                    layers.extend([
                        nn.Conv2d(out_c, next_in_c, kernel_size=1),
                        nn.BatchNorm2d(next_in_c),
                        nn.SiLU(inplace=True)
                    ])
            return nn.Sequential(*layers)

        self.stage1 = make_stage(ch[0], ch[1], ch[2])
        self.stage2 = make_stage(ch[2], ch[2], ch[3])
        self.stage3 = make_stage(ch[3], ch[3], ch[4])
        self.stage4 = make_stage(ch[4], ch[4], ch[4])

    def forward(self, x):
        if spconv is None and isinstance(x, tuple):
            features, indices, spatial_shape, batch_size = x
            B = batch_size
            C = features.shape[1]
            H, W = spatial_shape[1], spatial_shape[2]
            dense_x = torch.zeros((B, C, H, W), device=features.device, dtype=features.dtype)
            
            b = indices[:, 0].long()
            y = indices[:, 2].long()
            x_idx = indices[:, 3].long()
            
            dense_x[b, :, y, x_idx] = features
            x = dense_x

        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return [x1, x2, x3, x4]

class HierarchicalDecoder(nn.Module):
    """
    A dense decoder that reconstructs the original image from sparse features.
    
    It 'fills in' the holes left by masking with a learnable mask token 
    and then uses progressive transposed convolutions to restore resolution.
    """
    def __init__(self, encoder_channels: List[int], out_channels=3):
        super().__init__()
        # learnable vector representing masked regions
        self.mask_token = nn.Parameter(torch.zeros(1, encoder_channels[-1]))
        
        # Bottleneck to process the highest latent space
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], encoder_channels[-1], 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling stages (transposed convolutions)
        self.up4 = self._make_up_block(encoder_channels[-1], encoder_channels[-2])
        self.up3 = self._make_up_block(encoder_channels[-2], encoder_channels[-3])
        self.up2 = self._make_up_block(encoder_channels[-3], encoder_channels[-4])
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[-4], 64, kernel_size=4, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )

    def _make_up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, sparse_features: List, mask: torch.Tensor):
        # 1. Take highest feature map and convert to dense
        x_sparse = sparse_features[-1]
        
        if spconv is not None and isinstance(x_sparse, spconv.SparseConvTensor):
            # spconv dense() returns (B, C, D, H, W).
            x_dense = x_sparse.dense()
            if x_dense.dim() == 5 and x_dense.shape[2] == 1:
                x_dense = x_dense.squeeze(2)
        else:
            # Fallback path: it's already dense
            x_dense = x_sparse
            
        # 2. Fill masked holes with learnable mask tokens.
        # Downsample the binary mask to match the bottleneck resolution.
        mask_down = F.interpolate(mask, size=(x_dense.shape[2], x_dense.shape[3]), mode='nearest')
        
        # Merge visible features with mask tokens for hidden regions.
        mask_token_expanded = self.mask_token.view(1, -1, 1, 1).expand_as(x_dense)
        x_dense = x_dense * (1 - mask_down) + mask_token_expanded * mask_down
        
        # 3. Progressive Upsampling to original image dimensions
        x = self.bottleneck(x_dense)
        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        
        return x

class SparKNetwork(nn.Module):
    """
    The main SparK (Sparse masKed modeling) network.
    
    This architecture combines a sparse hierarchical encoder (ResNet or CSPDarknet) 
    with a dense hierarchical decoder to reconstruct original pixel values 
    from partially visible inputs.
    """
    def __init__(self, backbone_type="resnet50", in_channels=3, pretrained=False):
        """
        Args:
            backbone_type: Type of hierarchical backbone to use (resnet50, yolo11l, etc.)
            in_channels: Number of input channels. Note: SparK typically processes 
                        3-channel adapted inputs for simplicity.
            pretrained: Whether to load ImageNet pretrained weights for the encoder.
        """
        super().__init__()
        if backbone_type == "resnet50":
            self.encoder = SparseResNetEncoder(in_channels=3, pretrained=pretrained)
            encoder_channels = [256, 512, 1024, 2048]
        elif backbone_type.startswith("yolo11"):
            self.encoder = SparseCSPDarkNetEncoder(in_channels=3, backbone_type=backbone_type)
            encoder_channels = self.encoder.encoder_channels
            if pretrained:
                self._load_yolo_pretrained(backbone_type)
        else:
            # Default to standard CSPDarkNet (e.g. YOLOv8n/s scale)
            self.encoder = SparseCSPDarkNetEncoder(in_channels=3, backbone_type="yolo11n")
            encoder_channels = self.encoder.encoder_channels
            if pretrained:
                print(f"Warning: Pretrained weights not yet supported for {backbone_type} in SparK. Starting from scratch.")
            
        self.decoder = HierarchicalDecoder(encoder_channels, out_channels=3)

    def _load_yolo_pretrained(self, backbone_type: str):
        """
        Helper to load YOLO weights from Ultralytics into our simplified encoder.
        """
        try:
            from ultralytics import YOLO
            print(f"Loading pretrained {backbone_type} weights from Ultralytics...")
            # We download the classification model to get pure backbone weights
            # Names: yolo11n-cls.pt, yolo11s-cls.pt, etc.
            yolo = YOLO(f"{backbone_type}-cls.pt")
            yolo_state = yolo.model.state_dict()
            
            mapping = {
                "model.0.conv.weight": "encoder.stem.0.weight",
                "model.0.bn.weight": "encoder.stem.1.ln.weight",
                "model.0.bn.bias": "encoder.stem.1.ln.bias",
                
                "model.1.conv.weight": "encoder.stage1.0.weight",
                "model.1.bn.weight": "encoder.stage1.1.ln.weight",
                "model.1.bn.bias": "encoder.stage1.1.ln.bias",
                
                "model.3.conv.weight": "encoder.stage2.0.weight",
                "model.3.bn.weight": "encoder.stage2.1.ln.weight",
                "model.3.bn.bias": "encoder.stage2.1.ln.bias",
                
                "model.5.conv.weight": "encoder.stage3.0.weight",
                "model.5.bn.weight": "encoder.stage3.1.ln.weight",
                "model.5.bn.bias": "encoder.stage3.1.ln.bias",
                
                "model.7.conv.weight": "encoder.stage4.0.weight",
                "model.7.bn.weight": "encoder.stage4.1.ln.weight",
                "model.7.bn.bias": "encoder.stage4.1.ln.bias",
            }
            
            new_state = {}
            for k, v in yolo_state.items():
                if k in mapping:
                    new_state[mapping[k]] = v
            
            msg = self.load_state_dict(new_state, strict=False)
            print(f"  YOLO pretrained weights loaded: {len(new_state)} tensors. Missing: {len(msg.missing_keys)}")
            
        except Exception as e:
            print(f"Warning: Could not load YOLO pretrained weights: {e}")

    def forward(self, sparse_tensor, mask):
        """
        Args:
            sparse_tensor: spconv.SparseConvTensor containing unmasked pixels.
            mask: Binary mask grid (1=masked, 0=unmasked).
        Returns:
            reconstructed: Reconstructed dense image of shape (B, 3, H, W).
        """
        # 1. Encode unmasked pixels into hierarchical features
        features = self.encoder(sparse_tensor)
        
        # 2. Decode features back into dense pixel space, filling in masked holes
        reconstructed = self.decoder(features, mask)
        return reconstructed
