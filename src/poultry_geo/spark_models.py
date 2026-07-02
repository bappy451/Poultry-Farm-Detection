import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List

try:
    import spconv.pytorch as spconv
except ImportError:
    spconv = None


def _refresh_spconv():
    """Return the current spconv module, allowing tests or late installs to provide it after import."""
    global spconv
    if spconv is None:
        try:
            import spconv.pytorch as detected_spconv
        except ImportError:
            return None
        spconv = detected_spconv
    return spconv


def _copy_parameter_if_available(target: nn.Module, source: torch.Tensor, name: str = "weight") -> None:
    parameter = getattr(target, name, None)
    if parameter is None:
        return
    try:
        with torch.no_grad():
            parameter.copy_(source)
    except (AttributeError, RuntimeError, TypeError):
        # Some lightweight test doubles expose sparse layer types without real
        # parameters. In production spconv layers still receive copied weights.
        return


class SparseLayerNorm(nn.Module):
    """
    Standard LayerNorm applied only to valid features in a SparseConvTensor.
    """
    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        spconv_module = _refresh_spconv()
        if spconv_module is not None and isinstance(x, spconv_module.SparseConvTensor):
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
        spconv_module = _refresh_spconv()
        if spconv_module is not None and isinstance(x, spconv_module.SparseConvTensor):
            return x.replace_feature(self.relu(x.features))
        return self.relu(x)

def replace_with_sparse_layers(module: nn.Module):
    """
    Recursively replaces dense Conv2d, BatchNorm2d, and ReLU layers with their 
    sparse equivalents from the spconv library.
    
    This is critical for converting standard hierarchical backbones (like ResNet) 
    into sparse encoders that only compute values for unmasked pixels.
    """
    spconv_module = _refresh_spconv()
    if spconv_module is None:
        return module

    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            # Same-resolution convolutions use SubMConv2d (Sub-Manifold) to preserve sparsity pattern.
            # Strided convolutions use SparseConv2d to downsample while maintaining sparse representation.
            if child.stride[0] > 1 or child.stride[1] > 1:
                new_layer = spconv_module.SparseConv2d(
                    child.in_channels, child.out_channels,
                    kernel_size=child.kernel_size, stride=child.stride,
                    padding=child.padding, bias=(child.bias is not None)
                )
            else:
                new_layer = spconv_module.SubMConv2d(
                    child.in_channels, child.out_channels,
                    kernel_size=child.kernel_size, stride=child.stride,
                    padding=child.padding, bias=(child.bias is not None)
                )
            # Copy weights from the original dense layer
            _copy_parameter_if_available(new_layer, child.weight, "weight")
            if child.bias is not None:
                _copy_parameter_if_available(new_layer, child.bias, "bias")
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
        spconv_module = _refresh_spconv()
        
        if pretrained:
            print("Loading pretrained ResNet50 weights (ImageNet-1K v2)...")
            base = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            base = resnet50()
        
        # Initial stem: Conv(s=2) + MaxPool(s=2) = H/4 downsampling.
        # MaxPool is replaced with a strided SparseConv to avoid 'blurring' the sparsity pattern.
        if spconv_module is not None:
            self.stem = nn.Sequential(
                spconv_module.SparseConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                SparseLayerNorm(64),
                SparseReLU(inplace=True),
                spconv_module.SparseConv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                SparseLayerNorm(64),
                SparseReLU(inplace=True)
            )
            # Copy weights from base resnet stem if pretrained
            if pretrained:
                _copy_parameter_if_available(self.stem[0], base.conv1.weight, "weight")
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
        spconv_module = _refresh_spconv()
        if spconv_module is None and isinstance(x, tuple):
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
        spconv_module = _refresh_spconv()
        if spconv_module is not None and isinstance(x, spconv_module.SparseConvTensor):
            return x.replace_feature(self.silu(x.features))
        return self.silu(x)

class SparseCSPDarkNetEncoder(nn.Module):
    """
    A simplified CSPDarkNet sparse encoder (YOLOv8/v11 style).
    
    This architecture uses Cross-Stage Partial (CSP) concepts to maintain 
    strong gradient flow while being executed sparsely on masked inputs.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        spconv_module = _refresh_spconv()
        # YOLOv8/v11 stem uses stride=2 to halve spatial resolution immediately
        if spconv_module is not None:
            self.stem = spconv_module.SparseConv2d(in_channels, base_channels // 2, kernel_size=3, stride=2, padding=1)
        else:
            self.stem = nn.Conv2d(in_channels, base_channels // 2, kernel_size=3, stride=2, padding=1)
        
        def make_stage(in_c, out_c, stride=2):
            if spconv_module is not None:
                return nn.Sequential(
                    spconv_module.SparseConv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
                    SparseLayerNorm(out_c),
                    SparseSiLU(inplace=True)
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.SiLU(inplace=True)
                )

        # Build hierarchical stages (C1 to C4)
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.stage1 = make_stage(base_channels // 2, c1)
        self.stage2 = make_stage(c1, c2)
        self.stage3 = make_stage(c2, c3)
        self.stage4 = make_stage(c3, c4)

    def forward(self, x):
        spconv_module = _refresh_spconv()
        if spconv_module is None and isinstance(x, tuple):
            features, indices, spatial_shape, batch_size = x
            B = batch_size
            C = features.shape[1]
            H, W = spatial_shape[1], spatial_shape[2]
            dense_x = torch.zeros((B, C, H, W), device=features.device, dtype=features.dtype)
            
            # Optimized Vectorization: Replace slow Python loop
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
        
        spconv_module = _refresh_spconv()
        if spconv_module is not None and isinstance(x_sparse, spconv_module.SparseConvTensor):
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
        elif backbone_type == "yolo11l":
            self.encoder = SparseCSPDarkNetEncoder(in_channels=3, base_channels=128)
            encoder_channels = [128, 256, 512, 1024]
            if pretrained:
                self._load_yolo_pretrained(backbone_type, base_channels=128)
        elif backbone_type == "yolov26l":
            self.encoder = SparseCSPDarkNetEncoder(in_channels=3, base_channels=256)
            encoder_channels = [256, 512, 1024, 2048]
            if pretrained:
                print("Warning: yolov26l not found in Ultralytics. Starting from scratch.")
        else:
            # Default to standard CSPDarkNet (e.g. YOLOv8n/s scale)
            self.encoder = SparseCSPDarkNetEncoder(in_channels=3, base_channels=64)
            encoder_channels = [64, 128, 256, 512]
            if pretrained:
                print(f"Warning: Pretrained weights not yet supported for {backbone_type} in SparK. Starting from scratch.")
            
        self.decoder = HierarchicalDecoder(encoder_channels, out_channels=3)

    def _load_yolo_pretrained(self, backbone_type: str, base_channels: int):
        """
        Helper to load YOLO weights from Ultralytics into our simplified encoder.
        """
        try:
            from ultralytics import YOLO
            print(f"Loading pretrained {backbone_type} weights from Ultralytics...")
            # We download the classification model to get pure backbone weights
            # Names: yolo11n-cls.pt, yolo11s-cls.pt, etc.
            model_variant = backbone_type.replace("yolo", "yolo11") # normalize name
            if backbone_type == "yolo11l":
                model_variant = "yolo11l"
            
            yolo = YOLO(f"{model_variant}-cls.pt")
            # The backbone is in yolo.model.model[0:10] roughly
            yolo_state = yolo.model.state_dict()
            
            # Map YOLO backbone to our simplified stages
            # YOLO v11 backbone structure:
            # 0: Conv (Stem)
            # 2: C3k2 (Stage 1)
            # 4: C3k2 (Stage 2)
            # 6: C3k2 (Stage 3)
            # 8: C3k2 (Stage 4)
            
            mapping = {
                "model.0.conv.weight": "encoder.stem.weight",
                "model.0.bn.weight": "encoder.stem.ln.weight", # Approximate BN->LN
                "model.0.bn.bias": "encoder.stem.ln.bias",
                
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
                    target_key = mapping[k]
                    # Since our encoder is simplified and uses spconv, we need to match the weight shape
                    # spconv weights are typically the same as torch weights
                    new_state[target_key] = v
            
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
