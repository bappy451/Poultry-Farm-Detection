import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from poultry_geo.spark import require_spconv

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


def dense_conv_weight_to_spconv(weight: torch.Tensor) -> torch.Tensor:
    """Convert a PyTorch Conv2d OIHW kernel to spconv 2.x OHWI layout."""
    if weight.ndim != 4:
        raise ValueError(f"Expected a 4-D OIHW Conv2d kernel, got {tuple(weight.shape)}.")
    return weight.permute(0, 2, 3, 1).contiguous()


def spconv_weight_to_dense_conv(weight: torch.Tensor) -> torch.Tensor:
    """Convert an spconv 2.x OHWI kernel to PyTorch Conv2d OIHW layout."""
    if weight.ndim != 4:
        raise ValueError(f"Expected a 4-D OHWI sparse kernel, got {tuple(weight.shape)}.")
    return weight.permute(0, 3, 1, 2).contiguous()


def _copy_parameter(target: nn.Module, source: torch.Tensor, name: str = "weight", *,
                    conv_oiwh_to_krsc: bool = False) -> None:
    """Copy a parameter exactly, rejecting unsupported/mismatched layouts."""
    parameter = getattr(target, name, None)
    if parameter is None:
        raise RuntimeError(f"Sparse layer has no {name!r} parameter to initialize.")
    value = dense_conv_weight_to_spconv(source) if conv_oiwh_to_krsc else source
    if tuple(parameter.shape) != tuple(value.shape):
        raise RuntimeError(
            f"Cannot initialize sparse {name}: target shape {tuple(parameter.shape)} "
            f"does not match converted source shape {tuple(value.shape)}."
        )
    with torch.no_grad():
        parameter.copy_(value)


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

class SparseBatchNorm(nn.Module):
    """BatchNorm over active sparse features, preserving BN parameters/statistics."""
    def __init__(self, source: nn.BatchNorm2d):
        super().__init__()
        self.bn = nn.BatchNorm1d(source.num_features, eps=source.eps, momentum=source.momentum,
                                 affine=source.affine, track_running_stats=source.track_running_stats)
        self.bn.load_state_dict(source.state_dict())

    def forward(self, x):
        spconv_module = _refresh_spconv()
        if spconv_module is not None and isinstance(x, spconv_module.SparseConvTensor):
            if self.training and x.features.shape[0] < 2:
                raise RuntimeError(
                    "SparseBatchNorm requires at least two active features while training; "
                    "increase image/batch size or reduce the mask ratio."
                )
            return x.replace_feature(self.bn(x.features))
        return self.bn(x)


class SparseReLU(nn.Module):
    """
    ReLU activation applied to the feature vector of a SparseConvTensor.
    """
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        spconv_module = _refresh_spconv()
        if spconv_module is not None and isinstance(x, spconv_module.SparseConvTensor):
            return x.replace_feature(self.relu(x.features))
        return self.relu(x)

def _sparse_add(left, right):
    """Add sparse residual branches while respecting their coordinate sets."""
    spconv_module = _refresh_spconv()
    if spconv_module is None or not isinstance(left, spconv_module.SparseConvTensor):
        return left + right
    if torch.equal(left.indices, right.indices):
        return left.replace_feature(left.features + right.features)
    from spconv.pytorch import functional as spconv_functional
    return spconv_functional.sparse_add(left, right)

def _sparse_chunk(x, chunks, dim=1):
    spconv_module = _refresh_spconv()
    if spconv_module is None or not isinstance(x, spconv_module.SparseConvTensor):
        return x.chunk(chunks, dim)
    feat_chunks = x.features.chunk(chunks, dim=1)
    return [spconv_module.SparseConvTensor(fc, x.indices, x.spatial_shape, x.batch_size) for fc in feat_chunks]

def _sparse_split(x, split_size, dim=1):
    spconv_module = _refresh_spconv()
    if spconv_module is None or not isinstance(x, spconv_module.SparseConvTensor):
        return x.split(split_size, dim)
    feat_splits = x.features.split(split_size, dim=1)
    return [spconv_module.SparseConvTensor(fs, x.indices, x.spatial_shape, x.batch_size) for fs in feat_splits]

def _sparse_cat(tensors, dim=1):
    spconv_module = _refresh_spconv()
    if spconv_module is None or not isinstance(tensors[0], spconv_module.SparseConvTensor):
        return torch.cat(tensors, dim)
    feat_cat = torch.cat([t.features for t in tensors], dim=1)
    t0 = tensors[0]
    return spconv_module.SparseConvTensor(feat_cat, t0.indices, t0.spatial_shape, t0.batch_size)

def patch_yolo_modules(module: nn.Module):
    spconv_module = _refresh_spconv()
    if spconv_module is None:
        return module

    for name, child in module.named_modules():
        cname = child.__class__.__name__
        
        if cname in ['C2f', 'C3k2']:
            def sparse_c3k2_forward(x, _block=child):
                out1 = _block.cv1(x)
                y = list(_sparse_chunk(out1, 2, 1))
                for m in _block.m:
                    y.append(m(y[-1]))
                return _block.cv2(_sparse_cat(y, 1))
            child.forward = sparse_c3k2_forward
            
        elif cname in ['C3', 'C3k']:
            def sparse_c3_forward(x, _block=child):
                return _block.cv3(_sparse_cat((_block.m(_block.cv1(x)), _block.cv2(x)), 1))
            child.forward = sparse_c3_forward
            
        elif cname == 'Bottleneck':
            def sparse_bottleneck_forward(x, _block=child):
                out = _block.cv2(_block.cv1(x))
                if _block.add:
                    return _sparse_add(x, out)
                return out
            child.forward = sparse_bottleneck_forward
            
        elif cname == 'SPPF':
            # Save original forward to call it on dense tensors
            _block_original_forward = child.forward
            def sparse_sppf_forward(x, _block=child, _orig_fwd=_block_original_forward):
                spm = _refresh_spconv()
                if spm is not None and isinstance(x, spm.SparseConvTensor):
                    dense_x = x.dense()
                    # We fill masked regions with a very small number so max pooling ignores them
                    # Actually dense() fills with 0. 0 is fine for SPPF because features are after SiLU which are mostly > 0.
                    # Or we can just let it max pool the 0s.
                    out = _orig_fwd(dense_x)
                    
                    b_idx = x.indices[:, 0].long()
                    y_idx = x.indices[:, 1].long()
                    x_idx = x.indices[:, 2].long()
                    sparse_features = out[b_idx, :, y_idx, x_idx]
                    return x.replace_feature(sparse_features)
                else:
                    return _orig_fwd(x)
            child.forward = sparse_sppf_forward
            
        elif cname == 'C2PSA':
            def sparse_c2psa_forward(x, _block=child):
                cv1_out = _block.cv1(x)
                a, b = _sparse_split(cv1_out, (_block.c, _block.c), dim=1)
                b = _block.m(b)
                return _block.cv2(_sparse_cat((a, b), 1))
            child.forward = sparse_c2psa_forward
            
        elif cname == 'Attention':
            def sparse_attention_forward(x, _block=child):
                spm = _refresh_spconv()
                if spm is not None and isinstance(x, spm.SparseConvTensor):
                    dense_x = x.dense()
                    B, C, H, W = dense_x.shape
                    N = H * W
                    qkv = _block.qkv(dense_x)
                    q, k, v = qkv.view(B, _block.num_heads, _block.key_dim * 2 + _block.head_dim, N).split(
                        [_block.key_dim, _block.key_dim, _block.head_dim], dim=2
                    )
                    attn = (q.transpose(-2, -1) @ k) * _block.scale
                    attn = attn.softmax(dim=-1)
                    out = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + _block.pe(v.reshape(B, C, H, W))
                    out = _block.proj(out)
                    
                    b_idx = x.indices[:, 0].long()
                    y_idx = x.indices[:, 1].long()
                    x_idx = x.indices[:, 2].long()
                    sparse_features = out[b_idx, :, y_idx, x_idx]
                    return x.replace_feature(sparse_features)
                else:
                    B, C, H, W = x.shape
                    N = H * W
                    qkv = _block.qkv(x)
                    q, k, v = qkv.view(B, _block.num_heads, _block.key_dim * 2 + _block.head_dim, N).split(
                        [_block.key_dim, _block.key_dim, _block.head_dim], dim=2
                    )
                    attn = (q.transpose(-2, -1) @ k) * _block.scale
                    attn = attn.softmax(dim=-1)
                    out = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + _block.pe(v.reshape(B, C, H, W))
                    return _block.proj(out)
            child.forward = sparse_attention_forward
            
    return module


def _patch_sparse_resnet_bottlenecks(module: nn.Module) -> None:
    """Make torchvision Bottleneck residual addition safe for sparse tensors."""
    for block in module.modules():
        if block.__class__.__name__ != "Bottleneck":
            continue

        def sparse_forward(x, _block=block):
            identity = x
            out = _block.conv1(x)
            out = _block.bn1(out)
            out = _block.relu(out)
            out = _block.conv2(out)
            out = _block.bn2(out)
            out = _block.relu(out)
            out = _block.conv3(out)
            out = _block.bn3(out)
            if _block.downsample is not None:
                identity = _block.downsample(x)
            out = _sparse_add(out, identity)
            return _block.relu(out)

        block.forward = sparse_forward

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
        if child.__class__.__name__ in ['Attention', 'SPPF']:
            continue  # Attention and SPPF handle their own densification; their inner layers must remain dense.
            
            
        if isinstance(child, nn.Conv2d):
            # Same-resolution convolutions use SubMConv2d (Sub-Manifold) to preserve sparsity pattern.
            # Strided convolutions use SparseConv2d to downsample while maintaining sparse representation.
            if child.stride[0] > 1 or child.stride[1] > 1:
                new_layer = spconv_module.SparseConv2d(
                    child.in_channels, child.out_channels,
                    kernel_size=child.kernel_size, stride=child.stride,
                    padding=child.padding, dilation=child.dilation,
                    groups=child.groups, bias=(child.bias is not None)
                )
            else:
                new_layer = spconv_module.SubMConv2d(
                    child.in_channels, child.out_channels,
                    kernel_size=child.kernel_size, stride=child.stride,
                    padding=child.padding, dilation=child.dilation,
                    groups=child.groups, bias=(child.bias is not None)
                )
            # Copy weights from the original dense layer
            _copy_parameter(new_layer, child.weight, "weight", conv_oiwh_to_krsc=True)
            if child.bias is not None:
                _copy_parameter(new_layer, child.bias, "bias")
            setattr(module, name, new_layer)
        elif isinstance(child, nn.BatchNorm2d):
            setattr(module, name, SparseBatchNorm(child))
        elif isinstance(child, nn.ReLU):
            setattr(module, name, SparseReLU(inplace=child.inplace))
        elif isinstance(child, nn.SiLU):
            setattr(module, name, SparseSiLU(inplace=child.inplace))
        elif isinstance(child, nn.MaxPool2d):
            new_layer = spconv_module.SparseMaxPool2d(
                kernel_size=child.kernel_size, stride=child.stride, padding=child.padding
            )
            setattr(module, name, new_layer)
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
                SparseBatchNorm(base.bn1),
                SparseReLU(inplace=True),
                spconv_module.SparseConv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                SparseBatchNorm(nn.BatchNorm2d(64)),
                SparseReLU(inplace=True)
            )
            # Copy weights from base resnet stem if pretrained
            if pretrained:
                _copy_parameter(self.stem[0], base.conv1.weight, "weight", conv_oiwh_to_krsc=True)
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

        # ResNet stages converted to sparse operations
        self.stage1 = replace_with_sparse_layers(base.layer1)
        self.stage2 = replace_with_sparse_layers(base.layer2)
        self.stage3 = replace_with_sparse_layers(base.layer3)
        self.stage4 = replace_with_sparse_layers(base.layer4)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            _patch_sparse_resnet_bottlenecks(stage)

    def forward(self, x):
        # Fallback: if spconv is absent and x is a tuple, reconstruct dense tensor.
        # Indices format: [N, 3] -> (batch, y, x); spatial_shape = [H, W].
        spconv_module = _refresh_spconv()
        if spconv_module is None and isinstance(x, tuple):
            features, indices, spatial_shape, batch_size = x
            B = batch_size
            H, W = spatial_shape[0], spatial_shape[1]
            dense_x = torch.zeros((B, features.shape[1], H, W), device=features.device, dtype=features.dtype)
            b   = indices[:, 0].long()
            y   = indices[:, 1].long()
            x_i = indices[:, 2].long()
            dense_x[b, :, y, x_i] = features
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
        self.silu = nn.SiLU(inplace=False)

    def forward(self, x):
        spconv_module = _refresh_spconv()
        if spconv_module is not None and isinstance(x, spconv_module.SparseConvTensor):
            return x.replace_feature(self.silu(x.features))
        return self.silu(x)

class SparseCSPDarkNetEncoder(nn.Module):
    """
    A YOLO11 (CSPDarkNet) based sparse encoder for SparK SSL pretraining.
    Automatically identifies and patches YOLO modules to operate efficiently on spconv tensors.
    """
    def __init__(self, backbone_type="yolo11l", in_channels=3, pretrained=False):
        super().__init__()
        from ultralytics import YOLO
        import copy
        
        if not backbone_type.startswith("yolo"):
            raise ValueError(f"SparseCSPDarkNetEncoder expects a yolo backbone, got {backbone_type}")
            
        model_name = backbone_type if backbone_type.endswith('.pt') else f"{backbone_type}-obb.pt"
        if not pretrained:
            model_name = model_name.replace(".pt", ".yaml")
            
        print(f"Loading YOLO backbone from {model_name} (pretrained={pretrained})...")
        model = YOLO(model_name)
        
        # The backbone is layers 0 through 10
        self.backbone = model.model.model[:11]
        
        if in_channels != 3:
            orig_conv = self.backbone[0].conv
            self.backbone[0].conv = nn.Conv2d(in_channels, orig_conv.out_channels, 
                                              kernel_size=orig_conv.kernel_size, stride=orig_conv.stride,
                                              padding=orig_conv.padding, bias=(orig_conv.bias is not None))
            if pretrained:
                print(f"Warning: in_channels={in_channels}, discarding pretrained weights for stem")
                
        # Patch to sparse layers
        self.backbone = replace_with_sparse_layers(self.backbone)
        self.backbone = patch_yolo_modules(self.backbone)

    def forward(self, x):
        features = []
        # Fallback to dense reconstruction for non-spconv testing
        spconv_module = _refresh_spconv()
        if spconv_module is None and isinstance(x, tuple):
            feat, indices, spatial_shape, batch_size = x
            B, H, W = batch_size, spatial_shape[0], spatial_shape[1]
            dense_x = torch.zeros((B, feat.shape[1], H, W), device=feat.device, dtype=feat.dtype)
            dense_x[indices[:, 0].long(), :, indices[:, 1].long(), indices[:, 2].long()] = feat
            x = dense_x

        for i, module in enumerate(self.backbone):
            x = module(x)
            if i in [2, 4, 6, 10]:
                features.append(x)
        return features

class HierarchicalDecoder(nn.Module):
    """
    A dense decoder that reconstructs the original image from sparse features.

    It 'fills in' the holes left by masking with a learnable mask token
    for each hierarchical scale, and then uses progressive transposed convolutions
    and skip connections to restore resolution (UNet-style).
    """
    def __init__(self, encoder_channels: List[int], out_channels=3):
        super().__init__()
        # 1 mask token per stage to fill holes at every scale
        self.mask_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, ch)) for ch in encoder_channels
        ])

        # Bottleneck to process the highest latent space
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], encoder_channels[-1], 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Upsampling stages with skip connections
        self.up_conv4 = nn.ConvTranspose2d(encoder_channels[-1], encoder_channels[-2], kernel_size=2, stride=2)
        self.proj3 = nn.Sequential(
            nn.Conv2d(encoder_channels[-2], encoder_channels[-2], 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up_conv3 = nn.ConvTranspose2d(encoder_channels[-2], encoder_channels[-3], kernel_size=2, stride=2)
        self.proj2 = nn.Sequential(
            nn.Conv2d(encoder_channels[-3], encoder_channels[-3], 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up_conv2 = nn.ConvTranspose2d(encoder_channels[-3], encoder_channels[-4], kernel_size=2, stride=2)
        self.proj1 = nn.Sequential(
            nn.Conv2d(encoder_channels[-4], encoder_channels[-4], 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[-4], 64, kernel_size=4, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )

    def forward(self, sparse_features: List, mask: torch.Tensor):
        dense_features = []
        spconv_module = _refresh_spconv()

        # 1. Convert all hierarchical features to dense.
        #    With 2-D spatial_shape=[H,W], sf.dense() returns [B, C, H, W] directly.
        for sf in sparse_features:
            if spconv_module is not None and isinstance(sf, spconv_module.SparseConvTensor):
                df = sf.dense()
                # Safety squeeze in case spatial_shape was inadvertently set to 3-D.
                if df.dim() == 5 and df.shape[2] == 1:
                    df = df.squeeze(2)
                dense_features.append(df)
            else:
                dense_features.append(sf)

        # 2. Fill masked holes with learnable mask tokens at ALL scales.
        filled_features = []
        for i, df in enumerate(dense_features):
            mask_down = F.interpolate(mask, size=(df.shape[2], df.shape[3]), mode='nearest')
            token = self.mask_tokens[i].to(dtype=df.dtype).view(1, -1, 1, 1).expand_as(df)
            filled = df * (1 - mask_down) + token * mask_down
            filled_features.append(filled)

        if len(filled_features) != 4:
            raise ValueError("HierarchicalDecoder requires exactly four encoder feature maps.")
        x1, x2, x3, x4 = filled_features
        if x3.shape[-2:] != tuple(dim * 2 for dim in x4.shape[-2:]) or x2.shape[-2:] != tuple(dim * 2 for dim in x3.shape[-2:]) or x1.shape[-2:] != tuple(dim * 2 for dim in x2.shape[-2:]):
            raise ValueError("Encoder feature maps must form a factor-of-two spatial pyramid.")

        # 3. Progressive Upsampling with Skip Connections.
        x = self.bottleneck(x4)

        x = self.up_conv4(x)
        x = x + x3
        x = self.proj3(x)

        x = self.up_conv3(x)
        x = x + x2
        x = self.proj2(x)

        x = self.up_conv2(x)
        x = x + x1
        x = self.proj1(x)

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
        require_spconv()
        if backbone_type != "resnet50" and not backbone_type.startswith("yolo"):
            raise ValueError(
                f"Unsupported SparK backbone {backbone_type!r}. Supported are 'resnet50' or 'yolo...'"
            )
        self.backbone_type = backbone_type
        if backbone_type == "resnet50":
            self.encoder = SparseResNetEncoder(in_channels=in_channels, pretrained=pretrained)
            encoder_channels = [256, 512, 1024, 2048]
        else:
            self.encoder = SparseCSPDarkNetEncoder(backbone_type=backbone_type, in_channels=in_channels, pretrained=pretrained)
            # Perform a dummy pass to infer encoder channels (since YOLO variants change channel counts)
            with torch.no_grad():
                dummy_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                dummy = torch.zeros(1, in_channels, 256, 256, device=dummy_device)
                self.encoder.to(dummy_device)
                
                if getattr(self.encoder, "backbone", None) is not None:
                    # Pass through original dense layers before we try to use it
                    # Actually, the layers are already sparse if spconv is present. 
                    # If spconv is present, we must generate a dummy SparseConvTensor
                    spm = _refresh_spconv()
                    if spm is not None:
                        dummy_sparse = spm.SparseConvTensor(
                            features=dummy.view(dummy.shape[0], dummy.shape[1], -1).permute(0, 2, 1).reshape(-1, dummy.shape[1]).to(dummy.device),
                            indices=torch.cartesian_prod(torch.arange(1), torch.arange(256), torch.arange(256)).int().to(dummy.device),
                            spatial_shape=[256, 256],
                            batch_size=1
                        )
                        out = self.encoder(dummy_sparse)
                    else:
                        out = self.encoder(dummy)
                encoder_channels = [o.features.shape[1] if hasattr(o, 'features') else o.shape[1] for o in out]
            print(f"Inferred {backbone_type} encoder channels: {encoder_channels}")

        self.decoder = HierarchicalDecoder(encoder_channels, out_channels=3)

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





