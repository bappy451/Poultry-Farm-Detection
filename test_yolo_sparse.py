import torch
import torch.nn as nn
from ultralytics import YOLO

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from poultry_geo.spark_models import _refresh_spconv, replace_with_sparse_layers

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
        
        if cname in ['C2f', 'C3k2', 'C3k']:
            def sparse_c3k2_forward(x, _block=child):
                out1 = _block.cv1(x)
                y = list(_sparse_chunk(out1, 2, 1))
                for m in _block.m:
                    y.append(m(y[-1]))
                return _block.cv2(_sparse_cat(y, 1))
            child.forward = sparse_c3k2_forward
            
        elif cname == 'SPPF':
            def sparse_sppf_forward(x, _block=child):
                y = [_block.cv1(x)]
                for _ in range(getattr(_block, 'n', 3)):
                    y.append(_block.m(y[-1]))
                res = _block.cv2(_sparse_cat(y, 1))
                if getattr(_block, 'add', False):
                    from poultry_geo.spark_models import _sparse_add
                    return _sparse_add(res, x)
                return res
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

if __name__ == "__main__":
    from poultry_geo.spark import SparseTransformer
    class DummyAdapter(nn.Module):
        def forward(self, x): return x

    model = YOLO("yolo11n-obb.pt") # use nano for fast test
    backbone = model.model.model[:11] # layers 0 to 10
    
    print("Converting to sparse...")
    backbone = replace_with_sparse_layers(backbone)
    backbone = patch_yolo_modules(backbone)
    
    print("Generating sparse input...")
    dummy_img = torch.randn(2, 3, 256, 256).cuda()
    backbone = backbone.cuda()
    
    transformer = SparseTransformer(DummyAdapter(), patch_size=32, mask_ratio=0.6).cuda()
    sparse_tensor, mask = transformer(dummy_img)
    
    print("Running forward pass...")
    try:
        # Note: YOLO backbone does not usually expect Sequential execution natively,
        # but layers 0 to 10 are sequential. Let's just run them one by one.
        x = sparse_tensor
        for i, m in enumerate(backbone):
            x = m(x)
            print(f"Layer {i} ({type(m).__name__}): output shape {x.features.shape} spatial {x.spatial_shape}")
        print("Success! Final Output type:", type(x))
    except Exception as e:
        import traceback
        traceback.print_exc()

