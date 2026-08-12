import os
import torch
import torch.nn as nn
from typing import Dict, Any

from poultry_geo.distill_models import DistillationStudent
from poultry_geo.distill_loss import FeatureDistillationLoss
from poultry_geo.ssl_models import RemoteSensingViT

class DistillConfig:
    def __init__(self, precision='bf16', backbone_type='yolo11l', lr=1e-3, batch_size=32, vit_name='B'):
        self.precision = precision
        self.backbone_type = backbone_type
        self.lr = lr
        self.batch_size = batch_size
        self.vit_name = vit_name
        self.vit_embed_dim = 768 if vit_name == 'B' else (384 if vit_name == 'S' else 1024)

def load_teacher(checkpoint_path: str, config: DistillConfig, device: torch.device) -> nn.Module:
    print(f"Loading DINO Teacher ViT-{config.vit_name} from {checkpoint_path}...")
    teacher = RemoteSensingViT(
        image_size=224,
        patch_size=16,
        embed_dim=config.vit_embed_dim,
        num_heads=12 if config.vit_name == 'B' else (6 if config.vit_name == 'S' else 16),
        depth=12,
        precision=config.precision
    )
    
    if checkpoint_path.lower() == 'official':
        model_name = f'dino_vit{config.vit_name.lower()}16'
        print(f"Downloading official HuggingFace/TorchHub weights: {model_name}")
        hub_model = torch.hub.load('facebookresearch/dino:main', model_name)
        state_dict = hub_model.state_dict()
        
        # Remove CLS token and its positional embedding to match RemoteSensingViT (dense patches only)
        if 'pos_embed' in state_dict and state_dict['pos_embed'].shape[1] == 197:
            state_dict['pos_embed'] = state_dict['pos_embed'][:, 1:, :] # drop cls token pos embed
        if 'cls_token' in state_dict:
            del state_dict['cls_token']
            
        teacher.load_state_dict(state_dict, strict=False)
        print("Official teacher weights loaded successfully.")
    elif os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['teacher'] if 'teacher' in ckpt else ckpt
        # The teacher in DINOMultiCropWrapper might have 'backbone.' prefix
        backbone_state = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
        if backbone_state:
            teacher.load_state_dict(backbone_state, strict=True)
        else:
            teacher.load_state_dict(state_dict, strict=False)
        print("Teacher weights loaded successfully.")
    else:
        print(f"WARNING: Teacher checkpoint '{checkpoint_path}' not found! Using randomly initialized teacher.")
        
    teacher.to(device)
    teacher.eval()
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    for param in teacher.parameters():
        param.requires_grad = False
    return teacher

def train_one_step(student, teacher, criterion, optimizer, scaler, images, config):
    student.train()
    optimizer.zero_grad(set_to_none=True)
    
    dtype = torch.float32
    if config.precision == 'fp16': dtype = torch.float16
    elif config.precision == 'bf16': dtype = torch.bfloat16
    
    # Use AMP for speedup
    with torch.autocast(device_type=images.device.type, enabled=(config.precision != 'fp32'), dtype=dtype):
        # 1. Teacher feature extraction
        with torch.no_grad():
            teacher_tokens = teacher.forward_features(images)
            
        # 2. Student projection
        student_tokens = student(images)
        
        # 3. Distillation Loss (Cosine Similarity)
        loss = criterion(student_tokens, teacher_tokens)
        
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
        
    return loss.item()
