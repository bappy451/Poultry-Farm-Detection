import torch
import torch.nn as nn

class YOLODenseEncoder(nn.Module):
    """
    Standard dense YOLO backbone extractor for distillation.
    Extracts the multi-scale feature maps from the backbone.
    """
    def __init__(self, backbone_type="yolo11l"):
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package is required to use YOLO backbones.")
            
        print(f"Instantiating dense {backbone_type} backbone from Ultralytics for distillation...")
        yolo = YOLO(f"{backbone_type}-cls.pt")
        
        # YOLO backbone breakdown
        # 0: Conv (s=2)
        # 1: Conv (s=2), 2: C2f/C3 (s=4)
        # 3: Conv (s=2), 4: C2f/C3 (s=8)
        # 5: Conv (s=2), 6: C2f/C3 (s=16) -> P4
        # 7: Conv (s=2), 8: C2f/C3, 9: SPPF (s=32) -> P5
        
        self.stem = yolo.model.model[0]
        self.stage1 = nn.Sequential(*yolo.model.model[1:3])
        self.stage2 = nn.Sequential(*yolo.model.model[3:5])
        self.stage3 = nn.Sequential(*yolo.model.model[5:7])
        self.stage4 = nn.Sequential(*yolo.model.model[7:10])

        # Dynamically determine the channels of P4 (stage3) for the projector
        dummy_x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            x = self.stem(dummy_x)
            x1 = self.stage1(x)
            x2 = self.stage2(x1)
            x3 = self.stage3(x2)
            self.p4_channels = x3.shape[1]

    def forward(self, x):
        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return x1, x2, x3, x4

class DistillationProjector(nn.Module):
    """
    Projects the CNN feature map to the Vision Transformer embedding dimension
    and flattens the spatial grid to match the ViT's patch token format [B, N, D].
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, H/16, W/16]
        x = self.proj(x) # -> [B, D, H/16, W/16]
        B, D, H, W = x.shape
        # Flatten and transpose: [B, D, H*W] -> [B, H*W, D]
        x = x.view(B, D, -1).transpose(1, 2).contiguous()
        return x

class DistillationStudent(nn.Module):
    """
    Combined module containing the YOLO encoder and the distillation projector.
    """
    def __init__(self, backbone_type="yolo11l", vit_embed_dim=768):
        super().__init__()
        self.encoder = YOLODenseEncoder(backbone_type=backbone_type)
        self.projector = DistillationProjector(
            in_channels=self.encoder.p4_channels, 
            out_channels=vit_embed_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through YOLO
        _, _, p4, _ = self.encoder(x)
        # Project P4 (stride 16) to match ViT patch tokens
        projected_tokens = self.projector(p4)
        return projected_tokens
