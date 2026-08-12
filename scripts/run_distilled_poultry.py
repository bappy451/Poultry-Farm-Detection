import os
import argparse
import torch
from ultralytics import YOLO

def extract_backbone(distill_ckpt_path):
    print(f"Loading distilled checkpoint from {distill_ckpt_path}")
    ckpt = torch.load(distill_ckpt_path, map_location='cpu')
    student_state = ckpt['model'] if 'model' in ckpt else ckpt
    
    # Extract only the backbone weights, strip "backbone." prefix
    backbone_state = {}
    for k, v in student_state.items():
        if k.startswith('backbone.'):
            backbone_state[k.replace('backbone.', '')] = v
            
    print(f"Extracted {len(backbone_state)} layers from the distilled backbone.")
    return backbone_state

def main():
    parser = argparse.ArgumentParser(description="Train YOLO on Poultry Dataset using Distilled Backbone")
    parser.add_argument("--distill-ckpt", type=str, required=True, help="Path to the distilled last.pt")
    parser.add_argument("--data", type=str, required=True, help="Path to poultry.yaml dataset config")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--model-cfg", type=str, default="yolo11n.yaml", help="YOLO architecture config (e.g. yolo11n.yaml)")
    parser.add_argument("--project", type=str, default="runs/poultry", help="Project output dir")
    parser.add_argument("--name", type=str, default="distilled_run", help="Run name")
    
    args = parser.parse_args()
    
    # 1. Initialize YOLO from scratch
    print(f"Initializing fresh YOLO model from {args.model_cfg}")
    model = YOLO(args.model_cfg)
    
    # 2. Load distilled backbone weights
    backbone_weights = extract_backbone(args.distill_ckpt)
    
    # YOLO's backbone is located in model.model (the nn.Module)
    # We load the weights strictly for the matching layers
    missing, unexpected = model.model.load_state_dict(backbone_weights, strict=False)
    
    print(f"Loaded distilled backbone weights. Missing (mostly neck/head): {len(missing)}, Unexpected: {len(unexpected)}")
    
    # 3. Train on Poultry Dataset
    print(f"Starting downstream training on {args.data}")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        pretrained=False # Crucial: Don't overwrite our distilled weights with COCO!
    )
    
    print("Training complete!")
    print(f"Results saved to {os.path.join(args.project, args.name)}")
    
if __name__ == "__main__":
    main()
