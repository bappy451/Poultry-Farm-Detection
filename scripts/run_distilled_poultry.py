import argparse
import os
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from poultry_geo.distill_models import (
    DISTILLATION_ARTIFACT_FORMAT,
    export_ultralytics_backbone_state,
    validate_backbone_compatibility,
)


def extract_backbone(distill_path: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load a v2 portable artifact or safely translate a legacy training checkpoint."""
    path = Path(distill_path)
    print(f"Loading distilled checkpoint from {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Distillation checkpoint must be a dictionary.")
    if checkpoint.get("format") == DISTILLATION_ARTIFACT_FORMAT:
        state = checkpoint.get("backbone_state_dict")
        if not isinstance(state, dict):
            raise ValueError("Distillation artifact has no backbone_state_dict.")
        config = {"student_model": checkpoint.get("student_model"), **checkpoint.get("metadata", {})}
        return state, config
    raise ValueError(
        "This is a legacy distillation checkpoint, not a v2 portable backbone artifact. "
        "It may have been trained with the old classifier-based student and is not a valid "
        "initialization for the COCO-versus-distillation experiment. Re-run train-distill and "
        "use yolo_backbone_distilled_best.pt."
    )


def load_distilled_backbone(model: YOLO, backbone_weights: dict[str, torch.Tensor]) -> int:
    """Load only after every expected detection-backbone tensor has been verified."""
    compatible = validate_backbone_compatibility(model.model.state_dict(), backbone_weights)
    missing, unexpected = model.model.load_state_dict(compatible, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys after validated backbone load: {unexpected}")
    print(f"Loaded all {len(compatible)} distilled backbone tensors; neck/head remain randomly initialized.")
    return len(compatible)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO detector from an exactly compatible distilled backbone")
    parser.add_argument("--distill-ckpt", required=True, help="Path to yolo_backbone_distilled_best.pt (preferred) or a legacy checkpoint")
    parser.add_argument("--data", required=True, help="Path to poultry.yaml dataset config")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model-cfg", default=None, help="Must match the distilled student YAML; inferred from v2 artifact when omitted")
    parser.add_argument("--project", default="runs/poultry")
    parser.add_argument("--name", default="distilled_run")
    args = parser.parse_args()

    backbone_weights, metadata = extract_backbone(args.distill_ckpt)
    artifact_model = metadata.get("student_model")
    model_cfg = args.model_cfg or artifact_model
    if not model_cfg:
        raise ValueError("Legacy checkpoint lacks student_model; pass --model-cfg explicitly.")
    if artifact_model and args.model_cfg and args.model_cfg != artifact_model:
        raise ValueError(f"--model-cfg={args.model_cfg} does not match distilled student_model={artifact_model}.")
    print(f"Initializing exactly compatible YOLO detector from {model_cfg}")
    model = YOLO(model_cfg)
    load_distilled_backbone(model, backbone_weights)
    print(f"Starting downstream training on {args.data}")
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project=args.project, name=args.name, pretrained=False)
    print(f"Training complete. Results saved to {os.path.join(args.project, args.name)}")


if __name__ == "__main__":
    main()
