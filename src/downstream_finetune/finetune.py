import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

from .finetune_resnet50 import train_resnet50_segmentation


def load_spark_backbone_into_yolo(yolo_model, spark_backbone_path):
    """Load only a verified, architecture-identical SparK-to-YOLO export.

    The current supported SparK encoder is ResNet50, so it cannot be injected
    into a YOLO graph.  Refuse the old partial `strict=False` mapping rather
    than silently training a pretrained YOLO baseline under a SparK label.
    """
    if not spark_backbone_path:
        raise ValueError("A SparK checkpoint path is required.")
    path = Path(spark_backbone_path)
    if not path.is_file():
        raise FileNotFoundError(f"SparK checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    backbone_type = checkpoint.get("backbone_type") if isinstance(checkpoint, dict) else None
    if backbone_type != "yolo11l":
        raise ValueError(
            "This project has no verified SparK-to-YOLO adapter. The supported SparK "
            f"artifact is {backbone_type or 'unknown'}, which cannot be injected into YOLO."
        )
    raise NotImplementedError(
        "Sparse YOLO SparK exports are disabled until an architecture-pinned adapter "
        "proves nonzero, expected-module weight coverage."
    )


def finetune_with_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_yaml = config["data"]["path"]
    training_params = config.get("training", {})
    output_base = Path(training_params.get("project", "runs/downstream"))
    output_base.mkdir(parents=True, exist_ok=True)
    results_all, metrics_all = [], []

    for model_config in config["models"]:
        model_path = model_config.get("pretrained_weights", model_config.get("path"))
        model_name = model_config.get("name", Path(model_path).stem if model_path else "model")
        model_type = model_config.get("type", "yolo")
        is_spark, is_random = model_config.get("is_spark", False), model_config.get("is_random", False)
        if model_type == "resnet50":
            train_resnet50_segmentation(
                data_yaml=str(data_yaml), model_weights=model_path, output_dir=str(output_base / model_name),
                epochs=training_params.get("epochs", 50), batch_size=training_params.get("batch", 16),
                imgsz=training_params.get("imgsz", 640), pretrained=model_config.get("pretrained", False),
            )
            continue
        base_yolo = model_config.get("base_yolo", "yolo11l-seg.pt")
        if is_spark:
            model = load_spark_backbone_into_yolo(YOLO(base_yolo), model_path)
        elif is_random:
            model = YOLO(base_yolo.replace(".pt", ".yaml"))
        else:
            if not model_path:
                raise ValueError(f"Non-random model {model_name!r} requires pretrained_weights/path.")
            model = YOLO(model_path)
        results = model.train(data=str(data_yaml), epochs=training_params.get("epochs", 50),
                              imgsz=training_params.get("imgsz", 640), batch=training_params.get("batch", 16),
                              project=str(output_base), name=model_name, save=True,
                              **training_params.get("extra_args", {}))
        results_all.append(results)
        metrics_all.append(model.val(split="test"))
    return results_all, metrics_all


def finetune_yolo(model_path, data_yaml_path, output_dir, epochs=50, imgsz=640, batch=16):
    model = YOLO(model_path)
    results = model.train(data=str(data_yaml_path), epochs=epochs, imgsz=imgsz, batch=batch,
                          project=str(output_dir), name=Path(model_path).stem, save=True)
    return results, model.val(split="test")
