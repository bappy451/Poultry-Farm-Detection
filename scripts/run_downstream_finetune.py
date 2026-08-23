import os
import argparse
import yaml
import logging
from pathlib import Path

from downstream_finetune.prepare_data import prepare_full_dataset
from downstream_finetune.finetune_vit import train_vit_segmentation
from downstream_finetune.finetune import finetune_with_config, load_spark_backbone_into_yolo
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Launcher")

def ensure_dataset(config):
    train_dir = Path(config['dataset']['train_dir'])
    test_dir = Path(config['dataset']['test_dir'])
    val_split = config['dataset'].get('val_split', 0.1)

    data_yaml = train_dir / "data.yaml"
    if not data_yaml.exists():
        logger.info(f"data.yaml not found at {data_yaml}. Preparing dataset...")
        prepare_full_dataset(str(train_dir), str(test_dir), val_split=val_split)
    else:
        logger.info(f"Dataset already prepared. Found {data_yaml}")

    return str(data_yaml)

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_yaml_path = ensure_dataset(config)

    training_params = config.get('training', {})
    epochs = training_params.get('epochs', 50)
    batch_size = training_params.get('batch_size', 16)
    num_workers = training_params.get('num_workers', 8)
    imgsz = training_params.get('imgsz', 640)
    project_dir = Path(training_params.get('project_dir', 'runs/downstream'))

    models = config.get('models', [])

    for m_cfg in models:
        model_name = m_cfg.get('name', 'model')
        model_type = m_cfg.get('type')
        logger.info(f"\n======================================")
        logger.info(f"Starting Training for: {model_name} ({model_type})")
        logger.info(f"======================================")

        output_dir = project_dir / model_name

        if model_type == 'yolo':
            base_yolo = m_cfg.get('base_yolo', 'yolo11l-seg.pt')
            weights = m_cfg.get('pretrained_weights')
            is_spark = m_cfg.get('is_spark', False)
            is_random = m_cfg.get('is_random', False)

            if is_random:
                # Load architecture without pretrained weights
                # Requires a .yaml configuration for the model, e.g., 'yolo11l-seg.yaml'
                model_yaml = base_yolo.replace('.pt', '.yaml')
                logger.info(f"Loading random weights from {model_yaml}")
                model = YOLO(model_yaml)
            else:
                # Load with default COCO/ImageNet pretrained weights
                logger.info(f"Loading pretrained base from {base_yolo}")
                model = YOLO(base_yolo)

                if is_spark:
                    if not weights:
                        raise ValueError(f"SparK model {model_name!r} requires pretrained_weights.")
                    model = load_spark_backbone_into_yolo(model, weights)

            model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                project=str(project_dir),
                name=model_name,
                workers=num_workers,
                save=True
            )
            logger.info(f"Evaluating YOLO {model_name}...")
            model.val(split='test')

        elif model_type == 'vit':
            weights = m_cfg.get('pretrained_weights')
            is_random = m_cfg.get('is_random', False)
            pretrained = m_cfg.get('pretrained', False) # True for ImageNet

            # If is_random is True, we don't pass weights and don't set pretrained=True
            if is_random:
                weights = None
                pretrained = False

            train_vit_segmentation(
                data_yaml=data_yaml_path,
                model_weights=weights,
                output_dir=str(output_dir),
                epochs=epochs,
                batch_size=batch_size,
                imgsz=imgsz,
                embed_dim=m_cfg.get('embed_dim', 384),
                depth=m_cfg.get('depth', 12),
                num_heads=m_cfg.get('num_heads', 6),
                patch_size=m_cfg.get('patch_size', 16),
                pretrained=pretrained,
                num_workers=num_workers
            )

        else:
            logger.error(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/downstream_finetune.yaml", help="Path to downstream training config")
    args = parser.parse_args()
    main(args.config)
