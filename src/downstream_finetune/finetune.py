import os
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

def load_spark_backbone_into_yolo(yolo_model, spark_backbone_path):
    """
    Loads SparK pretrained backbone weights into a YOLO model.
    """
    print(f"Loading SparK backbone from {spark_backbone_path}...")
    spark_state = torch.load(spark_backbone_path, map_location='cpu')
    
    # If it's a full SparK checkpoint, extract the encoder
    if 'model_state_dict' in spark_state:
        spark_state = {k.replace('encoder.', ''): v for k, v in spark_state['model_state_dict'].items() if k.startswith('encoder.')}
    
    yolo_state = yolo_model.model.state_dict()
    
    # Mapping for YOLOv8/v11 backbones
    # SparK simplified CSPDarkNet mapping
    mapping = {
        "stem.weight": "model.0.conv.weight",
        "stem.ln.weight": "model.0.bn.weight",
        "stem.ln.bias": "model.0.bn.bias",
        
        "stage1.0.weight": "model.1.conv.weight",
        "stage1.1.ln.weight": "model.1.bn.weight",
        "stage1.1.ln.bias": "model.1.bn.bias",
        
        "stage2.0.weight": "model.2.conv.weight", # Note: YOLO might have different indices
        "stage2.1.ln.weight": "model.2.bn.weight",
        "stage2.1.ln.bias": "model.2.bn.bias",
    }
    
    # Dynamic mapping based on the SparK encoder structure
    translated_state = {}
    for k, v in spark_state.items():
        # Try to find a match in mapping
        found = False
        for sk, yk in mapping.items():
            if k == sk:
                if yk in yolo_state:
                    if yolo_state[yk].shape == v.shape:
                        translated_state[yk] = v
                        found = True
                    else:
                        print(f"Warning: Shape mismatch for {yk} ({k}): {v.shape} vs {yolo_state[yk].shape}. Skipping.")
                break
        
        if not found:
            # Fallback for other potential mappings or direct matches
            if k in yolo_state:
                if yolo_state[k].shape == v.shape:
                    translated_state[k] = v
                else:
                    # Only print mismatch for layers that were likely intended to match
                    if 'weight' in k or 'bias' in k:
                        pass # Silently skip other mismatches unless explicitly mapped

    msg = yolo_model.model.load_state_dict(translated_state, strict=False)
    print(f"SparK weights loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
    return yolo_model

def finetune_with_config(config_path):
    """
    Fine-tunes YOLO models based on a YAML configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_yaml = config['data']['path']
    models_to_train = config['models']
    training_params = config.get('training', {})
    
    results_all = []
    metrics_all = []
    
    output_base = Path(training_params.get('project', 'runs/downstream'))
    output_base.mkdir(parents=True, exist_ok=True)

    for m_cfg in models_to_train:
        model_path = m_cfg['path']
        model_name = m_cfg.get('name', Path(model_path).stem)
        is_spark = m_cfg.get('is_spark', False)
        base_yolo = m_cfg.get('base_yolo', 'yolo11l-seg.pt')

        print(f"\n--- Training Model: {model_name} ---")
        
        if is_spark:
            # Load a base YOLO model first, then inject SparK weights
            model = YOLO(base_yolo)
            model = load_spark_backbone_into_yolo(model, model_path)
        else:
            model = YOLO(model_path)

        # Run training
        results = model.train(
            data=str(data_yaml),
            epochs=training_params.get('epochs', 50),
            imgsz=training_params.get('imgsz', 640),
            batch=training_params.get('batch', 16),
            project=str(output_base),
            name=model_name,
            save=True,
            **training_params.get('extra_args', {})
        )
        
        # Run evaluation
        print(f"Evaluating {model_name}...")
        metrics = model.val(split='test')
        
        results_all.append(results)
        metrics_all.append(metrics)
        
    return results_all, metrics_all

def finetune_yolo(model_path, data_yaml_path, output_dir, epochs=50, imgsz=640, batch=16):
    """
    Legacy function for backward compatibility.
    """
    model = YOLO(model_path)
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(output_dir),
        name=Path(model_path).stem,
        save=True,
    )
    metrics = model.val(split='test')
    return results, metrics

if __name__ == "__main__":
    # For quick testing
    pass
