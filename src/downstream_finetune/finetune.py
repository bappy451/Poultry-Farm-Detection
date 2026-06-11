import os
from pathlib import Path
from ultralytics import YOLO

def finetune_yolo(model_path, data_yaml_path, output_dir, epochs=50, imgsz=640, batch=16):
    """
    Fine-tunes a YOLO segmentation model.
    """
    # Load the pretrained model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # Run training
    print(f"Starting fine-tuning on {data_yaml_path}...")
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(output_dir),
        name="finetune_yolo26l",
        save=True,
        # device="auto" can be problematic on some systems
    )
    
    print(f"Fine-tuning complete. Results saved to {output_dir}")

    # Run evaluation on the test set
    print("Running evaluation on the test set...")
    metrics = model.val(split='test')
    print(f"Test evaluation complete. Metrics: {metrics}")

    return results, metrics

if __name__ == "__main__":
    # Paths based on user instructions
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODEL_PATH = PROJECT_ROOT / "Data" / "yolo26l-seg.pt"
    DATA_YAML = PROJECT_ROOT / "Data" / "poultry_houses_Training" / "data.yaml"
    OUTPUT_DIR = PROJECT_ROOT / "runs" / "downstream"

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
    elif not DATA_YAML.exists():
        print(f"Error: Data YAML not found at {DATA_YAML}. Run prepare_data.py first.")
    else:
        finetune_yolo(MODEL_PATH, DATA_YAML, OUTPUT_DIR)
