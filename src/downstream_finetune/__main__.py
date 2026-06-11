import argparse
from pathlib import Path
from .prepare_data import prepare_full_dataset
from .finetune import finetune_yolo, finetune_with_config

def main():
    parser = argparse.ArgumentParser(description="Downstream Fine-tuning for Poultry House Detection")
    parser.add_argument("--config", type=str, help="Path to YAML config file for training")
    parser.add_argument("--train-dir", type=str, default="Data/poultry_houses_Training", help="Directory containing the training dataset")
    parser.add_argument("--test-dir", type=str, default="Data/poultry_houses_test", help="Directory containing the test dataset")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of training data to use for validation")
    parser.add_argument("--model-path", type=str, default="Data/yolo26l-seg.pt", help="Path to the pretrained YOLO model")
    parser.add_argument("--output-dir", type=str, default="runs/downstream", help="Directory to save results")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare the dataset, don't start training")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip data preparation")
    
    args = parser.parse_args()

    if args.config:
        print(f"Starting config-based training with {args.config}...")
        finetune_with_config(args.config)
        return

    train_dir = Path(args.train_dir)
    data_yaml = train_dir / "data.yaml"

    # 1. Prepare data if forced or if data.yaml doesn't exist
    if not args.skip_prepare:
        print("Preparing dataset...")
        prepare_full_dataset(
            train_base_dir=args.train_dir,
            test_base_dir=args.test_dir,
            val_split=args.val_split
        )
    
    if args.prepare_only:
        return

    # 2. Start fine-tuning (legacy path)
    finetune_yolo(
        model_path=args.model_path,
        data_yaml_path=data_yaml,
        output_dir=args.output_dir,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )

if __name__ == "__main__":
    main()
