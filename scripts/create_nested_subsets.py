import os
import shutil
import random
from pathlib import Path
import yaml

def create_nested_subsets(source_yaml, fractions=[0.50, 0.25, 0.10, 0.05, 0.01]):
    """
    Creates perfectly nested training subsets for a YOLO dataset.
    The 1% subset is contained entirely within the 5% subset, etc.
    Validation and Test sets are kept identical.
    """
    source_yaml = Path(source_yaml)
    if not source_yaml.exists():
        print(f"Error: {source_yaml} not found.")
        return

    with open(source_yaml, 'r') as f:
        config = yaml.safe_load(f)

    base_dir = Path(config['path'])
    train_images_dir = base_dir / config['train']
    train_labels_dir = train_images_dir.parent.parent / 'labels' / train_images_dir.name
    
    # Get all training images
    all_train_imgs = sorted(list(train_images_dir.glob("*.*")))
    valid_imgs = [p for p in all_train_imgs if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    # Check corresponding labels
    pairs = []
    for img_path in valid_imgs:
        lbl_path = train_labels_dir / (img_path.stem + '.txt')
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))

    if not pairs:
        print(f"No valid image-label pairs found in {train_images_dir}")
        return

    print(f"\nProcessing {source_yaml.name}")
    print(f"Found {len(pairs)} total training samples.")

    # Shuffle once with a fixed seed. This ensures perfect nesting.
    random.seed(42)
    random.shuffle(pairs)

    total = len(pairs)
    dataset_name = source_yaml.stem.split('_')[0]
    out_dir_base = Path("datasets") / f"{dataset_name}_nested"

    for frac in sorted(fractions, reverse=True): # Start from largest (50%) to smallest (1%)
        n_samples = int(total * frac)
        subset_pairs = pairs[:n_samples]
        
        subset_name = f"{int(frac*100)}percent"
        target_dir = out_dir_base / subset_name
        print(f"  Creating {subset_name} subset with {n_samples} samples -> {target_dir}")
        
        # Create directories
        (target_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (target_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        
        # Copy Train pairs
        for img_src, lbl_src in subset_pairs:
            img_dst = target_dir / 'images' / 'train' / img_src.name
            lbl_dst = target_dir / 'labels' / 'train' / lbl_src.name
            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)
            
        # Copy Val/Test directories exactly as they are
        for split in ['val', 'test']:
            if split in config:
                split_src_img = base_dir / config[split]
                split_src_lbl = split_src_img.parent.parent / 'labels' / split_src_img.name
                
                if split_src_img.exists():
                    shutil.copytree(split_src_img, target_dir / 'images' / split, dirs_exist_ok=True)
                if split_src_lbl.exists():
                    shutil.copytree(split_src_lbl, target_dir / 'labels' / split, dirs_exist_ok=True)

        # Generate new YAML
        yaml_path = Path("configs") / f"{dataset_name}_nested_{subset_name}.yaml"
        new_config = config.copy()
        new_config['path'] = str(target_dir.absolute())
        
        with open(yaml_path, 'w') as f:
            yaml.dump(new_config, f, sort_keys=False)
        print(f"    Config saved to {yaml_path}")

if __name__ == "__main__":
    # Create nested subsets for both Poultry and GRSD
    create_nested_subsets("configs/poultry_80_10_10.yaml", fractions=[0.50, 0.25, 0.10, 0.05, 0.01])
    create_nested_subsets("configs/grsd_80_10_10.yaml", fractions=[0.50, 0.25, 0.10, 0.05, 0.01])
    print("\nAll nested subsets created successfully!")
