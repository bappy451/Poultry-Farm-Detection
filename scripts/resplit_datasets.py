import os
import shutil
import random
from pathlib import Path
import yaml

def resplit_dataset(dataset_name, source_img_dirs, target_dir, class_names):
    """
    Gathers all images and labels from source directories, shuffles, 
    and splits them 80/10/10 into train/val/test in the target directory.
    """
    print(f"\nResplitting dataset: {dataset_name}")
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Create target structure
    for split in ['train', 'val', 'test']:
        (target_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (target_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
    all_pairs = []
    
    for img_dir in source_img_dirs:
        img_dir = Path(img_dir)
        if not img_dir.exists():
            print(f"  Warning: Source dir {img_dir} does not exist. Skipping.")
            continue
            
        lbl_dir = img_dir.parent / 'labels'
        if not lbl_dir.exists():
            # Sometimes YOLO datasets have 'images' and 'labels' at the same level as train/val
            lbl_dir = img_dir.parents[1] / 'labels' / img_dir.name
            
        if not lbl_dir.exists():
            print(f"  Warning: Could not find labels for {img_dir}. Looked in {lbl_dir}")
            continue
            
        for img_path in img_dir.glob("*.*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            if lbl_path.exists():
                all_pairs.append((img_path, lbl_path))
                
    if not all_pairs:
        print(f"  No valid image-label pairs found for {dataset_name}. Skipping.")
        return
        
    print(f"  Found {len(all_pairs)} total image-label pairs.")
    
    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(all_pairs)
    
    total = len(all_pairs)
    n_train = int(total * 0.8)
    n_val = int(total * 0.1)
    
    splits = {
        'train': all_pairs[:n_train],
        'val': all_pairs[n_train:n_train+n_val],
        'test': all_pairs[n_train+n_val:]
    }
    
    for split_name, pairs in splits.items():
        print(f"  Copying {len(pairs)} files to {split_name} split...")
        for img_src, lbl_src in pairs:
            img_dst = target_dir / 'images' / split_name / img_src.name
            lbl_dst = target_dir / 'labels' / split_name / lbl_src.name
            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)
            
    # Generate YAML
    yaml_path = Path("configs") / f"{dataset_name}_80_10_10.yaml"
    yaml_content = {
        'path': str(target_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"  Created config: {yaml_path}")


if __name__ == "__main__":
    # 1. Poultry
    poultry_sources = [
        "temp/poultry_houses_train/train/images",
        "temp/poultry_houses_test/test/images"
    ]
    poultry_classes = {0: 'broiler breeder', 1: 'broiler house'}
    resplit_dataset("poultry", poultry_sources, "datasets/poultry_80_10_10", poultry_classes)
    
    # 2. GRSD
    grsd_sources = [
        "datasets/GRSD_YOLO/images/train",
        "datasets/GRSD_YOLO/images/val"
    ]
    grsd_classes = {0: 'plastic greenhouse'}
    resplit_dataset("grsd", grsd_sources, "datasets/grsd_80_10_10", grsd_classes)
    
    # 3. DOTA
    dota_sources = [
        "temp/dota_coco_format/dota/images/train",
        "temp/dota_coco_format/dota/images/val",
        "temp/dota_coco_format/dota/images/test"
    ]
    dota_classes = {
        0: 'plane', 1: 'ship', 2: 'storage-tank', 3: 'baseball-diamond', 4: 'tennis-court',
        5: 'basketball-court', 6: 'ground-track-field', 7: 'harbor', 8: 'bridge', 
        9: 'small-vehicle', 10: 'large-vehicle', 11: 'roundabout', 12: 'swimming-pool',
        13: 'helicopter', 14: 'soccer-ball-field', 15: 'container-crane'
    }
    resplit_dataset("dota", dota_sources, "datasets/dota_80_10_10", dota_classes)
