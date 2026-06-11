import json
import os
import shutil
import random
from pathlib import Path

def convert_coco_to_yolo_labels(coco_data, images_info, labels_dir):
    labels_dir.mkdir(parents=True, exist_ok=True)
    cat_id_to_idx = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}
    
    img_annots = {}
    for annot in coco_data['annotations']:
        img_id = annot['image_id']
        img_annots.setdefault(img_id, []).append(annot)

    for img_id, img_info in images_info.items():
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        label_file = labels_dir / (Path(file_name).stem + '.txt')
        
        with open(label_file, 'w') as f:
            if img_id in img_annots:
                for annot in img_annots[img_id]:
                    cat_id = annot['category_id']
                    class_idx = cat_id_to_idx[cat_id]
                    
                    segmentation = annot['segmentation']
                    if not segmentation:
                        continue
                        
                    for poly in segmentation:
                        if len(poly) < 6:
                            continue
                        
                        normalized_poly = []
                        for i in range(0, len(poly), 2):
                            x = poly[i] / width
                            y = poly[i+1] / height
                            normalized_poly.extend([x, y])
                        
                        line = f"{class_idx} " + " ".join([f"{p:.6f}" for p in normalized_poly])
                        f.write(line + '\n')

def prepare_subset(source_dir, target_images_dir, target_labels_dir, coco_data, image_ids):
    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_labels_dir.mkdir(parents=True, exist_ok=True)
    
    images_info = {img['id']: img for img in coco_data['images'] if img['id'] in image_ids}
    
    # Move images
    for img_id, img_info in images_info.items():
        file_name = img_info['file_name']
        src_path = Path(source_dir) / file_name
        
        # Check if already in target (to avoid error if rerun)
        if (target_images_dir / file_name).exists():
            continue

        if not src_path.exists():
            # Check if it was already moved to a subdirectory 'images' in previous run
            src_path = Path(source_dir) / "images" / file_name
            
        if src_path.exists():
            shutil.move(str(src_path), str(target_images_dir / file_name))
        else:
            print(f"Warning: Image {file_name} not found in {source_dir} or {source_dir}/images")

    convert_coco_to_yolo_labels(coco_data, images_info, target_labels_dir)

def prepare_full_dataset(train_base_dir, test_base_dir, val_split=0.1, seed=42):
    random.seed(seed)
    train_base_dir = Path(train_base_dir)
    test_base_dir = Path(test_base_dir)
    
    # 1. Process Training and Validation
    train_src_dir = train_base_dir / "train"
    train_coco_path = train_src_dir / "_annotations.coco.json"
    
    if train_coco_path.exists():
        with open(train_coco_path, 'r') as f:
            train_coco = json.load(f)
        
        all_train_ids = [img['id'] for img in train_coco['images']]
        random.shuffle(all_train_ids)
        
        val_count = int(len(all_train_ids) * val_split)
        val_ids = set(all_train_ids[:val_count])
        train_ids = set(all_train_ids[val_count:])
        
        # Prepare Train
        prepare_subset(train_src_dir, train_base_dir / "train" / "images", 
                       train_base_dir / "train" / "labels", train_coco, train_ids)
        
        # Prepare Val
        prepare_subset(train_src_dir, train_base_dir / "val" / "images", 
                       train_base_dir / "val" / "labels", train_coco, val_ids)
        
        cat_names = [cat['name'] for cat in train_coco['categories']]
    else:
        print(f"Warning: Training annotations not found at {train_coco_path}")
        cat_names = []

    # 2. Process Test
    test_src_dir = test_base_dir / "test"
    test_coco_path = test_src_dir / "_annotations.coco.json"
    
    if test_coco_path.exists():
        with open(test_coco_path, 'r') as f:
            test_coco = json.load(f)
        
        test_ids = [img['id'] for img in test_coco['images']]
        prepare_subset(test_src_dir, test_base_dir / "test" / "images", 
                       test_base_dir / "test" / "labels", test_coco, set(test_ids))
    else:
        print(f"Warning: Test annotations not found at {test_coco_path}")

    # 3. Create data.yaml
    data_yaml_path = train_base_dir / "data.yaml"
    with open(data_yaml_path, 'w') as f:
        f.write(f"train: {(train_base_dir / 'train' / 'images').absolute()}\n")
        f.write(f"val: {(train_base_dir / 'val' / 'images').absolute()}\n")
        f.write(f"test: {(test_base_dir / 'test' / 'images').absolute()}\n")
        f.write(f"\nnc: {len(cat_names)}\n")
        f.write(f"names: {cat_names}\n")
    
    print(f"Full dataset prepared. data.yaml at {data_yaml_path}")

if __name__ == "__main__":
    # This is for backward compatibility or direct run if needed
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default="Data/poultry_houses_Training")
    parser.add_argument("--test-dir", type=str, default="Data/poultry_houses_test")
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()
    
    prepare_full_dataset(args.train_dir, args.test_dir, val_split=args.val_split)
