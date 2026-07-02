import os
import csv
import time
import argparse

def build_catalog(root_dir, output_csv):
    print(f"Scanning {root_dir} for images...")
    print("This may take several minutes for ~10 million files...")
    start = time.time()
    
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path'])
        
        count = 0
        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in valid_exts:
                    # Writing absolute path to the catalog
                    full_path = os.path.join(dirpath, name)
                    writer.writerow([full_path])
                    count += 1
                    
                    if count % 1000000 == 0:
                        print(f"  ... found {count:,} images so far ...")
                        
    elapsed = time.time() - start
    print(f"\nDone! Found {count:,} images in {elapsed:.2f} seconds.")
    print(f"Catalog saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=r"E:\NAIP_Dataset_PH", help="Root directory of images")
    parser.add_argument("--output-csv", default=r"trained_models\data_catalog.csv", help="Output catalog path")
    args = parser.parse_args()
    
    build_catalog(args.image_dir, args.output_csv)
