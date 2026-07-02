import csv
import random
import argparse
import os

def create_subset(input_csv, output_csv, fraction=0.10, seed=42):
    print(f"Reading full catalog from {input_csv}...")
    random.seed(seed)
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # We can't load 10 million rows into memory without some overhead, 
        # but 10 million strings is only ~1-2 GB in Python, which is perfectly fine.
        rows = list(reader)
        
    total_rows = len(rows)
    subset_size = int(total_rows * fraction)
    print(f"Total images found: {total_rows:,}")
    print(f"Sampling {fraction*100}% of images -> {subset_size:,} images...")
    
    subset = random.sample(rows, subset_size)
    
    print(f"Writing subset to {output_csv}...")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(subset)
        
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=r"trained_models\data_catalog.csv")
    parser.add_argument("--output-csv", default=r"trained_models\data_catalog_10percent.csv")
    parser.add_argument("--fraction", type=float, default=0.10)
    args = parser.parse_args()
    
    create_subset(args.input_csv, args.output_csv, args.fraction)
