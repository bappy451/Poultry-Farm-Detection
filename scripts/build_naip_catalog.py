"""Build a unified catalog for the NAIP dataset at E:\\NAIP_Dataset_PH.

Rules
-----
* Include ALL supported imagery under  MS_tiles_512_256_jpg
* For satlas-dataset-v1-naip-2011/naip  → only tci subdirectories (exclude ir)
* For satlas-dataset-v1-naip-small/naip_small → only tci subdirectories (exclude ir)

Output: a single CSV with one column "path" containing absolute paths.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".npy"})

ROOT = Path(r"E:\NAIP_Dataset_PH")

# ── Sources ────────────────────────────────────────────────────────────
SOURCES = [
    # (directory, include_filter, exclude_filter)
    # MS tiles: include everything
    (ROOT / "MS_tiles_512_256_jpg", None, None),
    # Satlas 2011: only tci dirs
    (ROOT / "satlas-dataset-v1-naip-2011" / "naip", "tci", "ir"),
    # Satlas small: only tci dirs
    (ROOT / "satlas-dataset-v1-naip-small" / "naip_small", "tci", "ir"),
]

OUTPUT_PATH = Path(r"C:\Users\mza0288\Documents\Poultry-Farm-Detection\trained_models\data_catalog.csv")


def collect_files(base_dir: Path, include_subdir: str | None, exclude_subdir: str | None) -> list[str]:
    """Collect all supported image files from base_dir, with optional subdir filtering."""
    files: list[str] = []
    
    if not base_dir.exists():
        print(f"  [WARNING] Directory does not exist, skipping: {base_dir}")
        return files
    
    count = 0
    for root_str, dirs, filenames in os.walk(str(base_dir)):
        # If we have an exclude filter, skip directories named that
        if exclude_subdir:
            # Modify dirs in-place to prevent os.walk from descending into excluded dirs
            dirs[:] = [d for d in dirs if d.lower() != exclude_subdir.lower()]
        
        # If we have an include filter, only process files that are inside matching subdirs
        if include_subdir:
            # Check if any part of the current path contains the include subdir
            parts = Path(root_str).parts
            if include_subdir.lower() not in [p.lower() for p in parts]:
                continue
        
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root_str, fname)
                files.append(full_path)
                count += 1
                if count % 50000 == 0:
                    print(f"    ... found {count:,} files so far in {base_dir.name}")
    
    return files


def main() -> None:
    all_files: list[str] = []
    
    for base_dir, include_subdir, exclude_subdir in SOURCES:
        print(f"Scanning: {base_dir}")
        found = collect_files(base_dir, include_subdir, exclude_subdir)
        print(f"  Found: {len(found):,} files")
        all_files.extend(found)
    
    if not all_files:
        print("ERROR: No files found! Check your paths.")
        sys.exit(1)
    
    all_files.sort()
    
    # Remove duplicates (shouldn't happen, but safety)
    unique = list(dict.fromkeys(all_files))
    if len(unique) != len(all_files):
        print(f"  Removed {len(all_files) - len(unique)} duplicates")
        all_files = unique
    
    print(f"\nTotal files: {len(all_files):,}")
    
    # Quick sanity: show breakdown by source
    ms_count = sum(1 for f in all_files if "MS_tiles" in f)
    satlas_2011_count = sum(1 for f in all_files if "naip-2011" in f)
    satlas_small_count = sum(1 for f in all_files if "naip-small" in f)
    ir_count = sum(1 for f in all_files if os.sep + "ir" + os.sep in f or "/ir/" in f)
    
    print(f"  MS_tiles_512_256_jpg : {ms_count:,}")
    print(f"  satlas-naip-2011/tci : {satlas_2011_count:,}")
    print(f"  satlas-naip-small/tci: {satlas_small_count:,}")
    print(f"  IR files (should be 0): {ir_count:,}")
    
    if ir_count > 0:
        print("ERROR: IR files were included! Check filtering logic.")
        sys.exit(1)
    
    # Save catalog
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"path": all_files})
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nCatalog saved to: {OUTPUT_PATH}")
    print(f"Total entries: {len(df):,}")


if __name__ == "__main__":
    main()
