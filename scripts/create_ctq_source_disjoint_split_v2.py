"""Create a collision-safe source-raster-disjoint OBB train/val/test-v2 split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.create_ctq_source_disjoint_split import label_path, load_source_manifest, resolve, source_split, write_catalog


def destination_name(image: Path) -> str:
    """Preserve extension while guaranteeing split-local image/label uniqueness."""
    digest = hashlib.sha256(str(image.resolve()).encode("utf-8")).hexdigest()[:20]
    return f"{digest}{image.suffix.lower()}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-data", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--unlabeled-root", default=None)
    parser.add_argument("--unlabeled-catalog", default=None)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1 or not 0 < args.val_fraction < 1:
        raise ValueError("test-fraction and val-fraction must be in (0, 1).")
    data, manifest, output_root, output_data = (resolve(value) for value in (args.data, args.source_manifest, args.output_root, args.output_data))
    unlabeled = resolve(args.unlabeled_root) if args.unlabeled_root else None
    catalog = resolve(args.unlabeled_catalog) if args.unlabeled_catalog else None
    if not data.is_file() or not manifest.is_file():
        raise FileNotFoundError("Input data YAML or source manifest is missing.")
    if output_root.exists() or output_data.exists():
        raise FileExistsError("Output root or data YAML already exists; choose new targets.")
    if (unlabeled is None) != (catalog is None):
        raise ValueError("--unlabeled-root and --unlabeled-catalog must be provided together.")
    if unlabeled is not None and not unlabeled.is_dir():
        raise FileNotFoundError(unlabeled)
    from scripts.run_ema_benchmark import _dataset_image_paths
    images = _dataset_image_paths(data, ("train", "val"))
    sources = load_source_manifest(manifest)
    missing = [image for image in images if image not in sources]
    if missing:
        raise ValueError(f"Source manifest omits {len(missing)} labeled images; first={missing[:2]}")
    groups: dict[str, list[Path]] = {}
    for image in images:
        groups.setdefault(sources[image], []).append(image)
    test_sources, val_sources = source_split(groups, test_fraction=args.test_fraction, val_fraction=args.val_fraction, seed=args.seed)
    split_sources = {"train": set(groups) - test_sources - val_sources, "val": val_sources, "test": test_sources}
    if any(not value for value in split_sources.values()):
        raise RuntimeError("Source partition contains an empty split.")
    output_root.mkdir(parents=True)
    counts: dict[str, int] = {}
    test_rows: list[dict[str, str]] = []
    for split, allowed in split_sources.items():
        image_dir, label_dir = output_root / "images" / split, output_root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        split_images = [image for source in sorted(allowed) for image in sorted(groups[source])]
        names: set[str] = set()
        for image in split_images:
            name = destination_name(image)
            if name in names:
                raise RuntimeError(f"Destination collision for {image}")
            names.add(name)
            shutil.copy2(image, image_dir / name)
            shutil.copy2(label_path(image), label_dir / Path(name).with_suffix(".txt"))
            if split == "test":
                test_rows.append({"image_path": str((image_dir / name).resolve()), "source_raster": sources[image]})
        counts[split] = len(split_images)
    payload = yaml.safe_load(data.read_text(encoding="utf-8")) or {}
    output_data.parent.mkdir(parents=True, exist_ok=True)
    output_data.write_text(yaml.safe_dump({"path": str(output_root), "train": "images/train", "val": "images/val", "test": "images/test", "nc": int(payload.get("nc", 2)), "names": payload.get("names", {0: "broiler breeder", 1: "broiler house"})}, sort_keys=False), encoding="utf-8")
    test_manifest = output_root / "test_v2_source_manifest.csv"
    with test_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "source_raster"))
        writer.writeheader()
        writer.writerows(test_rows)
    catalog_count = write_catalog(unlabeled, test_sources, catalog) if unlabeled and catalog else None
    audit = {"created_utc": datetime.now(timezone.utc).isoformat(), "implementation": "create_ctq_source_disjoint_split_v2", "input_data": str(data), "source_manifest": str(manifest), "seed": args.seed, "counts": counts, "source_counts": {split: len(value) for split, value in split_sources.items()}, "test_sources": sorted(test_sources), "val_sources": sorted(val_sources), "source_overlap_train_val": [], "source_overlap_train_test": [], "source_overlap_val_test": [], "test_source_manifest": str(test_manifest), "unlabeled_catalog": str(catalog) if catalog else None, "unlabeled_catalog_images": catalog_count}
    (output_root / "source_disjoint_split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"data": str(output_data), "audit": str(output_root / "source_disjoint_split_audit.json"), **counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
