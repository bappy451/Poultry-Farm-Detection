"""Create a source-raster-disjoint train/validation/test-v2 OBB split.

The input CSV is authoritative provenance: each labeled image must map to one
real source-raster ID.  The resulting test-v2 sources are excluded from the
optional unlabeled catalog, preventing transductive exposure during CTQ.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    lowered = [part.lower() for part in parts]
    if "images" not in lowered:
        raise ValueError(f"Cannot infer labels/ location from image path: {image}")
    index = len(lowered) - 1 - lowered[::-1].index("images")
    parts[index] = "labels"
    candidate = Path(*parts).with_suffix(".txt")
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing label for {image}: {candidate}")
    return candidate


def load_source_manifest(path: Path) -> dict[Path, str]:
    mapping: dict[Path, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "source_raster"}
        if reader.fieldnames is None or required - set(reader.fieldnames):
            raise ValueError(f"Source manifest must have columns {sorted(required)}")
        for line, row in enumerate(reader, start=2):
            image = Path(str(row["image_path"]).strip()).expanduser().resolve()
            source = str(row["source_raster"]).strip()
            if not image.is_file() or not source or source.lower() in {"unknown", "__unknown__"}:
                raise ValueError(f"Invalid source-manifest row {line}")
            old = mapping.setdefault(image, source)
            if old != source:
                raise ValueError(f"Conflicting source-raster mapping for {image}")
    return mapping


def source_split(groups: dict[str, list[Path]], *, test_fraction: float, val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    """Choose source groups by image count, keeping all source tiles together."""
    if len(groups) < 3:
        raise ValueError("Source-disjoint split needs at least three distinct source_raster groups.")
    names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(names)
    total = sum(len(group) for group in groups.values())

    def choose(available: list[str], target: int) -> set[str]:
        chosen: set[str] = set()
        count = 0
        for name in available:
            before = abs(count - target)
            after = abs(count + len(groups[name]) - target)
            if not chosen or after <= before:
                chosen.add(name)
                count += len(groups[name])
        if not chosen or len(chosen) == len(available):
            raise ValueError("Cannot form a non-empty source-level partition with requested fractions.")
        return chosen

    test = choose(names, max(1, round(total * test_fraction)))
    remaining = [name for name in names if name not in test]
    val = choose(remaining, max(1, round(sum(len(groups[name]) for name in remaining) * val_fraction)))
    if test & val or not (set(groups) - test - val):
        raise RuntimeError("Source split failed to produce three disjoint non-empty partitions.")
    return test, val


def iter_unlabeled(root: Path) -> Iterable[tuple[Path, str]]:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = path.relative_to(root)
        yield path, rel.parts[0] if len(rel.parts) > 1 else "__root__"


def write_catalog(root: Path, excluded_sources: set[str], output: Path) -> int:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite catalog: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "source_raster"))
        writer.writeheader()
        for path, source in iter_unlabeled(root):
            if source in excluded_sources:
                continue
            writer.writerow({"path": str(path.resolve()), "source_raster": source})
            count += 1
    if count == 0:
        raise RuntimeError("Test-source exclusion left an empty unlabeled catalog.")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Labeled data YAML; only train and val are used as input.")
    parser.add_argument("--source-manifest", required=True, help="CSV: image_path,source_raster for every input labeled image.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-data", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--val-fraction", type=float, default=0.10, help="Fraction of non-test images assigned to validation.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--unlabeled-root", default=None)
    parser.add_argument("--unlabeled-catalog", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.test_fraction < 1.0 or not 0.0 < args.val_fraction < 1.0:
        raise ValueError("test-fraction and val-fraction must be in (0, 1).")
    data = resolve(args.data)
    source_manifest = resolve(args.source_manifest)
    output_root = resolve(args.output_root)
    output_data = resolve(args.output_data)
    unlabeled_root = resolve(args.unlabeled_root) if args.unlabeled_root else None
    catalog = resolve(args.unlabeled_catalog) if args.unlabeled_catalog else None
    if not data.is_file() or not source_manifest.is_file():
        raise FileNotFoundError("Input data YAML or source manifest is missing.")
    if output_root.exists() or output_data.exists():
        raise FileExistsError("Output root or data YAML already exists; choose new targets.")
    if (unlabeled_root is None) != (catalog is None):
        raise ValueError("--unlabeled-root and --unlabeled-catalog must be provided together.")
    if unlabeled_root is not None and not unlabeled_root.is_dir():
        raise FileNotFoundError(unlabeled_root)

    from scripts.run_ema_benchmark import _dataset_image_paths

    images = _dataset_image_paths(data, ("train", "val"))
    sources = load_source_manifest(source_manifest)
    missing = [path for path in images if path not in sources]
    if missing:
        raise ValueError(f"Source manifest omits {len(missing)} labeled images; first={missing[:2]}")
    groups: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        groups[sources[image]].append(image)
    test_sources, val_sources = source_split(groups, test_fraction=args.test_fraction, val_fraction=args.val_fraction, seed=args.seed)
    split_sources = {
        "train": set(groups) - test_sources - val_sources,
        "val": val_sources,
        "test": test_sources,
    }
    output_root.mkdir(parents=True)
    counts: dict[str, int] = {}
    test_rows: list[dict[str, str]] = []
    for split, allowed_sources in split_sources.items():
        image_dir, label_dir = output_root / "images" / split, output_root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        split_images = [image for source in sorted(allowed_sources) for image in sorted(groups[source])]
        counts[split] = len(split_images)
        for image in split_images:
            label = label_path(image)
            shutil.copy2(image, image_dir / image.name)
            shutil.copy2(label, label_dir / label.name)
            if split == "test":
                test_rows.append({"image_path": str((image_dir / image.name).resolve()), "source_raster": sources[image]})
    if any(value == 0 for value in counts.values()):
        raise RuntimeError(f"Empty split created: {counts}")
    payload = yaml.safe_load(data.read_text(encoding="utf-8")) or {}
    output_data.parent.mkdir(parents=True, exist_ok=True)
    output_data.write_text(yaml.safe_dump({
        "path": str(output_root), "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": int(payload.get("nc", 2)), "names": payload.get("names", {0: "broiler breeder", 1: "broiler house"}),
    }, sort_keys=False), encoding="utf-8")
    test_manifest = output_root / "test_v2_source_manifest.csv"
    with test_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "source_raster"))
        writer.writeheader()
        writer.writerows(test_rows)
    catalog_count = write_catalog(unlabeled_root, test_sources, catalog) if unlabeled_root is not None and catalog is not None else None
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "input_data": str(data), "source_manifest": str(source_manifest),
        "seed": args.seed, "counts": counts, "source_counts": {key: len(value) for key, value in split_sources.items()},
        "test_sources": sorted(test_sources), "val_sources": sorted(val_sources),
        "source_overlap_train_val": sorted(split_sources["train"] & split_sources["val"]),
        "source_overlap_train_test": sorted(split_sources["train"] & split_sources["test"]),
        "source_overlap_val_test": sorted(split_sources["val"] & split_sources["test"]),
        "test_source_manifest": str(test_manifest), "unlabeled_catalog": str(catalog) if catalog else None,
        "unlabeled_catalog_images": catalog_count,
    }
    (output_root / "source_disjoint_split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"data": str(output_data), "audit": str(output_root / "source_disjoint_split_audit.json"), **counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
