"""Create deterministic, nested label subsets from the clean raw-train split.

The input YAML must point at ``datasets/poultry_clean_raw_90_10`` (or an
equivalent train/validation split).  Subsets are represented as Ultralytics-compatible
text manifests of absolute image paths, so no image bytes are duplicated.  Tile variants sharing
the same ``.rf.<hash>``-stripped group remain together in every subset.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
ROBOFLOW_VARIANT = re.compile(r"^(?P<base>.+?)\.rf\.[0-9a-f]{32}$", re.IGNORECASE)
DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-data",
        default=str(PROJECT_ROOT / "configs" / "poultry_clean_raw_90_10.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "datasets" / "poultry_clean_label_subsets"),
    )
    parser.add_argument(
        "--config-dir",
        default=str(PROJECT_ROOT / "configs" / "poultry_clean_label_subsets"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fractions",
        default="1,5,10,25,50,100",
        help="Comma-separated requested percentages; the generated subsets are nested.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def group_key(path: Path) -> str:
    match = ROBOFLOW_VARIANT.match(path.stem)
    return match.group("base") if match else path.stem


def resolve_entry(payload: dict[str, Any], yaml_path: Path, key: str) -> list[Path]:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"Dataset YAML has no '{key}' entry: {yaml_path}")
    root_value = payload.get("path")
    root = Path(root_value) if root_value else yaml_path.parent
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    values = value if isinstance(value, list) else [value]
    paths: list[Path] = []
    for item in values:
        candidate = Path(str(item))
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if candidate.is_dir():
            paths.extend(sorted(path for path in candidate.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES))
        elif candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(candidate)
        else:
            raise FileNotFoundError(f"Dataset entry does not exist: {candidate}")
    return sorted(set(paths))


def choose_nested_groups(
    groups: dict[str, list[Path]], fractions: list[float], seed: int
) -> dict[float, list[str]]:
    rng = random.Random(seed)
    ordered = sorted(groups)
    rng.shuffle(ordered)
    selected: list[str] = []
    result: dict[float, list[str]] = {}
    total = sum(len(paths) for paths in groups.values())
    cursor = 0
    for fraction in sorted(fractions):
        target = total if fraction >= 1.0 else max(1, round(total * fraction))
        while cursor < len(ordered) and sum(len(groups[name]) for name in selected) < target:
            selected.append(ordered[cursor])
            cursor += 1
        result[fraction] = list(selected)
    return result


def main() -> int:
    args = parse_args()
    fractions = sorted({float(token) / 100.0 for token in args.fractions.split(",") if token.strip()})
    if not fractions or any(fraction <= 0.0 or fraction > 1.0 for fraction in fractions):
        raise ValueError("Fractions must be percentages in the interval (0, 100].")
    clean_yaml = Path(args.clean_data).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    config_dir = Path(args.config_dir).expanduser().resolve()
    if not clean_yaml.is_file():
        raise FileNotFoundError(clean_yaml)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_dir}; use --overwrite or a new directory.")
    if args.overwrite and output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    payload = yaml.safe_load(clean_yaml.read_text(encoding="utf-8")) or {}
    train_images = resolve_entry(payload, clean_yaml, "train")
    val_images = resolve_entry(payload, clean_yaml, "val")
    if not train_images or not val_images:
        raise RuntimeError("Clean YAML must contain non-empty train and validation splits.")
    labels_root = train_images[0].parent.parent.parent / "labels"
    for image in train_images + val_images:
        label = labels_root / ("train" if image.parent.name == "train" else "val") / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing label for {image}: {label}")

    groups: dict[str, list[Path]] = defaultdict(list)
    for image in train_images:
        groups[group_key(image)].append(image)
    assignments = choose_nested_groups(groups, fractions, args.seed)
    manifests: dict[str, Any] = {
        "clean_data": str(clean_yaml),
        "seed": args.seed,
        "fractions": {},
    }

    for fraction, group_names in assignments.items():
        percent = int(round(fraction * 100))
        name = f"{percent}percent"
        selected = [path for group in group_names for path in sorted(groups[group])]
        train_manifest = manifest_dir / f"poultry_clean_{name}_train.txt"
        train_manifest.write_text(
            "".join(f"{path.resolve()}\n" for path in selected),
            encoding="utf-8",
        )
        config = {
            "path": str(clean_yaml.parent),
            "train": str(train_manifest.resolve()),
            "val": str((val_images[0].parent).resolve()),
            "test": payload.get("test"),
            "nc": payload.get("nc", 2),
            "names": payload.get("names", {0: "broiler breeder", 1: "broiler house"}),
        }
        config_path = config_dir / f"poultry_clean_{name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        manifests["fractions"][name] = {
            "requested_fraction": fraction,
            "actual_fraction": len(selected) / len(train_images),
            "image_count": len(selected),
            "group_count": len(group_names),
            "groups": sorted(group_names),
            "config": str(config_path),
            "train_images": [str(path.resolve()) for path in selected],
        }

    # Verify exact nesting explicitly before writing the manifest.
    names = sorted(manifests["fractions"], key=lambda name: int(name.replace("percent", "")))
    previous: set[str] = set()
    for name in names:
        current = set(manifests["fractions"][name]["train_images"])
        if not previous.issubset(current):
            raise RuntimeError(f"Nested-subset invariant failed at {name}.")
        previous = current
    manifest_path = output_dir / "subsets_manifest.json"
    manifest_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(f"Created {len(names)} nested clean subsets under {config_dir}")
    for name in names:
        row = manifests["fractions"][name]
        print(f"  {name}: {row['image_count']} images, {row['group_count']} groups, actual={row['actual_fraction']:.4f}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
