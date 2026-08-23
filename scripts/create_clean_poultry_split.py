"""Create a clean raw-train-only Poultry YOLO split and leakage audit.

The repository's historical ``poultry_80_10_10`` dataset was built by mixing
the raw train and raw test folders before resplitting.  This utility keeps the
raw train folder as the only source for train/validation and leaves the raw
test folder untouched.  It also keeps all Roboflow ``.rf.<hash>`` variants of
one tile in the same train/validation split and records exact/perceptual and
group-level overlap with the external test folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
ROBoflow_VARIANT = re.compile(r"^(?P<base>.+?)\.rf\.[0-9a-f]{32}$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-root",
        default=str(PROJECT_ROOT / "temp" / "poultry_houses_train" / "train"),
        help="Raw training root containing images/ and labels/.",
    )
    parser.add_argument(
        "--test-root",
        default=str(PROJECT_ROOT / "temp" / "poultry_houses_test" / "test"),
        help="Untouched external-test root containing images/ and labels/.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "datasets" / "poultry_clean_raw_90_10"),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "poultry_clean_raw_90_10.yaml"),
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory/config after explicit confirmation.",
    )
    return parser.parse_args()


def image_group(path: Path) -> str:
    """Return the pre-Roboflow tile identity for a filename."""
    match = ROBoflow_VARIANT.match(path.stem)
    return match.group("base") if match else path.stem


def collect_pairs(root: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    image_dir = root / "images"
    label_dir = root / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and labels/ under {root}")
    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for image in sorted(image_dir.iterdir()):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = label_dir / f"{image.stem}.txt"
        if label.exists():
            pairs.append((image, label))
        else:
            missing.append(image.name)
    if missing:
        raise RuntimeError(f"Missing labels for {len(missing)} images under {root}; first={missing[:3]}")
    if not pairs:
        raise RuntimeError(f"No image/label pairs found under {root}")
    return pairs, missing


def choose_validation_groups(
    groups: dict[str, list[tuple[Path, Path]]],
    *,
    val_fraction: float,
    seed: int,
) -> set[str]:
    target = int(round(sum(len(items) for items in groups.values()) * val_fraction))
    names = list(groups)
    rng = random.Random(seed)
    rng.shuffle(names)
    # Greedily approach the target while retaining deterministic tie-breaking.
    selected: set[str] = set()
    total = 0
    while names:
        current_error = abs(total - target)
        candidates = [name for name in names if abs(total + len(groups[name]) - target) <= current_error]
        if not candidates:
            break
        name = min(candidates, key=lambda item: (abs(total + len(groups[item]) - target), item))
        selected.add(name)
        total += len(groups[name])
        names.remove(name)

    # A small deterministic one-for-one improvement handles an unlucky greedy
    # choice without ever splitting a tile group.
    all_names = set(groups)
    for _ in range(4):
        best: tuple[int, str, str] | None = None
        for outgoing in sorted(selected):
            for incoming in sorted(all_names - selected):
                new_total = total - len(groups[outgoing]) + len(groups[incoming])
                candidate = (abs(new_total - target), outgoing, incoming)
                if best is None or candidate < best:
                    best = candidate
        if best is None or best[0] >= abs(total - target):
            break
        _, outgoing, incoming = best
        selected.remove(outgoing)
        selected.add(incoming)
        total = total - len(groups[outgoing]) + len(groups[incoming])
    if not selected or len(selected) == len(groups):
        raise RuntimeError("Unable to create non-empty train/validation group split.")
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path) -> int:
    """Return a dependency-light 64-bit perceptual hash for audit purposes."""
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("L").resize((8, 8))
        values = list(image.getdata())
    threshold = sum(values) / len(values)
    value = 0
    for pixel in values:
        value = (value << 1) | int(pixel >= threshold)
    return value


def near_duplicate_summary(
    queries: list[Path],
    candidates: list[Path],
    *,
    threshold: int = 8,
) -> dict[str, Any]:
    candidate_hashes = [average_hash(path) for path in candidates]
    minimums: list[int] = []
    for query in queries:
        query_hash = average_hash(query)
        minimums.append(min((query_hash ^ item).bit_count() for item in candidate_hashes))
    counts = {f"<= {cutoff}": sum(value <= cutoff for value in minimums) for cutoff in (0, 4, 8, 12, 16)}
    return {
        "queries": len(queries),
        "candidates": len(candidates),
        "threshold_hamming": threshold,
        "flagged": sum(value <= threshold for value in minimums),
        "minimum_hamming": min(minimums),
        "median_minimum_hamming": median(minimums),
        "mean_minimum_hamming": mean(minimums),
        "p90_minimum_hamming": sorted(minimums)[int(0.9 * (len(minimums) - 1))],
        "counts": counts,
    }


def ensure_clean_target(output_root: Path, config_path: Path, overwrite: bool) -> None:
    if not overwrite and (output_root.exists() or config_path.exists()):
        raise FileExistsError(
            f"Clean split already exists ({output_root} or {config_path}). Use a new path or --overwrite."
        )
    if overwrite:
        if output_root.exists():
            shutil.rmtree(output_root)
        if config_path.exists():
            config_path.unlink()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1.")
    train_root = Path(args.train_root).expanduser().resolve()
    test_root = Path(args.test_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    ensure_clean_target(output_root, config_path, args.overwrite)

    train_pairs, _ = collect_pairs(train_root)
    test_pairs, _ = collect_pairs(test_root)
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for pair in train_pairs:
        groups[image_group(pair[0])].append(pair)
    validation_groups = choose_validation_groups(groups, val_fraction=args.val_fraction, seed=args.seed)
    val_pairs = [pair for name in sorted(validation_groups) for pair in groups[name]]
    train_pairs_clean = [pair for name in sorted(set(groups) - validation_groups) for pair in groups[name]]

    for split, pairs in (("train", train_pairs_clean), ("val", val_pairs)):
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image, label in pairs:
            shutil.copy2(image, image_dir / image.name)
            shutil.copy2(label, label_dir / label.name)

    yaml_payload = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "test": str((test_root / "images").resolve()),
        "nc": 2,
        "names": {0: "broiler breeder", 1: "broiler house"},
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8")

    train_images = [image for image, _ in train_pairs_clean]
    val_images = [image for image, _ in val_pairs]
    test_images = [image for image, _ in test_pairs]
    train_hashes = {sha256(path): str(path) for path in train_images}
    val_hashes = {sha256(path): str(path) for path in val_images}
    test_hashes = {sha256(path): str(path) for path in test_images}
    train_groups = {image_group(path) for path in train_images}
    val_groups = {image_group(path) for path in val_images}
    test_groups = {image_group(path) for path in test_images}
    audit = {
        "protocol": "raw_train_only_grouped_split",
        "seed": args.seed,
        "val_fraction_requested": args.val_fraction,
        "raw_train_root": str(train_root),
        "raw_test_root": str(test_root),
        "train_images": len(train_images),
        "val_images": len(val_images),
        "test_images": len(test_images),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "validation_groups": sorted(validation_groups),
        "group_overlap_train_val": sorted(train_groups & val_groups),
        "group_overlap_train_test": sorted(train_groups & test_groups),
        "group_overlap_val_test": sorted(val_groups & test_groups),
        "exact_sha256_train_val": len(set(train_hashes) & set(val_hashes)),
        "exact_sha256_train_test": len(set(train_hashes) & set(test_hashes)),
        "exact_sha256_val_test": len(set(val_hashes) & set(test_hashes)),
        "near_duplicate_audit": {
            "test_vs_train": near_duplicate_summary(test_images, train_images),
            "test_vs_val": near_duplicate_summary(test_images, val_images),
        },
    }
    if audit["group_overlap_train_test"] or audit["group_overlap_val_test"]:
        audit["warning"] = (
            "The raw test images are byte-disjoint but share tile group IDs with raw train. "
            "Treat this as an augmentation/tile holdout, not a geographic/group-independent test, "
            "unless source provenance confirms otherwise."
        )
    (output_root / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"Created clean split: {config_path}")
    print(f"  train={len(train_images)} images across {len(train_groups)} groups")
    print(f"  val={len(val_images)} images across {len(val_groups)} groups")
    print(f"  test={len(test_images)} images across {len(test_groups)} groups (untouched)")
    print(f"  exact train/test hashes={audit['exact_sha256_train_test']}")
    print(f"  shared train/test groups={len(audit['group_overlap_train_test'])}")
    print(f"  audit={output_root / 'split_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
