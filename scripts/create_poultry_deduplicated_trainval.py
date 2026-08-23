"""Create content-deduplicated AL train/validation manifests with provenance.

The source dataset is read-only. Exact duplicate images are retained once per
split, train/validation overlap is forbidden, and duplicate annotations must
agree numerically before a canonical image is selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    indices = [index for index, value in enumerate(parts) if value.lower() == "images"]
    if not indices:
        raise ValueError(f"Image path has no images directory component: {image}")
    parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def _annotations(path: Path) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = tuple(float(value) for value in line.split())
        if len(values) != 9 or int(values[0]) != values[0] or int(values[0]) not in {0, 1}:
            raise ValueError(f"Invalid two-class OBB row at {path}:{line_number}")
        rows.append(values)
    return sorted(rows)


def _annotations_agree(paths: list[Path], tolerance: float = 1e-6) -> bool:
    reference = _annotations(paths[0])
    for path in paths[1:]:
        candidate = _annotations(path)
        if len(candidate) != len(reference):
            return False
        for first, second in zip(reference, candidate):
            if int(first[0]) != int(second[0]):
                return False
            if any(abs(a - b) > tolerance for a, b in zip(first[1:], second[1:])):
                return False
    return True


def _collect(source: Path) -> list[Path]:
    if source.is_file():
        paths = [_resolve(line.strip()) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif source.is_dir():
        paths = sorted(path.resolve() for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    else:
        raise FileNotFoundError(source)
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"Image source is empty or contains missing files: {source}")
    if len(set(paths)) != len(paths):
        raise ValueError(f"Image source contains duplicate paths: {source}")
    return paths


def _deduplicate(paths: list[Path], split: str) -> tuple[list[Path], list[dict[str, Any]]]:
    groups: dict[str, list[Path]] = {}
    for path in paths:
        groups.setdefault(_sha256(path), []).append(path)
    retained: list[Path] = []
    records: list[dict[str, Any]] = []
    for image_sha, images in sorted(groups.items()):
        images = sorted(images)
        labels = [_label_path(path) for path in images]
        if any(not path.is_file() for path in labels):
            raise FileNotFoundError(f"Missing label in duplicate group: {images}")
        if not _annotations_agree(labels):
            raise ValueError(f"Duplicate pixels have materially different annotations: {images}")
        canonical = images[0]
        retained.append(canonical)
        records.append(
            {
                "split": split,
                "image": str(canonical),
                "image_sha256": image_sha,
                "label": str(_label_path(canonical)),
                "label_sha256": _sha256(_label_path(canonical)),
                "discarded_exact_duplicates": [str(path) for path in images[1:]],
            }
        )
    return sorted(retained), records


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-source",
        default="datasets/poultry_clean_label_subsets/manifests/poultry_clean_100percent_train.txt",
    )
    parser.add_argument("--val-source", default="datasets/poultry_clean_raw_90_10/images/val")
    parser.add_argument(
        "--output-dir", default="datasets/poultry_clean_label_subsets/manifests/deduplicated_v1"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = _resolve(args.output_dir)
    train_manifest = output_dir / "train.txt"
    val_manifest = output_dir / "val.txt"
    provenance_path = output_dir / "provenance.json"
    targets = (train_manifest, val_manifest, provenance_path)
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError(f"Deduplicated manifests already exist under {output_dir}; use --overwrite.")

    train_source, val_source = _resolve(args.train_source), _resolve(args.val_source)
    raw_train, raw_val = _collect(train_source), _collect(val_source)
    train, train_records = _deduplicate(raw_train, "train")
    val, val_records = _deduplicate(raw_val, "val")
    train_hashes = {record["image_sha256"] for record in train_records}
    val_hashes = {record["image_sha256"] for record in val_records}
    if train_hashes & val_hashes:
        raise ValueError("Exact image content overlaps train and validation.")

    _atomic_text(train_manifest, "\n".join(str(path) for path in train) + "\n")
    _atomic_text(val_manifest, "\n".join(str(path) for path in val) + "\n")
    provenance = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "exact_sha256_within_split_deduplication",
        "annotation_tolerance": 1e-6,
        "source_train": str(train_source),
        "source_val": str(val_source),
        "raw_train_images": len(raw_train),
        "raw_val_images": len(raw_val),
        "retained_train_images": len(train),
        "retained_val_images": len(val),
        "discarded_train_duplicates": len(raw_train) - len(train),
        "discarded_val_duplicates": len(raw_val) - len(val),
        "exact_train_val_overlap": 0,
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "train_manifest_sha256": _sha256(train_manifest),
        "val_manifest_sha256": _sha256(val_manifest),
        "records": [*train_records, *val_records],
    }
    _atomic_text(provenance_path, json.dumps(provenance, indent=2, sort_keys=True))
    print(
        json.dumps(
            {key: provenance[key] for key in (
                "raw_train_images", "retained_train_images", "discarded_train_duplicates",
                "raw_val_images", "retained_val_images", "discarded_val_duplicates",
                "exact_train_val_overlap", "train_manifest", "val_manifest",
            )},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
