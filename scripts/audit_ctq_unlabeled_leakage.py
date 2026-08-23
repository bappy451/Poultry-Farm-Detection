"""Audit the CTQ unlabeled pool against a locked test split.

The audit detects byte-identical images without hashing every unlabeled tile,
and, when source provenance is supplied, detects test-source tiles inside the
unlabeled pool.  A source-provenance verdict is deliberately ``unverified``
unless a manifest maps every final-test image to a source-raster identifier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".npy"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def load_test_sources(path: Path | None) -> dict[Path, str]:
    """Return final-test image to source-raster mapping from a strict CSV."""
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)
    mapping: dict[Path, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "source_raster"}
        if reader.fieldnames is None or required - set(reader.fieldnames):
            raise ValueError(f"Source manifest must contain {sorted(required)}")
        for line_number, row in enumerate(reader, start=2):
            image = Path(str(row["image_path"]).strip()).expanduser().resolve()
            source = str(row["source_raster"]).strip()
            if not image.is_file() or not source or source.lower() in {"unknown", "__unknown__"}:
                raise ValueError(f"Invalid source-manifest row {line_number}")
            if image in mapping and mapping[image] != source:
                raise ValueError(f"Conflicting source_raster for {image}")
            mapping[image] = source
    return mapping


def audit(
    *,
    training_data: Path,
    final_test_data: Path,
    final_test_split: str,
    unlabeled_root: Path,
    test_source_manifest: Path | None = None,
    catalog_output: Path | None = None,
) -> dict[str, object]:
    """Perform exact-image and optional source-raster exclusion checks."""
    from scripts.run_ema_benchmark import _dataset_image_paths, audit_test_overlap

    overlap = audit_test_overlap(training_data, final_test_data, final_test_split=final_test_split)
    test_images = _dataset_image_paths(final_test_data, (final_test_split,))
    if not test_images:
        raise RuntimeError("Final-test split resolves to no images.")
    test_by_size: dict[int, list[Path]] = defaultdict(list)
    test_hashes: dict[str, list[Path]] = defaultdict(list)
    for image in test_images:
        test_by_size[image.stat().st_size].append(image)
    # Lazily hash only groups for which an unlabeled file has the same byte size.
    scanned = 0
    same_size_candidates = 0
    exact_matches: list[dict[str, str]] = []
    source_counts: dict[str, int] = defaultdict(int)
    catalog_rows: list[dict[str, str]] = []
    sources = load_test_sources(test_source_manifest)
    missing_sources = sorted(str(path) for path in test_images if sources and path not in sources)
    if sources and missing_sources:
        raise ValueError(f"Test source manifest omits {len(missing_sources)} final-test images; first={missing_sources[:2]}")
    test_source_ids = set(sources.values())
    unlabeled_source_ids: set[str] = set()
    for image in image_files(unlabeled_root):
        scanned += 1
        relative = image.relative_to(unlabeled_root)
        source_id = relative.parts[0] if len(relative.parts) > 1 else "__root__"
        unlabeled_source_ids.add(source_id)
        source_counts[source_id] += 1
        if image.stat().st_size in test_by_size:
            same_size_candidates += 1
            for test_image in test_by_size[image.stat().st_size]:
                digest = sha256(test_image)
                test_hashes[digest].append(test_image)
            digest = sha256(image)
            for test_image in test_hashes.get(digest, []):
                exact_matches.append({"unlabeled": str(image), "final_test": str(test_image)})
        if catalog_output is not None and source_id not in test_source_ids:
            catalog_rows.append({"path": str(image.resolve()), "source_raster": source_id})

    source_overlap = sorted(test_source_ids & unlabeled_source_ids)
    if catalog_output is not None:
        if not sources:
            raise ValueError("--write-catalog requires --test-source-manifest.")
        if catalog_output.exists():
            raise FileExistsError(f"Refusing to overwrite existing catalog: {catalog_output}")
        catalog_output.parent.mkdir(parents=True, exist_ok=True)
        with catalog_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("path", "source_raster"))
            writer.writeheader()
            writer.writerows(catalog_rows)

    source_verified = bool(sources)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_vs_final_test": overlap,
        "unlabeled_root": str(unlabeled_root),
        "unlabeled_images_scanned": scanned,
        "same_size_candidates_hashed": same_size_candidates,
        "exact_unlabeled_final_test_matches": exact_matches,
        "test_source_provenance": "verified" if source_verified else "unverified",
        "test_source_ids": sorted(test_source_ids),
        "test_source_ids_present_in_unlabeled_pool": source_overlap,
        "unlabeled_source_count": len(unlabeled_source_ids),
        "excluded_catalog": str(catalog_output) if catalog_output is not None else None,
        "catalog_images_after_test_source_exclusion": len(catalog_rows) if catalog_output is not None else None,
        "inductive_test_safe": bool(
            source_verified
            and not exact_matches
            and not source_overlap
            and int(overlap["exact_sha256_matches"]) == 0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", required=True, help="Training/validation YAML used by CTQ.")
    parser.add_argument("--final-test-data", required=True, help="Locked test-only YAML.")
    parser.add_argument("--final-test-split", default="test")
    parser.add_argument("--unlabeled-root", required=True)
    parser.add_argument("--test-source-manifest", default=None, help="CSV: image_path,source_raster for every final-test image.")
    parser.add_argument("--write-catalog", default=None, help="Optional allowed unlabeled CSV after excluding final-test sources.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    training = Path(args.training_data).expanduser().resolve()
    final_test = Path(args.final_test_data).expanduser().resolve()
    unlabeled = Path(args.unlabeled_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    manifest = Path(args.test_source_manifest).expanduser().resolve() if args.test_source_manifest else None
    catalog = Path(args.write_catalog).expanduser().resolve() if args.write_catalog else None
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output}")
    if not training.is_file() or not final_test.is_file() or not unlabeled.is_dir():
        raise FileNotFoundError("Training YAML, final-test YAML, or unlabeled root is missing.")
    report = audit(
        training_data=training,
        final_test_data=final_test,
        final_test_split=args.final_test_split,
        unlabeled_root=unlabeled,
        test_source_manifest=manifest,
        catalog_output=catalog,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "exact_matches": len(report["exact_unlabeled_final_test_matches"]),
        "source_provenance": report["test_source_provenance"],
        "source_collisions": len(report["test_source_ids_present_in_unlabeled_pool"]),
        "inductive_test_safe": report["inductive_test_safe"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
