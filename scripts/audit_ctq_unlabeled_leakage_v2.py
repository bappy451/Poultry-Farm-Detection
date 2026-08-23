"""Efficient exact-image and source-raster leakage audit for CTQ experiments.

This is a standalone replacement for the first audit script.  Test images are
hashed at most once per byte-size group, so a large unlabeled pool does not
repeatedly hash the same final-test image.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from scripts.audit_ctq_unlabeled_leakage import IMAGE_SUFFIXES, load_test_sources, sha256


def image_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def audit(*, training_data: Path, final_test_data: Path, final_test_split: str, unlabeled_root: Path, test_source_manifest: Path | None = None, catalog_output: Path | None = None) -> dict[str, object]:
    from scripts.run_ema_benchmark import _dataset_image_paths, audit_test_overlap

    overlap = audit_test_overlap(training_data, final_test_data, final_test_split=final_test_split)
    test_images = _dataset_image_paths(final_test_data, (final_test_split,))
    if not test_images:
        raise RuntimeError("Final-test split resolves to no images.")
    test_by_size: dict[int, list[Path]] = defaultdict(list)
    for image in test_images:
        test_by_size[image.stat().st_size].append(image)
    sources = load_test_sources(test_source_manifest)
    if sources:
        missing = [str(path) for path in test_images if path not in sources]
        if missing:
            raise ValueError(f"Test source manifest omits {len(missing)} final-test images; first={missing[:2]}")
    test_source_ids = set(sources.values())
    cached_test_hashes: dict[int, dict[str, list[Path]]] = {}
    scanned = same_size_candidates = 0
    exact_matches: list[dict[str, str]] = []
    source_counts: dict[str, int] = defaultdict(int)
    unlabeled_source_ids: set[str] = set()
    catalog_rows: list[dict[str, str]] = []
    for image in image_files(unlabeled_root):
        scanned += 1
        relative = image.relative_to(unlabeled_root)
        source_id = relative.parts[0] if len(relative.parts) > 1 else "__root__"
        source_counts[source_id] += 1
        unlabeled_source_ids.add(source_id)
        size = image.stat().st_size
        if size in test_by_size:
            same_size_candidates += 1
            if size not in cached_test_hashes:
                hashes: dict[str, list[Path]] = defaultdict(list)
                for test_image in test_by_size[size]:
                    hashes[sha256(test_image)].append(test_image)
                cached_test_hashes[size] = hashes
            for test_image in cached_test_hashes[size].get(sha256(image), []):
                exact_matches.append({"unlabeled": str(image), "final_test": str(test_image)})
        if catalog_output is not None and source_id not in test_source_ids:
            catalog_rows.append({"path": str(image.resolve()), "source_raster": source_id})
    source_overlap = sorted(test_source_ids & unlabeled_source_ids)
    if catalog_output is not None:
        if not sources:
            raise ValueError("--write-catalog requires --test-source-manifest.")
        if catalog_output.exists():
            raise FileExistsError(f"Refusing to overwrite catalog: {catalog_output}")
        catalog_output.parent.mkdir(parents=True, exist_ok=True)
        with catalog_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("path", "source_raster"))
            writer.writeheader()
            writer.writerows(catalog_rows)
    source_verified = bool(sources)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": "audit_ctq_unlabeled_leakage_v2",
        "training_vs_final_test": overlap,
        "unlabeled_root": str(unlabeled_root),
        "unlabeled_images_scanned": scanned,
        "same_size_candidates_hashed": same_size_candidates,
        "exact_unlabeled_final_test_matches": exact_matches,
        "test_source_provenance": "verified" if source_verified else "unverified",
        "test_source_ids": sorted(test_source_ids),
        "test_source_ids_present_in_unlabeled_pool": source_overlap,
        "unlabeled_source_count": len(unlabeled_source_ids),
        "unlabeled_images_per_source": dict(sorted(source_counts.items())),
        "excluded_catalog": str(catalog_output) if catalog_output else None,
        "catalog_images_after_test_source_exclusion": len(catalog_rows) if catalog_output else None,
        "inductive_test_safe": bool(source_verified and not exact_matches and not source_overlap and int(overlap["exact_sha256_matches"]) == 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--final-test-data", required=True)
    parser.add_argument("--final-test-split", default="test")
    parser.add_argument("--unlabeled-root", required=True)
    parser.add_argument("--test-source-manifest", default=None)
    parser.add_argument("--write-catalog", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    training, test, unlabeled, output = (Path(value).expanduser().resolve() for value in (args.training_data, args.final_test_data, args.unlabeled_root, args.output))
    manifest = Path(args.test_source_manifest).expanduser().resolve() if args.test_source_manifest else None
    catalog = Path(args.write_catalog).expanduser().resolve() if args.write_catalog else None
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output}")
    if not training.is_file() or not test.is_file() or not unlabeled.is_dir():
        raise FileNotFoundError("Training YAML, final-test YAML, or unlabeled root is missing.")
    report = audit(training_data=training, final_test_data=test, final_test_split=args.final_test_split, unlabeled_root=unlabeled, test_source_manifest=manifest, catalog_output=catalog)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "exact_matches": len(report["exact_unlabeled_final_test_matches"]), "source_provenance": report["test_source_provenance"], "source_collisions": len(report["test_source_ids_present_in_unlabeled_pool"]), "inductive_test_safe": report["inductive_test_safe"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
