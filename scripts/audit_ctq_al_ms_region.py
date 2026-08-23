"""Record an AL-labeled/MS-unlabeled CTQ leakage audit without inventing provenance."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-audit", required=True, help="Output of audit_ctq_unlabeled_leakage_v2.")
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--unlabeled-root", required=True)
    parser.add_argument("--labeled-region", required=True)
    parser.add_argument("--unlabeled-region", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    exact_path = Path(args.exact_audit).expanduser().resolve()
    training, test, unlabeled, output = (Path(value).expanduser().resolve() for value in (args.training_data, args.test_data, args.unlabeled_root, args.output))
    if not exact_path.is_file() or not training.is_file() or not test.is_file() or not unlabeled.is_dir():
        raise FileNotFoundError("Exact audit, data YAML, or unlabeled root is missing.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output}")
    exact = json.loads(exact_path.read_text(encoding="utf-8"))
    if Path(exact.get("unlabeled_root", "")).resolve() != unlabeled:
        raise ValueError("Exact audit was run on a different unlabeled root.")
    labeled_region, unlabeled_region = args.labeled_region.strip().upper(), args.unlabeled_region.strip().upper()
    if not labeled_region or not unlabeled_region or labeled_region == unlabeled_region:
        raise ValueError("labeled-region and unlabeled-region must be distinct non-empty geographic regions.")
    exact_matches = exact.get("exact_unlabeled_final_test_matches", [])
    train_test = exact.get("training_vs_final_test", {})
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "AL-labeled_MS-unlabeled",
        "training_data": str(training),
        "held_out_test_data": str(test),
        "unlabeled_root": str(unlabeled),
        "labeled_region_declared_by_user": labeled_region,
        "unlabeled_region_declared_by_user": unlabeled_region,
        "regions_are_distinct": True,
        "exact_audit": str(exact_path),
        "unlabeled_images_scanned": exact.get("unlabeled_images_scanned"),
        "exact_unlabeled_test_matches": len(exact_matches),
        "exact_train_test_matches": int(train_test.get("exact_sha256_matches", -1)),
        "unlabeled_pool_independent_of_AL_test_by_declared_state_and_exact_audit": bool(not exact_matches and int(train_test.get("exact_sha256_matches", -1)) == 0),
        "within_AL_source_raster_disjointness": "unverified: labeled Roboflow exports lack crop-to-raster provenance",
        "selection_rule": "Use only AL validation for selection; do not tune on the previously observed AL test.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("protocol", "unlabeled_images_scanned", "exact_unlabeled_test_matches", "exact_train_test_matches", "unlabeled_pool_independent_of_AL_test_by_declared_state_and_exact_audit")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
