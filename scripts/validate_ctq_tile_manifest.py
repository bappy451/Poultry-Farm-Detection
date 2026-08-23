"""CLI preflight for CTQ cross-tile supervision metadata."""

from __future__ import annotations

import argparse
import json

from scripts.ctq_tile_manifest import validate_cross_tile_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--root", default=None)
    parser.add_argument("--min-pairs", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(validate_cross_tile_manifest(args.manifest, root=args.root, min_pairs=args.min_pairs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
