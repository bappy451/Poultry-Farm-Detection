#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path = [path for path in sys.path if path != str(SCRIPT_DIR)]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
sys.modules.pop("poultry_geo", None)

from poultry_geo.mississippi_adjacency import write_adjacency_index_csv


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="create_adjacency_index.py",
        description="Build a MississippiFM adjacency index from existing tile filename metadata.",
    )
    parser.add_argument("input_dir")
    parser.add_argument("output_csv")
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=512)
    args = parser.parse_args(argv)
    output = write_adjacency_index_csv(
        args.input_dir,
        args.output_csv,
        overlap_ratio=args.overlap_ratio,
        tile_size=args.tile_size,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
