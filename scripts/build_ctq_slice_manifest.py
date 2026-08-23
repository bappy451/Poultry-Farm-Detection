"""Build and validate cross-tile geometry from source_id/slice_row_col image trees."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from PIL import Image

from scripts.ctq_tile_manifest import validate_cross_tile_manifest


PATTERN = re.compile(r"^slice_(?P<row>\d+)_(?P<col>\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
FIELDS = ("tile_path", "source_raster", "row_offset", "col_offset", "tile_width", "tile_height")


def build_manifest(root: str | Path, output: str | Path, *, stride: int = 256) -> Path:
    """Build an explicit manifest from the verified NAIP directory convention."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    rows: list[dict[str, object]] = []
    dimensions: set[tuple[int, int]] = set()
    for source in sorted(item for item in root_path.iterdir() if item.is_dir()):
        for path in sorted(item for item in source.iterdir() if item.is_file()):
            match = PATTERN.match(path.name)
            if match is None:
                continue
            with Image.open(path) as image:
                width, height = image.size
            dimensions.add((width, height))
            rows.append({
                "tile_path": path.relative_to(root_path).as_posix(),
                "source_raster": source.name,
                "row_offset": int(match.group("row")) * stride,
                "col_offset": int(match.group("col")) * stride,
                "tile_width": width,
                "tile_height": height,
            })
    if not rows:
        raise ValueError(f"No source_id/slice_row_col images found below {root_path}")
    if len(dimensions) != 1:
        raise ValueError(f"Expected a shared tile size, found {sorted(dimensions)}")
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    validate_cross_tile_manifest(output_path, root=root_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("output")
    parser.add_argument("--stride", type=int, default=256)
    args = parser.parse_args()
    output = build_manifest(args.root, args.output, stride=args.stride)
    print(json.dumps({"manifest": str(output), **validate_cross_tile_manifest(output, root=args.root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
