"""Create a strict image-to-source-raster provenance CSV template."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=("train", "val", "test"))
    args = parser.parse_args()
    data = Path(args.data).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(data)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite provenance template: {output}")
    from scripts.run_ema_benchmark import _dataset_image_paths

    images = _dataset_image_paths(data, tuple(args.splits))
    if not images:
        raise RuntimeError("No images found for requested splits.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "source_raster"))
        writer.writeheader()
        for image in images:
            writer.writerow({"image_path": str(image.resolve()), "source_raster": ""})
    print(f"Wrote {len(images)} provenance rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
