"""Evaluate a frozen CTQ screen at the validation-selected input resolution."""

from __future__ import annotations

import argparse
import sys

import yaml

from scripts.evaluate_ctq_reliable_balanced_v5_test import main as evaluate
from scripts.evaluate_ctq_reliable_balanced_v5_test import resolve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8")) or {}
    resolution = config.get("resolution_selection")
    if not isinstance(resolution, dict) or resolution.get("selection_used_test") is not False:
        raise ValueError("Refusing final test: use a validation-only config produced by the resolution selector.")
    adaptation = config.get("adaptation", {})
    if not config.get("data") or not adaptation.get("imgsz") or not adaptation.get("batch"):
        raise ValueError("Selected CTQ config lacks data, imgsz, or batch.")
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], "--screen-dir", args.screen_dir, "--data", str(config["data"]), "--output-dir", args.output_dir, "--imgsz", str(adaptation["imgsz"]), "--batch", str(adaptation["batch"]), "--workers", str(args.workers), "--device", str(args.device)]
        return evaluate()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
