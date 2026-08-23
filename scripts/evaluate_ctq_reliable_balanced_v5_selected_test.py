"""Evaluate a frozen CTQ screen using the resolution recorded in its config."""

from __future__ import annotations

import argparse

import yaml

from scripts.evaluate_ctq_reliable_balanced_v5_test import main as evaluate
from scripts.evaluate_ctq_reliable_balanced_v5_test import resolve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", required=True)
    parser.add_argument("--config", required=True, help="Validation-selected CTQ config produced by the resolution selector.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8")) or {}
    if "resolution_selection" not in config:
        raise ValueError("Refusing final test: config was not generated from a completed resolution screen.")
    resolution = config["resolution_selection"]
    if resolution.get("selection_used_test") is not False:
        raise ValueError("Refusing final test: resolution selection must be validation-only.")
    data = config.get("data")
    adaptation = config.get("adaptation", {})
    if not data or not adaptation.get("imgsz") or not adaptation.get("batch"):
        raise ValueError("Selected CTQ config lacks data, adaptation imgsz, or batch.")
    return evaluate(["--screen-dir", args.screen_dir, "--data", str(data), "--output-dir", args.output_dir, "--imgsz", str(adaptation["imgsz"]), "--batch", str(adaptation["batch"]), "--workers", str(args.workers), "--device", str(args.device)])


if __name__ == "__main__":
    raise SystemExit(main())
