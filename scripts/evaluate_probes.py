#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path = [path for path in sys.path if path != str(SCRIPT_DIR)]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
sys.modules.pop("poultry_geo", None)

from poultry_geo.evaluate_probes import evaluate_mississippi_probes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_probes.py",
        description="Evaluate frozen MississippiFM features with lightweight downstream probes.",
    )
    parser.add_argument("config_path")
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = evaluate_mississippi_probes(
        args.config_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        resume=args.resume,
        device=args.device,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
