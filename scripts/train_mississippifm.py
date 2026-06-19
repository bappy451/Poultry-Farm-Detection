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

from poultry_geo.train_mississippifm import load_mississippifm_config, train_mississippifm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_mississippifm.py",
        description="Train the overlap-aware MississippiFM encoder from a YAML config.",
    )
    parser.add_argument("config_path")
    parser.add_argument("data_root")
    parser.add_argument("adjacency_csv")
    parser.add_argument("output_dir")
    parser.add_argument("--model-size", default=None, choices=["vit_s", "vit_b", "vit_l"])
    parser.add_argument("--lambda-patch", type=float, default=None)
    parser.add_argument("--lambda-overlap", type=float, default=None)
    parser.add_argument("--lambda-gram", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_mississippifm_config(args.config_path)
    if args.model_size is not None:
        config.model_size = args.model_size
    if args.lambda_patch is not None:
        config.lambda_patch = args.lambda_patch
    if args.lambda_overlap is not None:
        config.lambda_overlap = args.lambda_overlap
    if args.lambda_gram is not None:
        config.lambda_gram = args.lambda_gram
    config.output_dir = Path(args.output_dir)
    summary = train_mississippifm(
        config,
        data_root=args.data_root,
        adjacency_csv=args.adjacency_csv,
        output_dir=args.output_dir,
        resume=args.resume,
        device=args.device,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
