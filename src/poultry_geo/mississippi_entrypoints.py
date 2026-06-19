from __future__ import annotations

import argparse
from pathlib import Path

from poultry_geo.evaluate_probes import evaluate_mississippi_probes
from poultry_geo.geographic_benchmark import run_geographic_benchmark
from poultry_geo.mississippi_adjacency import write_adjacency_index_csv
from poultry_geo.train_mississippifm import load_mississippifm_config, train_mississippifm


def create_adjacency_main() -> int:
    parser = argparse.ArgumentParser(description="Build a MississippiFM adjacency CSV from existing tile filenames.")
    parser.add_argument("input_dir")
    parser.add_argument("output_csv")
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=512)
    args = parser.parse_args()
    output = write_adjacency_index_csv(
        args.input_dir,
        args.output_csv,
        overlap_ratio=args.overlap_ratio,
        tile_size=args.tile_size,
    )
    print(output)
    return 0


def pretrain_main() -> int:
    parser = argparse.ArgumentParser(description="Train the overlap-aware MississippiFM encoder.")
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
    args = parser.parse_args()
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


def eval_probes_main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen MississippiFM features with probe tasks.")
    parser.add_argument("config_path")
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    summary = evaluate_mississippi_probes(
        args.config_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        resume=args.resume,
        device=args.device,
    )
    print(summary)
    return 0


def eval_benchmark_main() -> int:
    parser = argparse.ArgumentParser(description="Run held-out geographic and scale-shift MississippiFM benchmarks.")
    parser.add_argument("config_path")
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    summary = run_geographic_benchmark(
        args.config_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        resume=args.resume,
        device=args.device,
    )
    print(summary)
    return 0
