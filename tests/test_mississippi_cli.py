from __future__ import annotations

from pathlib import Path

from poultry_geo.cli import build_parser


def test_cli_parses_mississippi_create_adjacency_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "mississippi-create-adjacency",
            "tiles",
            "adjacency.csv",
            "--overlap-ratio",
            "0.5",
        ]
    )

    assert args.command == "mississippi-create-adjacency"
    assert args.input_dir == "tiles"
    assert args.output_csv == "adjacency.csv"
    assert args.overlap_ratio == 0.5


def test_cli_parses_mississippi_pretrain_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "mississippi-pretrain",
            "config.yaml",
            "tiles",
            "adjacency.csv",
            "trained_models/mississippifm/run_a",
            "--model-size",
            "vit_b",
            "--lambda-patch",
            "0.5",
            "--lambda-overlap",
            "1.5",
            "--lambda-gram",
            "0.25",
            "--resume",
        ]
    )

    assert args.command == "mississippi-pretrain"
    assert args.config_path == "config.yaml"
    assert args.data_root == "tiles"
    assert args.adjacency_csv == "adjacency.csv"
    assert args.output_dir == "trained_models/mississippifm/run_a"
    assert args.model_size == "vit_b"
    assert args.lambda_patch == 0.5
    assert args.lambda_overlap == 1.5
    assert args.lambda_gram == 0.25
    assert args.resume is True


def test_cli_parses_mississippi_eval_probes_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "mississippi-eval-probes",
            "probe_config.yaml",
            "checkpoint.pt",
            "trained_models/mississippifm/probes",
            "--resume",
        ]
    )

    assert args.command == "mississippi-eval-probes"
    assert args.config_path == "probe_config.yaml"
    assert args.checkpoint_path == "checkpoint.pt"
    assert args.output_dir == "trained_models/mississippifm/probes"
    assert args.resume is True


def test_cli_parses_mississippi_eval_benchmark_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "mississippi-eval-benchmark",
            "benchmark.yaml",
            "checkpoint.pt",
            "trained_models/mississippifm/benchmark",
            "--device",
            "cpu",
        ]
    )

    assert args.command == "mississippi-eval-benchmark"
    assert args.config_path == "benchmark.yaml"
    assert args.checkpoint_path == "checkpoint.pt"
    assert args.output_dir == "trained_models/mississippifm/benchmark"
    assert args.device == "cpu"


def test_mississippi_script_entrypoints_exist() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert (repo_root / "scripts" / "create_adjacency_index.py").exists()
    assert (repo_root / "scripts" / "train_mississippifm.py").exists()
    assert (repo_root / "scripts" / "evaluate_probes.py").exists()
    assert (repo_root / "scripts" / "geographic_benchmark.py").exists()
