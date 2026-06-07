from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from poultry_geo.cli import build_parser


def test_cli_parses_slice_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "slice",
            "input_dir",
            "output_dir",
            "--tile-size",
            "1024",
            "--format",
            "png",
            "--jpeg-quality",
            "90",
            "--log-file",
            "command.log",
        ]
    )

    assert args.command == "slice"
    assert args.input_geotiff == "input_dir"
    assert args.output_dir == "output_dir"
    assert args.tile_size == 1024
    assert args.format == "png"
    assert args.jpeg_quality == 90
    assert args.workers == "auto"
    assert args.log_file == "command.log"


def test_core_cli_import_does_not_require_opencv() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['cv2'] = None; import poultry_geo.cli",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_cli_parses_log_file_before_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["--log-file", "command.log", "slice", "input_dir", "output_dir"])

    assert args.command == "slice"
    assert args.log_file == "command.log"


def test_cli_parses_notebook_migration_commands() -> None:
    parser = build_parser()
    args = parser.parse_args(["create-rectangles", "in.csv", "out.csv"])

    assert args.command == "create-rectangles"
    assert args.input_csv == "in.csv"
    assert args.output_csv == "out.csv"


def test_cli_parses_transformer_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-detr",
            "images/",
            "annotations.json",
            "runs/detr",
            "--epochs",
            "2",
        ]
    )

    assert args.command == "train-detr"
    assert args.train_image_dir == "images/"
    assert args.train_annotations == "annotations.json"
    assert args.epochs == 2


def test_main_help_mentions_examples_and_major_workflows() -> None:
    parser = build_parser()
    help_text = parser.format_help()

    assert "Major workflows" in help_text
    assert "poultry-geo slice" in help_text
    assert "convert GeoTIFFs to display-friendly images" in help_text
    assert "--log-file" in help_text
    assert "logged automatically" in help_text


def test_slice_help_mentions_edge_snapping_and_workers() -> None:
    parser = build_parser()
    slice_parser = parser._subparsers._group_actions[0].choices["slice"]
    help_text = slice_parser.format_help()

    assert "edge snapping" in help_text
    assert "--workers" in help_text
    assert "no zero padding" in help_text
    assert "--log-file" in help_text


def test_script_wrappers_show_help_examples() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    python_script = repo_root / "scripts" / "poultry_geo.py"
    shell_script = repo_root / "scripts" / "slice_tiles.sh"
    generic_log_script = repo_root / "scripts" / "run_logged.sh"

    python_help = subprocess.run(
        [sys.executable, str(python_script), "--help"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    shell_help = subprocess.run(
        ["bash", str(shell_script), "--help"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    generic_help = subprocess.run(
        ["bash", str(generic_log_script), "--help"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "Use `poultry-geo <command> --help`" in python_help
    assert "--workers" in shell_help
    assert "--log-file" in python_help
    assert "--log-file" in shell_help
    assert "snaps the last window to the image edge" in shell_help
    assert "64" in shell_help and "256" in shell_help
    assert "runs any shell command" in generic_help
    assert "-- command [args...]" in generic_help


def test_windows_wrappers_exist_and_document_logging() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    poultry_ps1 = repo_root / "scripts" / "poultry_geo.ps1"
    poultry_cmd = repo_root / "scripts" / "poultry_geo.cmd"
    slice_ps1 = repo_root / "scripts" / "slice_tiles.ps1"
    slice_cmd = repo_root / "scripts" / "slice_tiles.cmd"
    common_ps1 = repo_root / "scripts" / "windows_common.ps1"

    assert poultry_ps1.exists()
    assert poultry_cmd.exists()
    assert slice_ps1.exists()
    assert slice_cmd.exists()
    assert common_ps1.exists()

    poultry_text = poultry_ps1.read_text(encoding="utf-8")
    slice_text = slice_ps1.read_text(encoding="utf-8")
    poultry_cmd_text = poultry_cmd.read_text(encoding="utf-8")
    slice_cmd_text = slice_cmd.read_text(encoding="utf-8")
    common_text = common_ps1.read_text(encoding="utf-8")

    assert "command.log" in poultry_text
    assert "command.log" in slice_text
    assert "ExecutionPolicy Bypass" in poultry_cmd_text
    assert "ExecutionPolicy Bypass" in slice_cmd_text
    assert "Start-ProjectLog" in common_text
    assert "Finish-ProjectLog" in common_text
    assert "Write-ProjectException" in common_text
    assert "duration=" in common_text


def test_cli_help_appends_to_single_log_file(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    python_script = repo_root / "scripts" / "poultry_geo.py"
    log_file = tmp_path / "command.log"

    first = subprocess.run(
        [sys.executable, str(python_script), "--log-file", str(log_file), "--help"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        [sys.executable, str(python_script), "--log-file", str(log_file), "--help"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert log_file.exists()
    assert "[logging] Appending command output" in first.stdout
    assert "[logging] Appending command output" in second.stdout

    log_text = log_file.read_text(encoding="utf-8")
    assert "Command Log" in log_text
    assert "Major workflows" in log_text
    assert "Finished (exit=0" in log_text
    assert log_text.count("Command Log") == 2


def test_cli_no_log_skips_log_creation(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    python_script = repo_root / "scripts" / "poultry_geo.py"
    log_file = tmp_path / "command.log"

    subprocess.run(
        [sys.executable, str(python_script), "--no-log", "--log-file", str(log_file), "--help"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert not log_file.exists()


def test_slice_wrapper_logs_without_outer_wrapper(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    shell_script = repo_root / "scripts" / "slice_tiles.sh"
    log_file = tmp_path / "command.log"

    completed = subprocess.run(
        ["bash", str(shell_script), "--log-file", str(log_file), "--help"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert log_file.exists()
    assert "[logging] Appending command output" in completed.stdout

    log_text = log_file.read_text(encoding="utf-8")
    assert "slice_tiles.sh" in log_text
    assert "Command Log" in log_text
    assert "Finished (exit=0" in log_text


def test_generic_logged_wrapper_captures_output(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    wrapper = repo_root / "scripts" / "run_logged.sh"
    log_file = tmp_path / "command.log"

    completed = subprocess.run(
        [
            "bash",
            str(wrapper),
            "--log-file",
            str(log_file),
            "--",
            sys.executable,
            "-c",
            "print('hello from wrapper')",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert log_file.exists()
    assert "hello from wrapper" in completed.stdout

    log_text = log_file.read_text(encoding="utf-8")
    assert "Command Log" in log_text
    assert "hello from wrapper" in log_text
    assert "Finished (exit=0" in log_text
