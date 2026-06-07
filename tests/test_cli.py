from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import shutil

import pytest

from poultry_geo.cli import _resolve_torch_device_alias, build_parser


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


def test_cli_parses_inspect_backbone_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "inspect-backbone",
            "tmp/test_jpg",
            "--batch-size",
            "2",
            "--crop-size",
            "224",
            "--channels",
            "5",
            "--precision",
            "nf4",
            "--patch-size",
            "16",
            "--embed-dim",
            "192",
            "--depth",
            "12",
            "--num-heads",
            "3",
            "--gradient-checkpointing",
        ]
    )

    assert args.command == "inspect-backbone"
    assert args.data_root == "tmp/test_jpg"
    assert args.channels == 5
    assert args.precision == "nf4"
    assert args.embed_dim == 192
    assert args.depth == 12
    assert args.num_heads == 3
    assert args.gradient_checkpointing is True


def test_cli_parses_inspect_segmentation_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "inspect-segmentation",
            "tmp/test_jpg",
            "--batch-size",
            "2",
            "--crop-size",
            "256",
            "--channels",
            "4",
            "--precision",
            "bf16",
            "--patch-size",
            "16",
            "--embed-dim",
            "192",
            "--depth",
            "8",
            "--num-heads",
            "6",
            "--use-lora",
            "--freeze-backbone",
            "--lora-rank",
            "8",
            "--decoder-hidden-channels",
            "96",
        ]
    )

    assert args.command == "inspect-segmentation"
    assert args.data_root == "tmp/test_jpg"
    assert args.channels == 4
    assert args.precision == "bf16"
    assert args.embed_dim == 192
    assert args.depth == 8
    assert args.num_heads == 6
    assert args.use_lora is True
    assert args.freeze_backbone is True
    assert args.lora_rank == 8
    assert args.decoder_hidden_channels == 96


def test_cli_parses_train_mim_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-mim",
            "tmp/test_jpg",
            "runs/mim",
            "--batch-size",
            "4",
            "--crop-size",
            "224",
            "--channels",
            "4",
            "--precision",
            "nf4",
            "--patch-size",
            "16",
            "--embed-dim",
            "192",
            "--depth",
            "8",
            "--num-heads",
            "6",
            "--mask-ratio",
            "0.6",
            "--epochs",
            "3",
            "--learning-rate",
            "0.0005",
            "--weight-decay",
            "0.01",
            "--use-lora",
            "--freeze-backbone",
            "--lora-rank",
            "8",
            "--lora-alpha",
            "16",
        ]
    )

    assert args.command == "train-mim"
    assert args.data_root == "tmp/test_jpg"
    assert args.output_dir == "runs/mim"
    assert args.precision == "nf4"
    assert args.mask_ratio == 0.6
    assert args.epochs == 3
    assert args.learning_rate == 0.0005
    assert args.weight_decay == 0.01
    assert args.use_lora is True
    assert args.freeze_backbone is True
    assert args.lora_rank == 8
    assert args.lora_alpha == 16


def test_resolve_torch_device_alias_supports_gpu_name(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert _resolve_torch_device_alias("gpu") == torch.device("cuda")


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
    inspect_ps1 = repo_root / "scripts" / "inspect_backbone.ps1"
    inspect_cmd = repo_root / "scripts" / "inspect_backbone.cmd"
    inspect_sh = repo_root / "scripts" / "inspect_backbone.sh"
    inspect_seg_ps1 = repo_root / "scripts" / "inspect_segmentation.ps1"
    inspect_seg_cmd = repo_root / "scripts" / "inspect_segmentation.cmd"
    inspect_seg_sh = repo_root / "scripts" / "inspect_segmentation.sh"
    train_mim_ps1 = repo_root / "scripts" / "train_mim.ps1"
    train_mim_cmd = repo_root / "scripts" / "train_mim.cmd"
    train_mim_sh = repo_root / "scripts" / "train_mim.sh"
    common_ps1 = repo_root / "scripts" / "windows_common.ps1"

    assert poultry_ps1.exists()
    assert poultry_cmd.exists()
    assert slice_ps1.exists()
    assert slice_cmd.exists()
    assert inspect_ps1.exists()
    assert inspect_cmd.exists()
    assert inspect_sh.exists()
    assert inspect_seg_ps1.exists()
    assert inspect_seg_cmd.exists()
    assert inspect_seg_sh.exists()
    assert train_mim_ps1.exists()
    assert train_mim_cmd.exists()
    assert train_mim_sh.exists()
    assert common_ps1.exists()

    poultry_text = poultry_ps1.read_text(encoding="utf-8")
    slice_text = slice_ps1.read_text(encoding="utf-8")
    inspect_ps1_text = inspect_ps1.read_text(encoding="utf-8")
    inspect_cmd_text = inspect_cmd.read_text(encoding="utf-8")
    inspect_sh_text = inspect_sh.read_text(encoding="utf-8")
    inspect_seg_ps1_text = inspect_seg_ps1.read_text(encoding="utf-8")
    inspect_seg_cmd_text = inspect_seg_cmd.read_text(encoding="utf-8")
    inspect_seg_sh_text = inspect_seg_sh.read_text(encoding="utf-8")
    train_mim_ps1_text = train_mim_ps1.read_text(encoding="utf-8")
    train_mim_cmd_text = train_mim_cmd.read_text(encoding="utf-8")
    train_mim_sh_text = train_mim_sh.read_text(encoding="utf-8")
    poultry_cmd_text = poultry_cmd.read_text(encoding="utf-8")
    slice_cmd_text = slice_cmd.read_text(encoding="utf-8")
    common_text = common_ps1.read_text(encoding="utf-8")

    assert "command.log" in poultry_text
    assert "command.log" in slice_text
    assert "ExecutionPolicy Bypass" in poultry_cmd_text
    assert "ExecutionPolicy Bypass" in slice_cmd_text
    assert "ExecutionPolicy Bypass" in inspect_cmd_text
    assert "ExecutionPolicy Bypass" in inspect_seg_cmd_text
    assert "ExecutionPolicy Bypass" in train_mim_cmd_text
    assert "inspect-backbone" in inspect_ps1_text
    assert "inspect-backbone" in inspect_sh_text
    assert "inspect-segmentation" in inspect_seg_ps1_text
    assert "inspect-segmentation" in inspect_seg_sh_text
    assert "train-mim" in train_mim_ps1_text
    assert "train-mim" in train_mim_sh_text
    assert "Start-ProjectLog" in common_text
    assert "Finish-ProjectLog" in common_text
    assert "Write-ProjectException" in common_text
    assert "duration=" in common_text


def test_windows_common_keeps_native_stderr_as_command_output(tmp_path: Path) -> None:
    if not sys.platform.startswith("win"):
        pytest.skip("Windows-specific PowerShell wrapper test.")

    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell is not available.")

    repo_root = Path(__file__).resolve().parents[1]
    common_ps1 = repo_root / "scripts" / "windows_common.ps1"
    log_file = tmp_path / "command.log"
    driver_script = tmp_path / "stderr_smoke.ps1"
    python_executable = sys.executable.replace("'", "''")
    common_path = str(common_ps1).replace("'", "''")
    log_path = str(log_file).replace("'", "''")
    cwd_path = str(tmp_path).replace("'", "''")
    driver_script.write_text(
        "\n".join(
            [
                "Set-StrictMode -Version Latest",
                "$ErrorActionPreference = 'Stop'",
                f". '{common_path}'",
                f"$logFile = '{log_path}'",
                "$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()",
                "Start-ProjectLog -LogFile $logFile -CommandText 'stderr-smoke' -InvocationCwd '" + cwd_path + "'",
                "$exitCode = Invoke-ProjectNativeCommand -Executable '" + python_executable + "' -Arguments @('-c', \"import sys; sys.stderr.write('benign stderr\\\\n')\") -LogFile $logFile -CaptureToLog",
                "Finish-ProjectLog -LogFile $logFile -ExitCode $exitCode -Stopwatch $stopwatch",
                "exit $exitCode",
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [shell, "-ExecutionPolicy", "Bypass", "-File", str(driver_script)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "benign stderr" in completed.stdout
    assert "NativeCommandError" not in completed.stdout
    assert "Wrapper failed with an exception" not in completed.stdout
    assert log_file.exists()


def test_inspect_backbone_command_prints_shape_summary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    torch = pytest.importorskip("torch")

    import poultry_geo.cli as cli
    import poultry_geo.naip_data as naip_data
    import poultry_geo.ssl_models as ssl_models

    batch = {
        "image": torch.ones((2, 5, 224, 224), dtype=torch.float32),
        "path": ["a.jpg", "b.jpg"],
    }

    class _FakeLoader(list):
        pass

    def fake_get_dataloaders(*args, **kwargs):
        return _FakeLoader([batch]), _FakeLoader([batch])

    class FakeAdapter(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int = 3, precision: str = "fp32", bypass_rgb: bool = True):
            super().__init__()
            self.requested_precision = precision
            self.resolved_precision = "bf16" if precision == "nf4" else precision

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs[:, :3].to(dtype=torch.bfloat16)

    class FakeViT(torch.nn.Module):
        def __init__(self, *args, precision: str = "fp32", **kwargs):
            super().__init__()
            self.requested_precision = precision
            self.resolved_precision = "bf16" if precision == "nf4" else precision
            self.grid_size = (14, 14)

        def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.ones((inputs.shape[0], 196, 64), device=inputs.device, dtype=inputs.dtype)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.forward_features(inputs)

    monkeypatch.setattr(naip_data, "get_dataloaders", fake_get_dataloaders)
    monkeypatch.setattr(ssl_models, "BandAdapter", FakeAdapter)
    monkeypatch.setattr(ssl_models, "RemoteSensingViT", FakeViT)

    exit_code = cli.main(
        [
            "--no-log",
            "--no-progress",
            "inspect-backbone",
            "tmp/test_jpg",
            "--batch-size",
            "2",
            "--crop-size",
            "224",
            "--channels",
            "5",
            "--precision",
            "nf4",
            "--patch-size",
            "16",
            "--embed-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "input_shape=(2, 5, 224, 224)" in stdout
    assert "adapted_shape=(2, 3, 224, 224)" in stdout
    assert "token_shape=(2, 196, 64)" in stdout
    assert "patch_grid=(14, 14)" in stdout
    assert "requested_precision=nf4" in stdout
    assert "resolved_precision=bf16" in stdout


def test_inspect_segmentation_command_prints_mask_summary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    torch = pytest.importorskip("torch")

    import poultry_geo.cli as cli
    import poultry_geo.naip_data as naip_data
    import poultry_geo.ssl_models as ssl_models

    batch = {
        "image": torch.ones((2, 4, 256, 256), dtype=torch.float32),
        "path": ["a.jpg", "b.jpg"],
    }

    class _FakeLoader(list):
        pass

    def fake_get_dataloaders(*args, **kwargs):
        return _FakeLoader([batch]), _FakeLoader([batch])

    class FakeSegmentationModel(torch.nn.Module):
        def __init__(
            self,
            *,
            in_channels: int,
            image_size: int,
            patch_size: int,
            embed_dim: int,
            depth: int,
            num_heads: int,
            precision: str = "fp32",
            use_lora: bool = False,
            freeze_backbone: bool = False,
            **kwargs,
        ):
            super().__init__()
            self.requested_precision = precision
            self.resolved_precision = "bf16" if precision == "nf4" else precision
            self.use_lora = use_lora
            self.freeze_backbone = freeze_backbone

        def forward_with_intermediates(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
            adapted = inputs[:, :3].to(dtype=torch.bfloat16)
            tokens = torch.ones((inputs.shape[0], 256, 64), device=inputs.device, dtype=torch.bfloat16)
            mask = torch.ones((inputs.shape[0], 1, 256, 256), device=inputs.device, dtype=torch.bfloat16)
            return {
                "adapted": adapted,
                "tokens": tokens,
                "mask": mask,
            }

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.forward_with_intermediates(inputs)["mask"]

    monkeypatch.setattr(naip_data, "get_dataloaders", fake_get_dataloaders)
    monkeypatch.setattr(ssl_models, "RemoteSensingSegmentationModel", FakeSegmentationModel)

    exit_code = cli.main(
        [
            "--no-log",
            "--no-progress",
            "inspect-segmentation",
            "tmp/test_jpg",
            "--batch-size",
            "2",
            "--crop-size",
            "256",
            "--channels",
            "4",
            "--precision",
            "nf4",
            "--patch-size",
            "16",
            "--embed-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--use-lora",
            "--freeze-backbone",
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "input_shape=(2, 4, 256, 256)" in stdout
    assert "adapted_shape=(2, 3, 256, 256)" in stdout
    assert "token_shape=(2, 256, 64)" in stdout
    assert "mask_shape=(2, 1, 256, 256)" in stdout
    assert "requested_precision=nf4" in stdout
    assert "resolved_precision=bf16" in stdout
    assert "use_lora=True" in stdout
    assert "freeze_backbone=True" in stdout


def test_train_mim_command_prints_training_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    import poultry_geo.cli as cli
    import poultry_geo.naip_data as naip_data
    import poultry_geo.ssl_models as ssl_models
    import poultry_geo.training as training

    batch = {
        "image": torch.ones((2, 4, 224, 224), dtype=torch.float32),
        "path": ["a.jpg", "b.jpg"],
    }
    captured: dict[str, object] = {}

    class _FakeLoader(list):
        pass

    def fake_get_dataloaders(*args, **kwargs):
        captured["loader_precision"] = kwargs["precision"]
        return _FakeLoader([batch]), _FakeLoader([batch])

    class FakeMIMModel(torch.nn.Module):
        def __init__(
            self,
            *,
            in_channels: int,
            image_size: int,
            patch_size: int,
            embed_dim: int,
            depth: int,
            num_heads: int,
            precision: str = "fp32",
            mask_ratio: float = 0.75,
            use_lora: bool = False,
            freeze_backbone: bool = False,
            quantization_config=None,
            **kwargs,
        ):
            super().__init__()
            self.requested_precision = precision
            self.resolved_precision = "bf16" if precision == "nf4" else precision
            self.use_lora = use_lora or precision == "nf4"
            self.freeze_backbone = freeze_backbone or precision == "nf4"
            self.parameter = torch.nn.Parameter(torch.zeros(1))
            captured["quantization_config"] = quantization_config

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.parameter.sum() + inputs.mean() * 0.0 + torch.tensor(0.25, device=inputs.device)

    class FakeTrainer:
        def __init__(
            self,
            model: torch.nn.Module,
            train_loader,
            optimizer,
            *,
            val_loader=None,
            device=None,
            precision: str = "fp32",
            scheduler=None,
            grad_clip_norm=None,
            log_every: int = 10,
        ):
            captured["trainer_precision"] = precision
            captured["trainer_device"] = str(device)

        def fit(self, epochs: int, output_dir: str | Path):
            return training.SSLTrainingSummary(
                output_dir=str(output_dir),
                epochs=epochs,
                train_batches=1,
                val_batches=1,
                final_train_loss=0.25,
                final_val_loss=0.2,
            )

    monkeypatch.setattr(naip_data, "get_dataloaders", fake_get_dataloaders)
    monkeypatch.setattr(ssl_models, "RemoteSensingMIMModel", FakeMIMModel)
    monkeypatch.setattr(training, "SSLTrainer", FakeTrainer)

    exit_code = cli.main(
        [
            "--no-log",
            "--no-progress",
            "train-mim",
            "tmp/test_jpg",
            str(tmp_path / "mim-run"),
            "--batch-size",
            "2",
            "--crop-size",
            "224",
            "--channels",
            "4",
            "--precision",
            "nf4",
            "--patch-size",
            "16",
            "--embed-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--epochs",
            "1",
            "--use-lora",
            "--freeze-backbone",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert captured["loader_precision"] == "bf16"
    assert captured["trainer_precision"] == "nf4"
    assert captured["trainer_device"] == "cpu"
    quantization_config = captured["quantization_config"]
    assert getattr(quantization_config, "load_in_4bit", False) is True
    assert str(getattr(quantization_config, "bnb_4bit_quant_type", "")).lower() == "nf4"

    stdout = capsys.readouterr().out
    assert "train-mim" in stdout
    assert "epochs=1" in stdout
    assert "final_train_loss=0.25" in stdout
    assert "final_val_loss=0.2" in stdout
    assert "requested_precision=nf4" in stdout


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
