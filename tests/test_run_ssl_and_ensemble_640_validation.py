from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import scripts.run_ssl_and_ensemble_640_validation as runner


def test_run_stage_builds_expected_commands(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, *, cwd, check):
        calls.append(command)
        assert cwd == runner.ROOT
        assert check is True

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner.run_stage(
        "ssl", "scripts.run_ssl_al_ms_anchored_640_validation", "ssl.yaml", reuse_existing=True
    )
    runner.run_stage(
        "ensemble", "scripts.evaluate_obb_seed_ensemble_validation", "ensemble.yaml", reuse_existing=True
    )
    assert calls[0][3:] == ["--config", "ssl.yaml", "--phase", "development", "--reuse-existing"]
    assert calls[1][3:] == ["--config", "ensemble.yaml", "--reuse-existing"]


def test_script_compiles() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(Path(runner.__file__))],
        cwd=runner.ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
