from __future__ import annotations

from pathlib import Path

import pytest

from poultry_geo.command_logging import LoggingConfig, command_log_context


def test_command_log_records_full_traceback_and_duration(tmp_path: Path) -> None:
    log_file = tmp_path / "command.log"

    with pytest.raises(RuntimeError, match="diagnostic detail"):
        with command_log_context(
            ["example"],
            config=LoggingConfig(enabled=True, log_file=log_file),
            program_name="test-command",
        ):
            raise RuntimeError("diagnostic detail")

    log_text = log_file.read_text(encoding="utf-8")
    assert "Command failed with an unhandled exception" in log_text
    assert "Traceback (most recent call last):" in log_text
    assert 'raise RuntimeError("diagnostic detail")' in log_text
    assert "RuntimeError: diagnostic detail" in log_text
    assert "duration=00:00:" in log_text
    assert "elapsed_seconds=" in log_text
    assert "Finished (exit=1" in log_text


def test_command_log_success_records_duration(tmp_path: Path) -> None:
    log_file = tmp_path / "command.log"

    with command_log_context(
        ["example"],
        config=LoggingConfig(enabled=True, log_file=log_file),
        program_name="test-command",
    ):
        print("completed")

    log_text = log_file.read_text(encoding="utf-8")
    assert "completed" in log_text
    assert "duration=00:00:" in log_text
    assert "elapsed_seconds=" in log_text
    assert "Finished (exit=0" in log_text
