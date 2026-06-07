from __future__ import annotations

import io

import pytest

from poultry_geo.progress import CommandProgress, command_progress_context


def test_determinate_progress_renders_percentage_and_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    stream = io.StringIO()
    progress = CommandProgress("slice", stream=stream, refresh_seconds=60)

    progress.start()
    progress.update(3, 4, detail="tile.tif")
    progress.finish(success=True)

    output = stream.getvalue()
    assert "75%" in output
    assert "3/4" in output
    assert "tile.tif" in output
    assert "100%" in output
    assert "\r" in output


def test_failed_command_progress_never_claims_completion() -> None:
    stream = io.StringIO()

    with pytest.raises(RuntimeError):
        with command_progress_context(["slice"], stream=stream, refresh_seconds=60):
            raise RuntimeError("failed")

    output = stream.getvalue()
    assert "FAILED" in output
    assert "100%" not in output


def test_no_progress_disables_output() -> None:
    stream = io.StringIO()

    with command_progress_context(["slice", "--no-progress"], stream=stream):
        pass

    assert stream.getvalue() == ""


def test_progress_overwrites_same_terminal_line() -> None:
    stream = io.StringIO()
    progress = CommandProgress("slice", stream=stream, refresh_seconds=60)

    progress.start()
    progress.update(1, 2, detail="first.tif")
    progress.update(2, 2, detail="second.tif")
    progress.finish(success=True)

    output = stream.getvalue()
    assert output.count("\r") == 3
    assert output.count("\n") == 1
