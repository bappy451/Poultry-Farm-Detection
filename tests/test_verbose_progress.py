from __future__ import annotations

import io
import pytest
from poultry_geo.progress import CommandProgress

def test_verbose_progress_renders_extra_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    stream = io.StringIO()
    # Assume we add a verbose flag to CommandProgress
    progress = CommandProgress("train", stream=stream, refresh_seconds=60, verbose=True)

    progress.start()
    # Assume detail can be a dict or a formatted string with keys
    progress.update(1, 10, detail="loss=0.5000, acc=0.8000, lr=0.0010")
    progress.finish(success=True)

    output = stream.getvalue()
    assert "loss=0.5000" in output
    assert "acc=0.8000" in output
    assert "lr=0.0010" in output
    # If we implement multi-line, check for that. 
    # For now, let's just ensure it's in the output.
