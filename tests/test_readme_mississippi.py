from __future__ import annotations

from pathlib import Path


def test_readme_documents_mississippifm_workflow() -> None:
    readme = (Path(__file__).resolve().parents[1] / "Readme.md").read_text(encoding="utf-8")

    assert "MississippiFM" in readme
    assert "known geographic overlap" in readme
    assert "mississippi-pretrain" in readme
    assert "mississippi-eval-probes" in readme
    assert "mississippi-eval-benchmark" in readme
    assert "create_adjacency_index.py" in readme
    assert "citation" in readme.lower()
