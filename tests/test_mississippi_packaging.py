from __future__ import annotations

from pathlib import Path
import tomllib


def test_pyproject_declares_mississippi_dependencies() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    dependencies = payload["project"]["dependencies"]
    ml_dependencies = payload["project"]["optional-dependencies"]["ml"]
    scripts = payload["project"]["scripts"]
    joined = "\n".join(dependencies + ml_dependencies).lower()

    assert "scikit-learn" in joined
    assert "timm" in joined
    assert "mmdet" in joined or "mmengine" in joined or "mmcv" in joined
    assert "mississippi-create-adjacency" in scripts
    assert "mississippi-pretrain" in scripts
    assert "mississippi-eval-probes" in scripts
    assert "mississippi-eval-benchmark" in scripts
