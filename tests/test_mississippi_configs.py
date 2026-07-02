from __future__ import annotations

from pathlib import Path

from poultry_geo.evaluate_probes import load_probe_config
from poultry_geo.geographic_benchmark import load_geographic_benchmark_config
from poultry_geo.train_mississippifm import load_mississippifm_config


def test_mississippi_example_configs_exist_and_parse() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pretrain_config = repo_root / "configs" / "mississippi_pretrain.yaml"
    probe_config = repo_root / "configs" / "mississippi_probes.yaml"
    benchmark_config = repo_root / "configs" / "mississippi_benchmark.yaml"

    assert pretrain_config.exists()
    assert probe_config.exists()
    assert benchmark_config.exists()

    loaded_pretrain = load_mississippifm_config(pretrain_config)
    loaded_probes = load_probe_config(probe_config)
    loaded_benchmark = load_geographic_benchmark_config(benchmark_config)

    assert loaded_pretrain.model_size in {"vit_s", "vit_b", "vit_l"}
    assert loaded_probes.tasks
    assert loaded_benchmark.tasks
    assert any(task.task_type == "detection" for task in loaded_probes.tasks)
    assert any(task.task_type == "retrieval" for task in loaded_probes.tasks)
    assert any(task.task_type == "context_augmentation" for task in loaded_probes.tasks)
    assert any(task.task_type == "detection" for task in loaded_benchmark.tasks)
