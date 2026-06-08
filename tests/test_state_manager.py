from __future__ import annotations

import copy
import csv
import json
import random
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

torch = pytest.importorskip("torch")

from poultry_geo.checkpoint_manager import StateManager
from poultry_geo.optimizers import build_optimizer
from poultry_geo.updated_train import run_stateful_grid_search


class DummyScaler:
    def __init__(self, state: dict[str, float] | None = None) -> None:
        self._state = dict(state or {"scale": 128.0, "growth_tracker": 4.0})

    def state_dict(self) -> dict[str, float]:
        return dict(self._state)

    def load_state_dict(self, state_dict: dict[str, float]) -> None:
        self._state = dict(state_dict)


def _write_rgb(path: Path, *, size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    array[..., 0] = 64
    array[..., 1] = 128
    array[..., 2] = 192
    Image.fromarray(array, mode="RGB").save(path)


def _write_mask(path: Path, *, size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(size, dtype=np.uint8)
    mask[size[0] // 4 : (3 * size[0]) // 4, size[1] // 4 : (3 * size[1]) // 4] = 255
    Image.fromarray(mask, mode="L").save(path)


def _install_fake_resnet(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResNetBackbone(torch.nn.Module):
        def __init__(self, out_channels: int = 32) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, out_channels, kernel_size=3, stride=8, padding=1)
            self.bn = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.ReLU(inplace=True)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.relu(self.bn(self.conv(inputs)))

    monkeypatch.setattr(
        "poultry_geo.models._build_resnet50_backbone",
        lambda pretrained=False: (FakeResNetBackbone(), 32),
    )


def _make_experiment_config(*, optimizer: str = "adamw", precision: str = "fp32") -> dict[str, object]:
    return {
        "data": {
            "image_dir": "images",
            "mask_dir": "masks",
            "image_size": 32,
            "n_channels": 3,
            "batch_size": 1,
            "num_workers": 0,
        },
        "runtime": {
            "device": "cpu",
            "seed": 17,
            "output_dir": "runs/grid",
            "pretrained": False,
        },
        "grid": {
            "backbone": "resnet50",
            "optimizer": optimizer,
            "precision": precision,
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "freeze_backbone": False,
            "use_lora": False,
        },
    }


def _run_one_optimizer_step(model: torch.nn.Module, optimizer) -> None:
    inputs = torch.randn(2, 3, 32, 32)
    targets = torch.rand(2, 1, 32, 32)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.allclose(left, right)
        return
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert set(left) == set(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right


@pytest.mark.parametrize("optimizer_name", ["adamw", "adan", "sophia"])
def test_state_manager_creates_and_restores_run_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    optimizer_name: str,
) -> None:
    _install_fake_resnet(monkeypatch)
    from poultry_geo.models import UnifiedRSModel

    config = _make_experiment_config(optimizer=optimizer_name)
    root_dir = tmp_path / "trained_models"
    manager = StateManager(
        config=config,
        training_name="training_name=resume-demo",
        experiment_id="exp-001",
        root_dir=root_dir,
    )

    model = UnifiedRSModel(n_channels=3, backbone_name="resnet50", image_size=32, precision="fp32")
    optimizer = build_optimizer(
        optimizer_name,
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
        weight_decay=1e-2,
    )
    scaler = DummyScaler()

    run_state = manager.prepare(model, optimizer, scaler)

    assert run_state.status == "new"
    assert run_state.start_epoch == 0
    assert run_state.run_dir == root_dir / "resume-demo" / "exp-001"
    assert run_state.run_dir.exists()
    assert (run_state.run_dir / "config_history.yaml").exists()

    _run_one_optimizer_step(model, optimizer)
    saved_model_state = copy.deepcopy(model.state_dict())
    saved_optimizer_state = copy.deepcopy(optimizer.state_dict())
    saved_scaler_state = copy.deepcopy(scaler.state_dict())
    manager.save_checkpoint(model, optimizer, scaler, epoch=0)

    expected_torch = torch.rand(4)
    expected_numpy = np.random.rand(4)
    expected_python = [random.random() for _ in range(4)]

    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)

    resumed_model = UnifiedRSModel(n_channels=3, backbone_name="resnet50", image_size=32, precision="fp32")
    resumed_optimizer = build_optimizer(
        optimizer_name,
        [parameter for parameter in resumed_model.parameters() if parameter.requires_grad],
        lr=1e-3,
        weight_decay=1e-2,
    )
    resumed_scaler = DummyScaler({"scale": 1.0, "growth_tracker": 0.0})

    resumed_state = manager.prepare(resumed_model, resumed_optimizer, resumed_scaler)

    assert resumed_state.status == "resume"
    assert resumed_state.start_epoch == 1
    assert resumed_state.seed == run_state.seed
    assert resumed_state.config == config
    _assert_nested_equal(resumed_model.state_dict(), saved_model_state)
    _assert_nested_equal(resumed_optimizer.state_dict(), saved_optimizer_state)
    assert resumed_scaler.state_dict() == saved_scaler_state
    assert torch.allclose(torch.rand(4), expected_torch)
    assert np.allclose(np.random.rand(4), expected_numpy)
    assert [random.random() for _ in range(4)] == expected_python


def test_state_manager_rejects_config_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_resnet(monkeypatch)
    from poultry_geo.models import UnifiedRSModel

    base_config = _make_experiment_config()
    root_dir = tmp_path / "trained_models"
    manager = StateManager(
        config=base_config,
        training_name="training_name=drift-demo",
        experiment_id="exp-001",
        root_dir=root_dir,
    )
    model = UnifiedRSModel(n_channels=3, backbone_name="resnet50", image_size=32, precision="fp32")
    optimizer = build_optimizer(
        "adamw",
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
        weight_decay=1e-2,
    )
    manager.prepare(model, optimizer, DummyScaler())
    manager.save_checkpoint(model, optimizer, DummyScaler(), epoch=0)

    drifted_config = copy.deepcopy(base_config)
    drifted_config["grid"]["learning_rate"] = 5e-4
    drifted_manager = StateManager(
        config=drifted_config,
        training_name="drift-demo",
        experiment_id="exp-001",
        root_dir=root_dir,
    )

    with pytest.raises(ValueError, match="does not match"):
        drifted_manager.prepare(
            UnifiedRSModel(n_channels=3, backbone_name="resnet50", image_size=32, precision="fp32"),
            build_optimizer(
                "adamw",
                [
                    parameter
                    for parameter in UnifiedRSModel(
                        n_channels=3,
                        backbone_name="resnet50",
                        image_size=32,
                        precision="fp32",
                    ).parameters()
                    if parameter.requires_grad
                ],
                lr=1e-3,
                weight_decay=1e-2,
            ),
            DummyScaler(),
        )


def test_state_manager_appends_metrics_csv_without_overwrite(tmp_path: Path) -> None:
    config = _make_experiment_config()
    manager = StateManager(
        config=config,
        training_name="append-demo",
        experiment_id="exp-001",
        root_dir=tmp_path / "trained_models",
    )
    manager.append_metrics(epoch=1, loss=0.5, accuracy=0.75, evaluation_matrix=[[8, 1], [2, 9]])
    first_bytes = manager.metrics_path.read_bytes()

    manager.append_metrics(epoch=2, loss=0.25, accuracy=0.9, evaluation_matrix=[[9, 0], [1, 10]])
    second_bytes = manager.metrics_path.read_bytes()

    assert second_bytes.startswith(first_bytes)
    rows = list(csv.DictReader(manager.metrics_path.open("r", encoding="utf-8", newline="")))
    assert [int(row["epoch"]) for row in rows] == [1, 2]
    assert json.loads(rows[0]["evaluation_matrix_json"]) == [[8, 1], [2, 9]]
    assert json.loads(rows[1]["evaluation_matrix_json"]) == [[9, 0], [1, 10]]


def test_run_stateful_grid_search_writes_and_resumes_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_resnet(monkeypatch)

    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    _write_rgb(image_dir / "tile_0.png")
    _write_mask(mask_dir / "tile_0.png")

    config_path = tmp_path / "config.yaml"
    config = {
        "data": {
            "image_dir": image_dir.as_posix(),
            "mask_dir": mask_dir.as_posix(),
            "image_size": 32,
            "n_channels": 3,
            "batch_size": 1,
            "num_workers": 0,
        },
        "runtime": {
            "device": "cpu",
            "seed": 23,
            "output_dir": (tmp_path / "runs").as_posix(),
            "pretrained": False,
            "epochs": 1,
        },
        "grid_search": {
            "backbone": ["resnet50"],
            "optimizer": ["adamw"],
            "precision": ["bf16"],
            "learning_rate": [1e-3],
            "weight_decay": [1e-2],
            "freeze_backbone": [False],
            "use_lora": [False],
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    root_dir = tmp_path / "trained_models"
    first_results = run_stateful_grid_search(
        config_path=config_path,
        training_name="training_name=segmentation-demo",
        epochs=1,
        root_dir=root_dir,
    )

    assert len(first_results) == 1
    assert first_results.loc[0, "status"] == "ok"
    run_dir = root_dir / "segmentation-demo" / str(first_results.loc[0, "experiment_id"])
    assert (run_dir / "config_history.yaml").exists()
    assert (run_dir / "model_checkpoint.pt").exists()
    assert (run_dir / "training_metrics.csv").exists()
    summary_csv = root_dir / "segmentation-demo" / "grid_search_results.csv"
    assert summary_csv.exists()

    second_results = run_stateful_grid_search(
        config_path=config_path,
        training_name="segmentation-demo",
        epochs=2,
        root_dir=root_dir,
    )

    metrics = pd.read_csv(run_dir / "training_metrics.csv")
    assert list(metrics["epoch"]) == [1, 2]
    assert all(np.isfinite(metrics["loss"]))
    assert all(np.isfinite(metrics["accuracy"]))
    assert json.loads(metrics.loc[0, "evaluation_matrix_json"]) != []
    assert json.loads(metrics.loc[1, "evaluation_matrix_json"]) != []
    summary = pd.read_csv(summary_csv)
    assert len(summary) == 1
    assert int(summary.loc[0, "last_epoch"]) == 2
    assert second_results.loc[0, "status"] == "ok"
    checkpoint = torch.load(run_dir / "model_checkpoint.pt", map_location="cpu", weights_only=False)
    assert int(checkpoint["epoch"]) == 1
    assert checkpoint["scaler_state_dict"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_run_stateful_grid_search_fp16_persists_grad_scaler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_resnet(monkeypatch)

    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    _write_rgb(image_dir / "tile_0.png")
    _write_mask(mask_dir / "tile_0.png")

    config_path = tmp_path / "config.yaml"
    config = {
        "data": {
            "image_dir": image_dir.as_posix(),
            "mask_dir": mask_dir.as_posix(),
            "image_size": 32,
            "n_channels": 3,
            "batch_size": 1,
            "num_workers": 0,
        },
        "runtime": {
            "device": "cuda",
            "seed": 31,
            "output_dir": (tmp_path / "runs").as_posix(),
            "pretrained": False,
            "epochs": 1,
        },
        "grid_search": {
            "backbone": ["resnet50"],
            "optimizer": ["adamw"],
            "precision": ["fp16"],
            "learning_rate": [1e-3],
            "weight_decay": [1e-2],
            "freeze_backbone": [False],
            "use_lora": [False],
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    root_dir = tmp_path / "trained_models"
    results = run_stateful_grid_search(
        config_path=config_path,
        training_name="training_name=fp16-demo",
        epochs=1,
        root_dir=root_dir,
    )

    run_dir = root_dir / "fp16-demo" / str(results.loc[0, "experiment_id"])
    checkpoint = torch.load(run_dir / "model_checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["scaler_state_dict"] is not None
    assert checkpoint["scaler_state_dict"] != {}
