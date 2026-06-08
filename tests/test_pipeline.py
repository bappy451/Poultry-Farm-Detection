from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from poultry_geo.grid_search import expand_grid_search_configs, load_grid_search_config, profile_single_step, run_grid_search
from poultry_geo.models import BandAdapter, UnifiedRSModel
from poultry_geo.optimizers import Adan, SophiaG, build_optimizer
from poultry_geo.pipeline import load_pipeline_config, run_pipeline


def test_load_pipeline_config_parses_steps(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "working_dir": "runs/example",
                "geotiff_dir": "data/geotiffs",
                "model_path": "weights/best.pt",
                "steps": {
                    "infer_yolo": True,
                    "sam_refine": True,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)

    assert config.working_dir == "runs/example"
    assert config.steps is not None
    assert config.steps.infer_yolo is True
    assert config.steps.sam_refine is True
    assert config.steps.cluster_farms is True


def test_run_pipeline_writes_expected_outputs(tmp_path: Path, monkeypatch) -> None:
    working_dir = tmp_path / "run"
    geotiff_dir = tmp_path / "geotiffs"
    geotiff_dir.mkdir()
    (geotiff_dir / "tile_1.tif").write_bytes(b"fake")

    config_path = tmp_path / "pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "working_dir": str(working_dir),
                "geotiff_dir": str(geotiff_dir),
                "model_path": "weights/best.pt",
                "steps": {
                    "slice_export": True,
                    "create_boundaries": True,
                    "infer_yolo": True,
                    "centroid_suppress": True,
                    "mask_nms": True,
                    "cluster_farms": True,
                    "export_polygons": True,
                    "sam_refine": False,
                    "aggregate_counties": False,
                    "runoff_screening": False,
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_slice_geotiff_to_files(geotiff_path, output_dir, tile_size, stride, to_jpeg):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "slice_0_0.tif"
        out.write_bytes(b"tile")
        return [out]

    def fake_create_boundaries(geotiff_dir_arg, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "tile_1.shp"
        out.write_text("shape", encoding="utf-8")
        return [out]

    def fake_infer(**kwargs):
        output_csv = Path(kwargs["output_csv"])
        pd.DataFrame(
            [
                {
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "confidence": 0.9,
                    "mask_points_gis": json.dumps([[0, 0], [1, 0], [1, 1], [0, 1]]),
                    "class": 0,
                    "tile_name": "tile_1.tif",
                    "area_geo": 1.0,
                }
            ]
        ).to_csv(output_csv, index=False)
        return output_csv

    def fake_centroid(df, distance_threshold, geometry_column):
        return df.copy()

    def fake_mask_nms(df, iou_threshold, geometry_column):
        return df.copy()

    def fake_detect_farms(df, eps_house, min_samples, eps_merge):
        return pd.DataFrame(
            [{"merged_farm_id": 0, "longitude": 0.0, "latitude": 0.0, "num_houses": 1}]
        )

    def fake_export_polygons(csv_path, output_dir, geometry_column, reference_crs):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "polygon_shapes.shp"
        out.write_text("shape", encoding="utf-8")
        return out

    monkeypatch.setattr("poultry_geo.pipeline.slice_geotiff_to_files", fake_slice_geotiff_to_files)
    monkeypatch.setattr("poultry_geo.pipeline.create_boundary_shapefiles_from_geotiffs", fake_create_boundaries)
    monkeypatch.setattr("poultry_geo.pipeline.run_yolo_inference", fake_infer)
    monkeypatch.setattr("poultry_geo.pipeline.centroid_clustering_suppression", fake_centroid)
    monkeypatch.setattr("poultry_geo.pipeline.mask_nms_by_area", fake_mask_nms)
    monkeypatch.setattr("poultry_geo.pipeline.detect_poultry_farms", fake_detect_farms)
    monkeypatch.setattr("poultry_geo.pipeline.save_polygon_shapefile_from_csv", fake_export_polygons)

    outputs = run_pipeline(load_pipeline_config(config_path))

    assert Path(outputs["summary_json"]).exists()
    assert Path(outputs["final_detection_csv"]).exists()
    assert Path(outputs["farm_centroids_csv"]).exists()
    assert Path(outputs["polygon_shapefile"]).exists()
    assert "slice_export" in outputs


def test_run_pipeline_can_aggregate_and_compute_runoff(tmp_path: Path, monkeypatch) -> None:
    working_dir = tmp_path / "run"
    geotiff_dir = tmp_path / "geotiffs"
    geotiff_dir.mkdir()
    (geotiff_dir / "tile_1.tif").write_bytes(b"fake")

    precip_csv = tmp_path / "precip.csv"
    precip_csv.write_text(
        "county,jan_mm,feb_mm\nA,100,50\n",
        encoding="utf-8",
    )
    adjustment_csv = tmp_path / "adjustments.csv"
    adjustment_csv.write_text(
        "adjustment_factor\n0.9\n1.0\n1.1\n",
        encoding="utf-8",
    )
    nass_csv = tmp_path / "nass.csv"
    nass_csv.write_text(
        "county,nass_farms\nA,1\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "working_dir": str(working_dir),
                "geotiff_dir": str(geotiff_dir),
                "model_path": "weights/best.pt",
                "precipitation_csv": str(precip_csv),
                "nass_count_csv": str(nass_csv),
                "adjustment_factors_csv": str(adjustment_csv),
                "steps": {
                    "slice_export": False,
                    "create_boundaries": False,
                    "infer_yolo": True,
                    "centroid_suppress": False,
                    "mask_nms": False,
                    "cluster_farms": False,
                    "export_polygons": False,
                    "sam_refine": False,
                    "aggregate_counties": True,
                    "runoff_screening": True,
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_infer(**kwargs):
        output_csv = Path(kwargs["output_csv"])
        pd.DataFrame(
            [
                {
                    "county": "A",
                    "area_geo": 1000.0,
                    "confidence": 0.9,
                    "mask_points_gis": json.dumps([[0, 0], [1, 0], [1, 1], [0, 1]]),
                    "class": 0,
                    "tile_name": "tile_1.tif",
                }
            ]
        ).to_csv(output_csv, index=False)
        return output_csv

    monkeypatch.setattr("poultry_geo.pipeline.run_yolo_inference", fake_infer)

    outputs = run_pipeline(load_pipeline_config(config_path))

    assert Path(outputs["county_summary_csv"]).exists()
    assert Path(outputs["county_runoff_csv"]).exists()
    assert "bootstrap_summary" in outputs


def _torch():
    return pytest.importorskip("torch")


def _write_rgb(path: Path, *, size: tuple[int, int] = (32, 32), channels: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((size[0], size[1], channels), dtype=np.uint8)
    for channel in range(channels):
        array[..., channel] = 40 * (channel + 1)
    if channels == 4:
        mode = "RGBA"
    else:
        mode = "RGB"
    Image.fromarray(array, mode=mode).save(path)


def _write_mask(path: Path, *, size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(size, dtype=np.uint8)
    mask[size[0] // 4 : (3 * size[0]) // 4, size[1] // 4 : (3 * size[1]) // 4] = 255
    Image.fromarray(mask, mode="L").save(path)


def _install_fake_timm(monkeypatch: pytest.MonkeyPatch, torch_module) -> None:
    class FakeAttention(torch_module.nn.Module):
        def __init__(self, embed_dim: int) -> None:
            super().__init__()
            self.qkv = torch_module.nn.Linear(embed_dim, embed_dim * 3)
            self.proj = torch_module.nn.Linear(embed_dim, embed_dim)

        def forward(self, inputs):
            outputs = self.qkv(inputs)[..., : inputs.shape[-1]]
            return self.proj(outputs)

    class FakeVitBlock(torch_module.nn.Module):
        def __init__(self, embed_dim: int) -> None:
            super().__init__()
            self.attn = FakeAttention(embed_dim)
            self.ffn = torch_module.nn.Linear(embed_dim, embed_dim)

        def forward(self, inputs):
            return inputs + self.ffn(self.attn(inputs))

    class FakeTimmViT(torch_module.nn.Module):
        def __init__(self, embed_dim: int = 48, patch_size: int = 16) -> None:
            super().__init__()
            self.embed_dim = embed_dim
            self.num_features = embed_dim
            self.num_prefix_tokens = 1
            self.patch_embed = types.SimpleNamespace(
                patch_size=(patch_size, patch_size),
                grid_size=(4, 4),
            )
            self.token_proj = torch_module.nn.Linear(3, embed_dim)
            self.blocks = torch_module.nn.ModuleList([FakeVitBlock(embed_dim)])

        def forward_features(self, inputs):
            pooled = torch_module.nn.functional.avg_pool2d(inputs, kernel_size=16, stride=16)
            tokens = pooled.flatten(2).transpose(1, 2)
            tokens = self.token_proj(tokens)
            cls = tokens.mean(dim=1, keepdim=True)
            outputs = torch_module.cat([cls, tokens], dim=1)
            for block in self.blocks:
                outputs = block(outputs)
            return outputs

    fake_timm = types.SimpleNamespace(create_model=lambda *args, **kwargs: FakeTimmViT())
    monkeypatch.setitem(sys.modules, "timm", fake_timm)


def _install_fake_resnet(monkeypatch: pytest.MonkeyPatch, torch_module) -> None:
    class FakeResNetBackbone(torch_module.nn.Module):
        def __init__(self, out_channels: int = 64) -> None:
            super().__init__()
            self.conv = torch_module.nn.Conv2d(3, out_channels, kernel_size=3, stride=8, padding=1)
            self.bn = torch_module.nn.BatchNorm2d(out_channels)
            self.relu = torch_module.nn.ReLU(inplace=True)

        def forward(self, inputs):
            return self.relu(self.bn(self.conv(inputs)))

    monkeypatch.setattr(
        "poultry_geo.models._build_resnet50_backbone",
        lambda pretrained=False: (FakeResNetBackbone(), 64),
    )


def test_load_grid_search_config_and_expand_unique_configs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data:
  image_dir: images
  mask_dir: masks
  image_size: 64
  n_channels: 3
  batch_size: 2
  num_workers: 0
runtime:
  device: cpu
  seed: 13
  output_dir: runs/grid
grid_search:
  backbone: [resnet50, resnet50]
  optimizer: [adamw, sophia]
  precision: [fp32, bf16]
  learning_rate: [0.001]
  weight_decay: [0.01]
  freeze_backbone: [false]
  use_lora: [false, false]
""".strip(),
        encoding="utf-8",
    )

    config = load_grid_search_config(config_path)
    experiments = expand_grid_search_configs(config)

    assert config["runtime"]["pretrained"] is False
    assert len(experiments) == 4
    assert all(experiment["data"]["image_size"] == 64 for experiment in experiments)
    assert {experiment["grid"]["optimizer"] for experiment in experiments} == {"adamw", "sophia"}
    assert {experiment["grid"]["precision"] for experiment in experiments} == {"fp32", "bf16"}


def test_band_adapter_projects_n_band_inputs_to_rgb() -> None:
    torch = _torch()
    adapter = BandAdapter(in_channels=5, out_channels=3)

    outputs = adapter(torch.randn(2, 5, 32, 32))

    assert outputs.shape == (2, 3, 32, 32)
    assert outputs.device.type == "cpu"


def test_unified_rs_model_routes_vit_tokens_back_to_spatial_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _torch()
    _install_fake_timm(monkeypatch, torch)

    model = UnifiedRSModel(
        n_channels=5,
        backbone_name="vit_tiny_patch16_224",
        image_size=64,
        precision="fp32",
    )
    inputs = torch.randn(2, 5, 64, 64)

    features = model.forward_features(inputs)
    logits = model(inputs)

    assert features.shape == (2, 48, 4, 4)
    assert logits.shape == (2, 1, 64, 64)
    assert logits.device.type == "cpu"


def test_unified_rs_model_passes_requested_image_size_to_timm(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _torch()
    captured_kwargs = {}

    class FakeTimmViT(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_dim = 32
            self.num_features = 32
            self.num_prefix_tokens = 1
            self.patch_embed = types.SimpleNamespace(
                patch_size=(16, 16),
                grid_size=(4, 5),
            )

        def forward_features(self, inputs):
            batch_size = inputs.shape[0]
            return torch.zeros(batch_size, 1 + 20, 32, dtype=inputs.dtype, device=inputs.device)

    def fake_create_model(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return FakeTimmViT()

    monkeypatch.setitem(sys.modules, "timm", types.SimpleNamespace(create_model=fake_create_model))

    model = UnifiedRSModel(
        n_channels=3,
        backbone_name="vit_tiny_patch16_224",
        image_size=(64, 80),
        precision="fp32",
    )
    logits = model(torch.randn(1, 3, 64, 80))

    assert captured_kwargs["img_size"] == (64, 80)
    assert captured_kwargs["global_pool"] == ""
    assert logits.shape == (1, 1, 64, 80)


def test_unified_rs_model_routes_resnet_backbone_to_spatial_map(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _torch()
    _install_fake_resnet(monkeypatch, torch)

    model = UnifiedRSModel(
        n_channels=4,
        backbone_name="resnet50",
        image_size=64,
        precision="fp32",
    )
    inputs = torch.randn(2, 4, 64, 64)

    features = model.forward_features(inputs)
    logits = model(inputs)

    assert features.shape == (2, 64, 8, 8)
    assert logits.shape == (2, 1, 64, 64)


def test_unified_rs_model_freezes_vit_backbone_without_lora(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _torch()
    _install_fake_timm(monkeypatch, torch)

    model = UnifiedRSModel(
        n_channels=3,
        backbone_name="vit_tiny_patch16_224",
        image_size=64,
        precision="fp32",
        freeze_backbone=True,
        use_lora=False,
    )

    backbone_params = list(model.backbone.parameters())
    adapter_params = list(model.adapter.parameters())
    decoder_params = list(model.decoder.parameters())

    assert backbone_params
    assert all(parameter.requires_grad is False for parameter in backbone_params)
    assert any(parameter.requires_grad for parameter in adapter_params)
    assert any(parameter.requires_grad for parameter in decoder_params)


def test_unified_rs_model_keeps_lora_trainable_when_backbone_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _torch()
    _install_fake_timm(monkeypatch, torch)

    model = UnifiedRSModel(
        n_channels=3,
        backbone_name="vit_tiny_patch16_224",
        image_size=64,
        precision="fp32",
        freeze_backbone=True,
        use_lora=True,
        lora_rank=4,
    )

    lora_params = [
        parameter
        for name, parameter in model.backbone.named_parameters()
        if "lora_" in name
    ]
    frozen_backbone_params = [
        parameter
        for name, parameter in model.backbone.named_parameters()
        if "lora_" not in name
    ]

    assert lora_params
    assert all(parameter.requires_grad for parameter in lora_params)
    assert all(parameter.requires_grad is False for parameter in frozen_backbone_params)


def test_build_optimizer_supports_adamw_adan_and_sophia() -> None:
    torch = _torch()

    def _step(name: str):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))
        optimizer = build_optimizer(name, [parameter], lr=1e-2, weight_decay=1e-2)
        loss = (parameter.pow(2)).sum()
        loss.backward()
        optimizer.step()
        return optimizer, parameter

    adamw, adamw_param = _step("adamw")
    adan, adan_param = _step("adan")
    sophia, sophia_param = _step("sophia")

    assert isinstance(adamw, torch.optim.AdamW)
    assert isinstance(adan, Adan)
    assert isinstance(sophia, SophiaG)
    assert torch.isfinite(adamw_param).all()
    assert torch.isfinite(adan_param).all()
    assert torch.isfinite(sophia_param).all()
    adan_state = next(iter(adan.state.values()))
    sophia_state = next(iter(sophia.state.values()))
    assert {"exp_avg", "exp_avg_sq", "exp_avg_diff"} <= set(adan_state)
    assert {"exp_avg", "hessian"} <= set(sophia_state)


def test_profile_single_step_bf16_runs_without_grad_scaler(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _torch()
    _install_fake_resnet(monkeypatch, torch)

    model = UnifiedRSModel(
        n_channels=3,
        backbone_name="resnet50",
        image_size=64,
        precision="bf16",
    )
    optimizer = build_optimizer("adamw", model.parameters(), lr=1e-3, weight_decay=1e-2)
    batch = {
        "image": torch.randn(2, 3, 64, 64),
        "mask": torch.randint(0, 2, (2, 1, 64, 64), dtype=torch.float32),
    }

    result = profile_single_step(model, batch, optimizer, device="cpu", precision="bf16")

    assert result["used_grad_scaler"] is False
    assert result["resolved_precision"] == "bf16"
    assert np.isfinite(result["loss"])


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="CUDA is not available")
def test_profile_single_step_fp16_uses_grad_scaler(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _torch()
    _install_fake_resnet(monkeypatch, torch)

    model = UnifiedRSModel(
        n_channels=3,
        backbone_name="resnet50",
        image_size=64,
        precision="fp16",
    )
    optimizer = build_optimizer("adamw", model.parameters(), lr=1e-3, weight_decay=1e-2)
    batch = {
        "image": torch.randn(2, 3, 64, 64),
        "mask": torch.randint(0, 2, (2, 1, 64, 64), dtype=torch.float32),
    }

    result = profile_single_step(model, batch, optimizer, device="cuda", precision="fp16")

    assert result["used_grad_scaler"] is True
    assert result["resolved_precision"] == "fp16"
    assert np.isfinite(result["loss"])


def test_run_grid_search_writes_results_csv_and_skips_invalid_resnet_lora(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _torch()
    _install_fake_timm(monkeypatch, torch)
    _install_fake_resnet(monkeypatch, torch)

    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    _write_rgb(image_dir / "a" / "tile_0.png", size=(32, 32))
    _write_rgb(image_dir / "b" / "tile_1.png", size=(32, 32))
    _write_mask(mask_dir / "a" / "tile_0.png", size=(32, 32))
    _write_mask(mask_dir / "b" / "tile_1.png", size=(32, 32))

    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "runs"
    config_path.write_text(
        f"""
data:
  image_dir: {image_dir.as_posix()}
  mask_dir: {mask_dir.as_posix()}
  image_size: 32
  n_channels: 3
  batch_size: 1
  num_workers: 0
runtime:
  device: cpu
  seed: 7
  output_dir: {output_dir.as_posix()}
  pretrained: false
grid_search:
  backbone: [resnet50, vit_tiny_patch16_224]
  optimizer: [adamw]
  precision: [fp32]
  learning_rate: [0.001]
  weight_decay: [0.01]
  freeze_backbone: [true]
  use_lora: [true]
""".strip(),
        encoding="utf-8",
    )

    results = run_grid_search(config_path)
    results_csv = output_dir / "grid_search_results.csv"

    assert results_csv.exists()
    assert len(results) == 2
    written = pd.read_csv(results_csv)
    assert len(written) == 2

    skipped = written.loc[written["backbone"] == "resnet50"].iloc[0]
    ok = written.loc[written["backbone"] == "vit_tiny_patch16_224"].iloc[0]
    assert skipped["status"] == "skipped"
    assert "LoRA" in skipped["reason"]
    assert ok["status"] == "ok"
    assert np.isfinite(ok["loss"])
