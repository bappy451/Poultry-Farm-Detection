from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from poultry_geo.ssl_models import RemoteSensingMIMModel, build_nf4_quantization_config
from poultry_geo.training import SSLTrainer


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _TrackingSampler:
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


class _SimpleLoader(list):
    def __init__(self, batches: list[dict[str, object]], sampler: object | None = None) -> None:
        super().__init__(batches)
        self.sampler = sampler


def test_mim_model_returns_expected_shapes_and_mask_ratio() -> None:
    torch.manual_seed(7)
    model = RemoteSensingMIMModel(
        in_channels=4,
        image_size=64,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        precision="fp32",
        mask_ratio=0.75,
    )
    outputs = model.forward_with_intermediates(torch.randn(2, 4, 64, 64))

    assert outputs["adapted"].shape == (2, 3, 64, 64)
    assert outputs["tokens"].shape == (2, 64, 32)
    assert outputs["mask"].shape == (2, 64)
    assert outputs["mask"].dtype == torch.bool
    assert outputs["reconstructed_patches"].shape == (2, 64, 192)
    assert outputs["target_patches"].shape == (2, 64, 192)
    assert outputs["loss"].ndim == 0
    assert torch.isfinite(outputs["loss"])
    assert abs(float(outputs["mask"].float().mean()) - 0.75) < 0.02


def test_masked_only_loss_ignores_unmasked_patches() -> None:
    reconstruction = torch.tensor(
        [
            [[1.0, 1.0], [9.0, 9.0], [3.0, 3.0]],
        ]
    )
    target = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0], [8.0, 8.0]],
        ]
    )
    mask = torch.tensor([[False, True, False]])

    loss = RemoteSensingMIMModel.compute_masked_loss(reconstruction, target, mask)

    assert torch.isclose(loss, torch.tensor(64.0))


def test_nf4_mim_model_falls_back_and_keeps_only_qlora_trainable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("poultry_geo.ssl_models._bitsandbytes_supports_nf4", lambda: False)

    model = RemoteSensingMIMModel(
        in_channels=4,
        image_size=64,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        precision="nf4",
        quantization_config=build_nf4_quantization_config(),
    )

    backbone_lora = [
        parameter
        for name, parameter in model.backbone.named_parameters()
        if "lora_" in name
    ]
    backbone_base = [
        parameter
        for name, parameter in model.backbone.named_parameters()
        if "lora_" not in name
    ]

    assert model.requested_precision == "nf4"
    assert model.resolved_precision == "bf16"
    assert model.use_lora is True
    assert model.freeze_backbone is True
    assert backbone_lora
    assert all(parameter.requires_grad for parameter in backbone_lora)
    assert backbone_base
    assert all(not parameter.requires_grad for parameter in backbone_base)
    assert model.mask_token.requires_grad is True
    assert all(parameter.requires_grad for parameter in model.reconstruction_head.parameters())


def test_ssl_trainer_bf16_single_step_has_finite_loss_and_gradients() -> None:
    batch = {
        "image": torch.randn(2, 4, 64, 64, dtype=torch.float32),
        "path": ["a.npy", "b.npy"],
    }
    model = RemoteSensingMIMModel(
        in_channels=4,
        image_size=64,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        precision="bf16",
        mask_ratio=0.75,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    trainer = SSLTrainer(
        model,
        _SimpleLoader([batch]),
        optimizer,
        device="cpu",
        precision="bf16",
    )

    metrics = trainer.train_step(batch)

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_ssl_trainer_fp16_single_step_has_finite_loss_and_gradients() -> None:
    batch = {
        "image": torch.randn(2, 4, 64, 64, dtype=torch.float32),
        "path": ["a.npy", "b.npy"],
    }
    model = RemoteSensingMIMModel(
        in_channels=4,
        image_size=64,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        precision="fp16",
        mask_ratio=0.75,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    trainer = SSLTrainer(
        model,
        _SimpleLoader([batch]),
        optimizer,
        device="cuda",
        precision="fp16",
    )

    metrics = trainer.train_step(batch)

    assert trainer.grad_scaler.is_enabled() is True
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_ssl_trainer_fit_writes_checkpoints_metrics_and_sets_epoch(tmp_path: Path) -> None:
    batch = {
        "image": torch.randn(2, 4, 64, 64, dtype=torch.float32),
        "path": ["a.npy", "b.npy"],
    }
    sampler = _TrackingSampler()
    train_loader = _SimpleLoader([batch, batch], sampler=sampler)
    val_loader = _SimpleLoader([batch])
    model = RemoteSensingMIMModel(
        in_channels=4,
        image_size=64,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        precision="fp32",
        mask_ratio=0.75,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    trainer = SSLTrainer(
        model,
        train_loader,
        optimizer,
        val_loader=val_loader,
        device="cpu",
        precision="fp32",
        log_every=100,
    )

    summary = trainer.fit(epochs=1, output_dir=tmp_path / "mim-run")

    assert sampler.epochs == [0]
    assert (tmp_path / "mim-run" / "last.pt").exists()
    assert (tmp_path / "mim-run" / "best.pt").exists()
    metrics_path = tmp_path / "mim-run" / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(metrics["history"]) == 1
    assert summary.train_batches == 2
    assert summary.val_batches == 1
    assert summary.final_train_loss is not None
    assert summary.final_val_loss is not None


def test_mim_model_ddp_smoke() -> None:
    if dist.is_initialized():
        pytest.skip("Distributed process group is already initialized.")

    port = _find_free_port()
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=0,
        world_size=1,
    )
    try:
        model = RemoteSensingMIMModel(
            in_channels=4,
            image_size=64,
            patch_size=8,
            embed_dim=32,
            depth=1,
            num_heads=4,
            precision="fp32",
            mask_ratio=0.75,
        )
        wrapped = DistributedDataParallel(model)
        loss = wrapped(torch.randn(2, 4, 64, 64))

        assert loss.ndim == 0
        assert torch.isfinite(loss)
    finally:
        dist.destroy_process_group()
