from __future__ import annotations

import socket
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from poultry_geo.ssl_models import BandAdapter, RemoteSensingViT


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", 0))
        except OSError as exc:
            pytest.skip(f"Local TCP sockets are unavailable for DDP smoke test: {exc}")
        return int(sock.getsockname()[1])


def test_band_adapter_projects_arbitrary_band_count_to_rgb() -> None:
    adapter = BandAdapter(in_channels=5, out_channels=3, precision="fp32")
    inputs = torch.randn(2, 5, 32, 32)

    outputs = adapter(inputs)

    assert outputs.shape == (2, 3, 32, 32)
    assert outputs.dtype == torch.float32
    assert outputs.device.type == "cpu"
    assert adapter.requested_precision == "fp32"
    assert adapter.resolved_precision == "fp32"


def test_band_adapter_bypasses_three_band_inputs_when_enabled() -> None:
    adapter = BandAdapter(in_channels=3, precision="bf16", bypass_rgb=True)
    inputs = torch.randn(2, 3, 16, 16, dtype=torch.float32)

    outputs = adapter(inputs)

    assert outputs.shape == (2, 3, 16, 16)
    assert outputs.dtype == torch.bfloat16
    assert torch.equal(outputs, inputs.to(dtype=torch.bfloat16))


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("fp32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ],
)
def test_band_adapter_honors_requested_output_precision(
    precision: str,
    expected_dtype: torch.dtype,
) -> None:
    adapter = BandAdapter(in_channels=7, precision=precision)
    outputs = adapter(torch.randn(1, 7, 8, 8))

    assert outputs.dtype == expected_dtype
    assert outputs.device.type == "cpu"


def test_nf4_precision_falls_back_to_bf16_when_bitsandbytes_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("poultry_geo.ssl_models._bitsandbytes_supports_nf4", lambda: False)

    adapter = BandAdapter(in_channels=4, precision="nf4")
    vit = RemoteSensingViT(
        image_size=32,
        patch_size=8,
        embed_dim=48,
        depth=2,
        num_heads=4,
        precision="nf4",
    )
    outputs = vit(adapter(torch.randn(1, 4, 32, 32)))

    assert adapter.resolved_precision == "bf16"
    assert vit.resolved_precision == "bf16"
    assert outputs.dtype == torch.bfloat16


def test_vit_forward_features_returns_patch_tokens_and_metadata() -> None:
    model = RemoteSensingViT(
        image_size=32,
        patch_size=8,
        embed_dim=48,
        depth=2,
        num_heads=4,
        precision="fp32",
    )
    inputs = torch.randn(2, 3, 32, 32)

    outputs = model.forward_features(inputs)

    assert outputs.shape == (2, 16, 48)
    assert model.num_patches == 16
    assert model.grid_size == (4, 4)
    assert model.embed_dim == 48
    assert outputs.device.type == "cpu"
    assert outputs.dtype == torch.float32


def test_vit_forward_matches_forward_features_output_shape() -> None:
    model = RemoteSensingViT(
        image_size=32,
        patch_size=8,
        embed_dim=64,
        depth=1,
        num_heads=4,
    )
    inputs = torch.randn(1, 3, 32, 32)

    assert model(inputs).shape == model.forward_features(inputs).shape == (1, 16, 64)


def test_gradient_checkpointing_runs_only_when_enabled_and_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_checkpoint(function, *args, **kwargs):
        calls.append(1)
        return function(*args)

    monkeypatch.setattr("poultry_geo.ssl_models.checkpoint_utils.checkpoint", fake_checkpoint)
    model = RemoteSensingViT(
        image_size=32,
        patch_size=8,
        embed_dim=32,
        depth=3,
        num_heads=4,
        gradient_checkpointing=True,
    )
    training_inputs = torch.randn(2, 3, 32, 32, requires_grad=True)

    model.train()
    outputs = model.forward_features(training_inputs)
    assert outputs.shape == (2, 16, 32)
    assert len(calls) == 3

    calls.clear()
    model.eval()
    model.forward_features(torch.randn(2, 3, 32, 32))
    assert calls == []


def test_vit_tracks_requested_output_precision_on_cpu() -> None:
    model = RemoteSensingViT(
        image_size=32,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        precision="fp16",
    )

    outputs = model.forward_features(torch.randn(2, 3, 32, 32))

    assert outputs.dtype == torch.float16
    assert outputs.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_models_preserve_cuda_device_and_precision() -> None:
    adapter = BandAdapter(in_channels=5, precision="fp16").to(device="cuda", dtype=torch.float16)
    model = RemoteSensingViT(
        image_size=32,
        patch_size=8,
        embed_dim=48,
        depth=2,
        num_heads=4,
        precision="fp16",
    ).to(device="cuda", dtype=torch.float16)
    inputs = torch.randn(2, 5, 32, 32, device="cuda", dtype=torch.float16)

    outputs = model.forward_features(adapter(inputs))

    assert outputs.device.type == "cuda"
    assert outputs.dtype == torch.float16


def test_ddp_can_wrap_adapter_and_vit_together() -> None:
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
        model = torch.nn.Sequential(
            BandAdapter(in_channels=5, precision="fp32"),
            RemoteSensingViT(
                image_size=32,
                patch_size=8,
                embed_dim=48,
                depth=2,
                num_heads=4,
                precision="fp32",
            ),
        )
        wrapped = DistributedDataParallel(model)
        outputs = wrapped(torch.randn(2, 5, 32, 32))

        assert outputs.shape == (2, 16, 48)
        assert outputs.dtype == torch.float32
    finally:
        dist.destroy_process_group()


def test_invalid_precision_patch_size_and_runtime_image_size_are_rejected() -> None:
    with pytest.raises(ValueError, match="fp32, fp16, bf16, or nf4"):
        BandAdapter(in_channels=4, precision="bad")

    with pytest.raises(ValueError, match="divisible"):
        RemoteSensingViT(
            image_size=30,
            patch_size=8,
            embed_dim=32,
            depth=2,
            num_heads=4,
        )

    model = RemoteSensingViT(
        image_size=32,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
    )
    with pytest.raises(ValueError, match="divisible"):
        model.forward_features(torch.randn(1, 3, 30, 32))
