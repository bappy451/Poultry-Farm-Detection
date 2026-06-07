from __future__ import annotations

import socket

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from poultry_geo.ssl_models import (
    RemoteSensingSegmentationModel,
    RemoteSensingViT,
    SegmentationHead,
    inject_lora_into_vit,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_freeze_backbone_keeps_lora_trainable() -> None:
    model = RemoteSensingSegmentationModel(
        in_channels=4,
        image_size=64,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        precision="fp32",
        use_lora=True,
        freeze_backbone=True,
        lora_rank=4,
        lora_alpha=8.0,
    )

    lora_params = [(name, param) for name, param in model.backbone.named_parameters() if "lora_" in name]
    frozen_backbone_params = [(name, param) for name, param in model.backbone.named_parameters() if "lora_" not in name]

    assert lora_params
    assert all(param.requires_grad for _, param in lora_params)
    assert all(not param.requires_grad for _, param in frozen_backbone_params)
    assert any(param.requires_grad for param in model.head.parameters())

    outputs = model(torch.randn(2, 4, 64, 64))
    assert outputs.shape == (2, 1, 64, 64)


@pytest.mark.parametrize(
    ("output_size", "grid_size"),
    [
        ((256, 256), (16, 16)),
        ((512, 512), (32, 32)),
    ],
)
@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("fp32", torch.float32),
        ("bf16", torch.bfloat16),
    ],
)
def test_segmentation_head_restores_binary_mask_resolution(
    output_size: tuple[int, int],
    grid_size: tuple[int, int],
    precision: str,
    expected_dtype: torch.dtype,
) -> None:
    head = SegmentationHead(
        embed_dim=24,
        output_size=output_size,
        hidden_channels=32,
        precision=precision,
    )
    tokens = torch.randn(2, grid_size[0] * grid_size[1], 24)

    masks = head(tokens, grid_size=grid_size)

    assert masks.shape == (2, 1, output_size[0], output_size[1])
    assert masks.dtype == expected_dtype
    assert masks.device.type == "cpu"


def test_lora_injection_preserves_feature_shape_and_precision() -> None:
    backbone = RemoteSensingViT(
        image_size=64,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        precision="bf16",
    )

    inject_lora_into_vit(
        backbone,
        rank=4,
        alpha=8.0,
        dropout=0.0,
        freeze_backbone=False,
    )
    outputs = backbone.forward_features(torch.randn(2, 3, 64, 64))

    assert outputs.shape == (2, 64, 32)
    assert outputs.dtype == torch.bfloat16
    assert backbone.blocks[0].attn.qkv.__class__.__name__ == "LoRALinear"
    assert backbone.blocks[0].attn.proj.__class__.__name__ == "LoRALinear"


def test_nf4_segmentation_model_falls_back_to_bf16_without_bitsandbytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("poultry_geo.ssl_models._bitsandbytes_supports_nf4", lambda: False)

    model = RemoteSensingSegmentationModel(
        in_channels=4,
        image_size=64,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        precision="nf4",
        use_lora=True,
        freeze_backbone=True,
        lora_rank=4,
    )

    outputs = model(torch.randn(1, 4, 64, 64))

    assert model.backbone.resolved_precision == "bf16"
    assert outputs.dtype == torch.bfloat16


def test_decoder_rejects_inconsistent_token_grid() -> None:
    head = SegmentationHead(
        embed_dim=16,
        output_size=64,
        hidden_channels=16,
        precision="fp32",
    )

    with pytest.raises(ValueError, match="grid_size"):
        head(torch.randn(1, 15, 16), grid_size=(4, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_segmentation_model_preserves_cuda_device_and_precision() -> None:
    model = RemoteSensingSegmentationModel(
        in_channels=4,
        image_size=64,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        precision="fp16",
        use_lora=True,
        freeze_backbone=True,
        lora_rank=4,
    ).to(device="cuda", dtype=torch.float16)
    inputs = torch.randn(2, 4, 64, 64, device="cuda", dtype=torch.float16)

    outputs = model(inputs)

    assert outputs.shape == (2, 1, 64, 64)
    assert outputs.device.type == "cuda"
    assert outputs.dtype == torch.float16


def test_segmentation_model_ddp_smoke() -> None:
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
        model = RemoteSensingSegmentationModel(
            in_channels=4,
            image_size=64,
            patch_size=8,
            embed_dim=32,
            depth=1,
            num_heads=4,
            precision="fp32",
            use_lora=True,
            freeze_backbone=True,
            lora_rank=4,
        )
        wrapped = DistributedDataParallel(model)
        outputs = wrapped(torch.randn(2, 4, 64, 64))

        assert outputs.shape == (2, 1, 64, 64)
        assert outputs.dtype == torch.float32
    finally:
        dist.destroy_process_group()
