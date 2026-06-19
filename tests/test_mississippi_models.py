from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from poultry_geo.mississippi_models import (
    MississippiFMEncoder,
    compute_gram_anchoring_loss,
    compute_overlap_alignment_loss,
)


def test_mississippi_encoder_forward_views_returns_expected_heads() -> None:
    model = MississippiFMEncoder(
        image_size=256,
        model_size="vit_s",
        projection_dim=64,
        precision="fp32",
    )
    inputs = torch.randn(2, 3, 256, 256)

    outputs = model.forward_views(global_views=inputs, local_views=inputs)

    assert outputs["global_cls"].shape == (2, model.embed_dim)
    assert outputs["global_distillation"].shape == (2, 64)
    assert outputs["patch_predictions"].shape[0] == 2
    assert outputs["overlap_projection"].shape[-1] == 64
    assert outputs["local_cls"].shape == (2, model.embed_dim)


def test_gram_anchoring_loss_is_zero_for_identical_embeddings() -> None:
    embeddings = torch.randn(2, 16, 32)

    loss = compute_gram_anchoring_loss(embeddings, embeddings.clone())

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_overlap_alignment_loss_is_zero_for_identical_corresponding_patches() -> None:
    left = torch.nn.functional.normalize(torch.randn(1, 8, 16), dim=-1)
    right = left.clone()
    indices = torch.tensor([[0, 2, 4, 6]])

    loss = compute_overlap_alignment_loss(left, right, indices, indices)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_mississippi_encoder_can_enable_lora_adapters() -> None:
    model = MississippiFMEncoder(image_size=256, model_size="vit_s", precision="fp32")

    model.enable_lora(rank=2, alpha=4.0, dropout=0.0)

    assert model.use_lora is True
    assert model.backbone.blocks[0].attn.qkv.__class__.__name__ == "LoRALinear"

    model.disable_lora()

    assert model.use_lora is False


def test_mississippi_encoder_can_use_timm_dinov3_backbone() -> None:
    pytest.importorskip("timm")
    model = MississippiFMEncoder(
        image_size=256,
        model_size="vit_s",
        projection_dim=64,
        precision="fp32",
        backbone_impl="timm_dinov3",
    )
    inputs = torch.randn(2, 3, 256, 256)

    outputs = model.forward_views(global_views=inputs)

    assert outputs["global_cls"].shape == (2, model.embed_dim)
    assert outputs["global_patches"].shape[1] == (256 // model.patch_size) ** 2


def test_mississippi_encoder_can_enable_and_disable_nf4() -> None:
    model = MississippiFMEncoder(image_size=256, model_size="vit_s", precision="fp32")

    config = model.enable_nf4()

    assert model.nf4_enabled is True
    assert config is not None

    model.disable_nf4()

    assert model.nf4_enabled is False
    assert model.quantization_config is None
