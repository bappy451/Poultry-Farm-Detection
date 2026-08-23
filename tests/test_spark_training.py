import pytest
import torch
import torch.nn as nn

from poultry_geo.train_spark_ssl import SparKConfig, calculate_spark_loss, save_spark_checkpoint


def test_loss_rejects_misaligned_geometry():
    with pytest.raises(ValueError, match="does not match"):
        calculate_spark_loss(torch.zeros(1, 3, 16, 16), torch.zeros(1, 3, 8, 8), torch.ones(1, 1, 16, 16), 4)


def test_checkpoint_persists_transformer_and_native_sparse_encoder_state(tmp_path):
    model = nn.Module()
    model.encoder = nn.Conv2d(3, 3, 1)
    transformer = nn.Module()
    transformer.adapter = nn.Conv2d(3, 3, 1)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(transformer.parameters()))
    config = SparKConfig(backbone_type="resnet50")
    save_spark_checkpoint(model, optimizer, None, config, str(tmp_path), 2, transformer=transformer)
    checkpoint = torch.load(tmp_path / "spark_unified_latest.pt", weights_only=False)
    assert checkpoint["format_version"] == 2
    assert isinstance(checkpoint["rng_state"], dict)
    for key, value in transformer.state_dict().items():
        assert torch.equal(checkpoint["transformer_state_dict"][key], value)
    sparse_export = torch.load(tmp_path / "sparse_encoder_latest.pt", weights_only=False)
    assert sparse_export["architecture"] == "spark_sparse_resnet50_custom_stem"
    assert "not supported for SparK-to-YOLO transfer" in sparse_export["export_note"]
