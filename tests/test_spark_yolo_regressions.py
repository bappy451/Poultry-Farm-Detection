from pathlib import Path

from poultry_geo.inference import _dedupe_records


def _record(class_id, confidence, points, source_id="source-a", source_crs="EPSG:26916"):
    import json
    return {
        "class": class_id,
        "confidence": confidence,
        "geometry_native": json.dumps(points),
        "source_id": source_id,
        "source_crs": source_crs,
    }


def test_cross_tile_nms_is_class_aware():
    square = [[0, 0], [2, 0], [2, 2], [0, 2]]
    overlap = [[0.1, 0.1], [2.1, 0.1], [2.1, 2.1], [0.1, 2.1]]
    records = [_record(0, 0.9, square), _record(0, 0.8, overlap), _record(1, 0.7, overlap)]
    kept = _dedupe_records(records, 0.5)
    assert len(kept) == 2
    assert {record["class"] for record in kept} == {0, 1}


def test_nms_does_not_cross_source_or_crs():
    square = [[0, 0], [2, 0], [2, 2], [0, 2]]
    records = [
        _record(0, 0.9, square, source_id="a.tif", source_crs="EPSG:3857"),
        _record(0, 0.8, square, source_id="b.tif", source_crs="EPSG:4326"),
    ]
    assert len(_dedupe_records(records, 0.5)) == 2


def test_sparse_weight_layout_round_trip():
    import torch
    from poultry_geo.spark_models import dense_conv_weight_to_spconv, spconv_weight_to_dense_conv

    dense = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 3, 5, 7)
    sparse = dense_conv_weight_to_spconv(dense)
    assert sparse.shape == (2, 5, 7, 3)
    assert torch.equal(spconv_weight_to_dense_conv(sparse), dense)


def test_cli_normalization_options_are_scoped_to_infer_yolo():
    source = Path("src/poultry_geo/cli.py").read_text(encoding="utf-8-sig")
    assert source.count("args.normalization_min") == 2
    assert source.count('"--normalization-min"') == 1


def test_spark_wrappers_only_advertise_resnet50():
    powershell = Path("scripts/train_spark.ps1").read_text(encoding="utf-8-sig")
    shell = Path("scripts/train_spark.sh").read_text(encoding="utf-8-sig")
    assert "cspdarknet" not in powershell.lower()
    assert '"yolo11l"' not in powershell.lower()
    assert "cspdarknet" not in shell.lower()


def test_legacy_downstream_scripts_fail_closed_for_spark_to_yolo():
    for name in [
        "downstream_compare.py", "downstream_fewshot_poultry.py", "downstream_scratch_compare.py",
        "prepare_and_train_dota.py", "prepare_and_train_grsd.py",
    ]:
        source = (Path("scripts") / name).read_text(encoding="utf-8-sig")
        assert "SparK-to-YOLO transfer is disabled" in source
        assert "backbone_dense_ep299.pt" not in source


def test_grid_exports_best_sparse_encoder_after_latest_checkpoint():
    source = Path("src/poultry_geo/spark_grid_search.py").read_text(encoding="utf-8-sig")
    assert "import shutil" in source
    assert source.index("save_spark_checkpoint(") < source.index("sparse_encoder_best.pt")

