from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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
