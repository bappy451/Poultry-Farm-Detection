from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

from poultry_geo.analytics import (
    aggregate_detections_by_county,
    apply_nass_farm_correction,
    bootstrap_adjusted_runoff,
    compute_county_runoff_screening,
    load_adjustment_factors,
)
from poultry_geo.geotiff import find_geotiffs, slice_geotiff_to_files
from poultry_geo.gis import create_boundary_shapefiles_from_geotiffs, save_polygon_shapefile_from_csv
from poultry_geo.inference import run_yolo_inference
from poultry_geo.postprocess import (
    centroid_clustering_suppression,
    detect_poultry_farms,
    load_detection_csv,
    mask_nms_by_area,
)
from poultry_geo.sam import run_sam_geo_csv_batch


@dataclass(slots=True)
class StepToggles:
    slice_export: bool = False
    create_boundaries: bool = True
    infer_yolo: bool = True
    centroid_suppress: bool = True
    mask_nms: bool = False
    cluster_farms: bool = True
    export_polygons: bool = True
    sam_refine: bool = False
    aggregate_counties: bool = False
    runoff_screening: bool = False


@dataclass(slots=True)
class SliceExportConfig:
    output_dir: str = "tiles"
    tile_size: int = 3000
    stride: int | None = None
    to_jpeg: bool = False


@dataclass(slots=True)
class PipelineConfig:
    working_dir: str
    geotiff_dir: str
    model_path: str | None = None
    sam_checkpoint_path: str | None = None
    window_size: int = 3000
    stride: int = 3000
    batch_size: int = 8
    confidence_threshold: float = 0.0
    centroid_distance_threshold: float = 10.0
    mask_iou_threshold: float = 0.5
    farm_eps_house: float = 50.0
    farm_min_samples: int = 2
    farm_eps_merge: float = 100.0
    polygon_geometry_column: str = "mask_points_gis"
    reference_crs: str = "EPSG:26916"
    sam_model_type: str = "vit_h"
    county_column: str = "county"
    roof_area_column: str = "area_geo"
    precipitation_csv: str | None = None
    runoff_coefficient: float = 0.8
    nass_farms_column: str = "nass_farms"
    detected_farms_column: str = "detected_farms"
    use_nass_correction: bool = True
    nass_count_csv: str | None = None
    adjustment_factors_csv: str | None = None
    adjustment_factor_column: str = "adjustment_factor"
    steps: StepToggles | None = None
    slice_export: SliceExportConfig | None = None


def _construct_dataclass(dataclass_type: type[Any], payload: dict[str, Any] | None) -> Any:
    payload = payload or {}
    allowed = {field.name for field in fields(dataclass_type)}
    filtered = {key: value for key, value in payload.items() if key in allowed}
    return dataclass_type(**filtered)


def load_pipeline_config(config_path: str | Path) -> PipelineConfig:
    config_path = Path(config_path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    base_fields = {field.name for field in fields(PipelineConfig)} - {"steps", "slice_export"}
    base_payload = {key: value for key, value in data.items() if key in base_fields}
    return PipelineConfig(
        **base_payload,
        steps=_construct_dataclass(StepToggles, data.get("steps")),
        slice_export=_construct_dataclass(SliceExportConfig, data.get("slice_export")),
    )


def _write_output_summary(summary_path: Path, outputs: dict[str, Any]) -> None:
    summary_path.write_text(json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8")


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    steps = config.steps or StepToggles()
    slice_export = config.slice_export or SliceExportConfig()
    working_dir = Path(config.working_dir)
    geotiff_dir = Path(config.geotiff_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Any] = {
        "working_dir": str(working_dir),
        "geotiff_dir": str(geotiff_dir),
        "steps": asdict(steps),
    }

    current_csv: Path | None = None
    county_summary_csv: Path | None = None

    if steps.slice_export:
        tile_root = working_dir / slice_export.output_dir
        slice_outputs: dict[str, list[str]] = {}
        for geotiff_path in find_geotiffs(geotiff_dir):
            output_dir = tile_root / geotiff_path.stem
            written = slice_geotiff_to_files(
                geotiff_path,
                output_dir,
                tile_size=slice_export.tile_size,
                stride=slice_export.stride,
                to_jpeg=slice_export.to_jpeg,
            )
            slice_outputs[geotiff_path.name] = [str(path) for path in written]
        outputs["slice_export"] = slice_outputs

    if steps.create_boundaries:
        boundary_dir = working_dir / "boundary_shapes"
        written = create_boundary_shapefiles_from_geotiffs(geotiff_dir, boundary_dir)
        outputs["boundary_shapes"] = [str(path) for path in written]

    if steps.infer_yolo:
        if not config.model_path:
            raise ValueError("infer_yolo requires model_path in the pipeline config.")
        raw_csv = working_dir / "raw_detections.csv"
        current_csv = run_yolo_inference(
            geotiff_dir=geotiff_dir,
            model_path=config.model_path,
            output_csv=raw_csv,
            window_size=config.window_size,
            stride=config.stride,
            batch_size=config.batch_size,
            confidence_threshold=config.confidence_threshold,
        )
        outputs["raw_detections_csv"] = str(current_csv)

    if steps.centroid_suppress:
        if current_csv is None:
            raise ValueError("centroid_suppress requires infer_yolo or an upstream CSV output.")
        centroid_csv = working_dir / "detections_centroid_suppressed.csv"
        centroid_clustering_suppression(
            load_detection_csv(current_csv),
            distance_threshold=config.centroid_distance_threshold,
            geometry_column=config.polygon_geometry_column,
        ).to_csv(centroid_csv, index=False)
        current_csv = centroid_csv
        outputs["centroid_suppressed_csv"] = str(current_csv)

    if steps.mask_nms:
        if current_csv is None:
            raise ValueError("mask_nms requires infer_yolo or an upstream CSV output.")
        nms_csv = working_dir / "detections_mask_nms.csv"
        mask_nms_by_area(
            load_detection_csv(current_csv),
            iou_threshold=config.mask_iou_threshold,
            geometry_column=config.polygon_geometry_column,
        ).to_csv(nms_csv, index=False)
        current_csv = nms_csv
        outputs["mask_nms_csv"] = str(current_csv)

    if steps.cluster_farms:
        if current_csv is None:
            raise ValueError("cluster_farms requires infer_yolo or an upstream CSV output.")
        farm_csv = working_dir / "farm_centroids.csv"
        detect_poultry_farms(
            load_detection_csv(current_csv),
            eps_house=config.farm_eps_house,
            min_samples=config.farm_min_samples,
            eps_merge=config.farm_eps_merge,
        ).to_csv(farm_csv, index=False)
        outputs["farm_centroids_csv"] = str(farm_csv)

    if steps.export_polygons:
        if current_csv is None:
            raise ValueError("export_polygons requires infer_yolo or an upstream CSV output.")
        polygon_dir = working_dir / "polygon_shapes"
        shapefile = save_polygon_shapefile_from_csv(
            current_csv,
            polygon_dir,
            geometry_column=config.polygon_geometry_column,
            reference_crs=config.reference_crs,
        )
        outputs["polygon_shapefile"] = str(shapefile)

    if steps.sam_refine:
        if current_csv is None:
            raise ValueError("sam_refine requires infer_yolo or an upstream CSV output.")
        if not config.sam_checkpoint_path:
            raise ValueError("sam_refine requires sam_checkpoint_path in the pipeline config.")
        sam_dir = working_dir / "sam_overlays"
        written = run_sam_geo_csv_batch(
            csv_path=current_csv,
            geotiff_dir=geotiff_dir,
            output_dir=sam_dir,
            checkpoint_path=config.sam_checkpoint_path,
            model_type=config.sam_model_type,
        )
        outputs["sam_overlays"] = [str(path) for path in written]

    if steps.aggregate_counties:
        if current_csv is None:
            raise ValueError("aggregate_counties requires infer_yolo or an upstream CSV output.")
        county_summary_csv = working_dir / "county_summary.csv"
        county_summary = aggregate_detections_by_county(
            load_detection_csv(current_csv),
            county_column=config.county_column,
            roof_area_column=config.roof_area_column,
        )
        county_summary.to_csv(county_summary_csv, index=False)
        outputs["county_summary_csv"] = str(county_summary_csv)

    if steps.runoff_screening:
        if county_summary_csv is None:
            raise ValueError("runoff_screening requires aggregate_counties to run first.")
        if not config.precipitation_csv:
            raise ValueError("runoff_screening requires precipitation_csv in the pipeline config.")

        county_summary = pd.read_csv(county_summary_csv)
        if config.use_nass_correction:
            if config.nass_farms_column not in county_summary.columns:
                if not config.nass_count_csv:
                    raise ValueError(
                        "runoff_screening with NASS correction requires either a "
                        f"'{config.nass_farms_column}' column or nass_count_csv in the pipeline config."
                    )
                nass_counts = pd.read_csv(config.nass_count_csv)
                required = {config.county_column, config.nass_farms_column}
                if not required.issubset(nass_counts.columns):
                    missing = ", ".join(sorted(required - set(nass_counts.columns)))
                    raise ValueError(f"NASS count CSV missing required columns: {missing}")
                county_summary = county_summary.merge(
                    nass_counts[[config.county_column, config.nass_farms_column]],
                    on=config.county_column,
                    how="left",
                    validate="1:1",
                )
            county_summary = apply_nass_farm_correction(
                county_summary,
                nass_farms_column=config.nass_farms_column,
                detected_farms_column=config.detected_farms_column,
            )

        precipitation = pd.read_csv(config.precipitation_csv)
        runoff = compute_county_runoff_screening(
            county_summary,
            precipitation,
            county_column=config.county_column,
            roof_area_column="roof_area_m2",
            runoff_coefficient=config.runoff_coefficient,
        )
        runoff_csv = working_dir / "county_runoff.csv"
        runoff.to_csv(runoff_csv, index=False)
        outputs["county_runoff_csv"] = str(runoff_csv)

        if config.adjustment_factors_csv:
            annual_volume_column = (
                "nass_adjusted_annual_runoff_m3"
                if "nass_adjusted_annual_runoff_m3" in runoff.columns
                else "annual_runoff_m3"
            )
            bootstrap = bootstrap_adjusted_runoff(
                runoff,
                adjustment_factors=load_adjustment_factors(
                    config.adjustment_factors_csv,
                    column=config.adjustment_factor_column,
                ),
                annual_volume_column=annual_volume_column,
            )
            outputs["bootstrap_summary"] = asdict(bootstrap)

    if current_csv is not None:
        outputs["final_detection_csv"] = str(current_csv)

    summary_path = working_dir / "pipeline_outputs.json"
    _write_output_summary(summary_path, outputs)
    outputs["summary_json"] = str(summary_path)
    return outputs
