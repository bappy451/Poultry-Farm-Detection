from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from textwrap import dedent

import pandas as pd

from poultry_geo.analytics import (
    aggregate_detections_by_county,
    apply_nass_farm_correction,
    bootstrap_adjusted_runoff,
    compute_county_runoff_screening,
    load_adjustment_factors,
)
from poultry_geo.command_logging import add_logging_arguments, command_log_context, resolve_logging_config
from poultry_geo.conversion import (
    convert_geotiff_to_png_or_jpg,
    csv_to_kml_polygons,
    sample_random_rows,
    shapefile_to_csv,
    shapefile_to_projected_csv,
)
from poultry_geo.exporting import export_ultralytics_model
from poultry_geo.geotiff import slice_geotiff_collection_to_files, slice_geotiff_to_files, stitch_geotiff_tiles
from poultry_geo.gis import (
    create_boundary_shapefiles_from_geotiffs,
    filter_shapefile_by_length_width_area_proximity,
    identify_poultry_farms_from_shapefile,
    merge_csv_files,
    merge_overlapping_polygons_with_largest_area_attr,
    process_shapefile_with_bbox_dimensions,
    save_bbox_shapefile_from_csv,
    save_polygon_shapefile_from_csv,
)
from poultry_geo.inference import run_yolo_inference
from poultry_geo.pipeline import load_pipeline_config, run_pipeline
from poultry_geo.postprocess import (
    centroid_clustering_suppression,
    detect_poultry_farms,
    load_detection_csv,
    make_rotated_rectangles,
    mask_nms_by_area,
    merge_adjacent_masks,
    merge_adjacent_polygons,
    merge_masks_by_edge_proximity,
    merge_rect_masks,
)
from poultry_geo.progress import command_progress_context
from poultry_geo.sam import batch_sam2_visualize, run_sam_geo_csv_batch, run_sam_on_bbox_csv
from poultry_geo.training import train_yolo_models
from poultry_geo.transformer_models import predict_hf_detr, train_hf_detr, train_mask2former
class RichHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show defaults while preserving hand-written help layout."""


class RichArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", RichHelpFormatter)
        super().__init__(*args, **kwargs)


def _add_command_parser(
    subparsers: argparse._SubParsersAction,
    name: str,
    *,
    help_text: str,
    description: str | None = None,
    epilog: str | None = None,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        name,
        help=help_text,
        description=description or help_text,
        epilog=epilog,
    )
    add_logging_arguments(parser)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = RichArgumentParser(
        prog="poultry-geo",
        description=dedent(
            """
            Structured poultry-house geospatial pipeline.

            Major workflows:
              - slice GeoTIFFs into TIFF/PNG/JPG tiles
              - stitch TIFF tiles back into a raster
              - convert GeoTIFFs to display-friendly images
              - run inference, post-process detections, and export GIS outputs
              - train or export models

            Use `poultry-geo <command> --help` for command-specific options and examples.
            Every command is logged automatically and appended to Code/command.log unless you pass --no-log.
            """
        ).strip(),
        epilog=dedent(
            """
            Examples:
              poultry-geo slice ../Data/Test_ms_ga ./tmp/tiles --tile-size 256 --format png --workers auto
              poultry-geo convert-geotiff ../Data/Test_ms_ga ./tmp/jpgs --format jpg --jpeg-quality 90
              poultry-geo infer-yolo ../Data/Test_ms_ga weights/best.pt detections.csv --window-size 3000 --stride 3000
            """
        ).strip(),
    )
    add_logging_arguments(parser)
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=RichArgumentParser,
    )

    slice_parser = _add_command_parser(
        subparsers,
        "slice",
        help_text="Slice one GeoTIFF or a directory of GeoTIFFs into square tiles.",
        description=dedent(
            """
            Slice one GeoTIFF or a directory tree of GeoTIFFs into square tiles.

            Default behavior is non-overlapping tiling (`stride = tile_size`) with edge snapping:
              - interior windows are exactly tile_size x tile_size
              - if dimensions do not divide evenly, the last window snaps to the image edge
              - the only default overlap is that final edge-snapped window
              - no zero padding is written or passed downstream

            Multiprocessing is file-level only: one worker processes one full GeoTIFF from open to final write.
            """
        ).strip(),
        epilog=dedent(
            """
            Examples:
              poultry-geo slice input.tif output_tiles --tile-size 256 --format png
              poultry-geo slice geotiffs/ output_tiles --tile-size 64 --stride 32 --format jpg --workers 4
              poultry-geo slice geotiffs/ output_tiles --tile-size 256 --format tif --workers auto
            """
        ).strip(),
    )
    slice_parser.add_argument("input_geotiff")
    slice_parser.add_argument("output_dir")
    slice_parser.add_argument("--tile-size", type=int, default=3000, help="Square tile size in pixels.")
    slice_parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sliding-window stride in pixels. Defaults to tile size for non-overlapping export.",
    )
    slice_parser.add_argument("--jpeg", action="store_true", help="Shortcut for JPEG output when --format is omitted.")
    slice_parser.add_argument(
        "--format",
        default=None,
        choices=["tif", "png", "jpg", "jpeg"],
        help="Tile output format.",
    )
    slice_parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality used when writing JPG tiles.",
    )
    slice_parser.add_argument(
        "--workers",
        default="auto",
        help="Number of file-level worker processes, or 'auto'.",
    )

    stitch_parser = _add_command_parser(
        subparsers,
        "stitch",
        help_text="Stitch tiled GeoTIFF slices back into one raster.",
        epilog=dedent(
            """
            Example:
              poultry-geo stitch output_tiles rebuilt.tif
            """
        ).strip(),
    )
    stitch_parser.add_argument("input_dir")
    stitch_parser.add_argument("output_geotiff")

    boundary_parser = _add_command_parser(
        subparsers,
        "create-boundaries",
        help_text="Create boundary shapefiles from GeoTIFF extents.",
    )
    boundary_parser.add_argument("geotiff_dir")
    boundary_parser.add_argument("output_dir")

    convert_parser = _add_command_parser(
        subparsers,
        "convert-geotiff",
        help_text="Convert GeoTIFFs to PNG or JPG.",
        description=dedent(
            """
            Convert one GeoTIFF or a directory tree of GeoTIFFs into display-friendly PNG or JPG images.

            The output directory mirrors the input folder structure when the input is a directory.
            """
        ).strip(),
        epilog=dedent(
            """
            Examples:
              poultry-geo convert-geotiff input.tif converted/ --format png
              poultry-geo convert-geotiff geotiffs/ converted/ --format jpg --jpeg-quality 90
            """
        ).strip(),
    )
    convert_parser.add_argument("input_dir")
    convert_parser.add_argument("output_dir")
    convert_parser.add_argument("--format", default="png", choices=["png", "jpg", "jpeg"])
    convert_parser.add_argument("--jpeg-quality", type=int, default=95)

    sample_parser = _add_command_parser(subparsers, "sample-rows", help_text="Sample random rows from a CSV or Excel file.")
    sample_parser.add_argument("input_path")
    sample_parser.add_argument("n_rows", type=int)
    sample_parser.add_argument("output_csv")
    sample_parser.add_argument("--seed", type=int, default=27)

    merge_csv_parser = _add_command_parser(subparsers, "merge-csv", help_text="Merge two CSV files into one output CSV.")
    merge_csv_parser.add_argument("input_csv_1")
    merge_csv_parser.add_argument("input_csv_2")
    merge_csv_parser.add_argument("output_csv")

    shape_to_csv_parser = _add_command_parser(subparsers, "shapefile-to-csv", help_text="Convert a shapefile to CSV.")
    shape_to_csv_parser.add_argument("input_shapefile")
    shape_to_csv_parser.add_argument("output_csv")

    shape_to_proj_parser = _add_command_parser(
        subparsers,
        "shapefile-to-projected-csv",
        help_text="Convert a shapefile to projected coordinates in CSV form.",
    )
    shape_to_proj_parser.add_argument("input_shapefile")
    shape_to_proj_parser.add_argument("output_csv")
    shape_to_proj_parser.add_argument("--target-epsg", default="26916")

    kml_parser = _add_command_parser(subparsers, "csv-to-kml", help_text="Convert detection CSV polygons to KML.")
    kml_parser.add_argument("input_csv")
    kml_parser.add_argument("output_kml")
    kml_parser.add_argument("--source-epsg", default="26916")

    polygons_parser = _add_command_parser(subparsers, "export-polygons", help_text="Export polygon detections from CSV to shapefile.")
    polygons_parser.add_argument("input_csv")
    polygons_parser.add_argument("output_dir")
    polygons_parser.add_argument("--geometry-column", default="mask_points_gis")
    polygons_parser.add_argument("--crs", default="EPSG:26916")

    bbox_parser = _add_command_parser(subparsers, "export-bboxes", help_text="Export detection bounding boxes from CSV to shapefile.")
    bbox_parser.add_argument("input_csv")
    bbox_parser.add_argument("output_dir")
    bbox_parser.add_argument("--crs", default="EPSG:26916")

    infer_parser = _add_command_parser(subparsers, "infer-yolo", help_text="Run tiled YOLO inference over GeoTIFFs and export CSV.")
    infer_parser.add_argument("geotiff_dir")
    infer_parser.add_argument("model_path")
    infer_parser.add_argument("output_csv")
    infer_parser.add_argument("--window-size", type=int, default=3000)
    infer_parser.add_argument("--stride", type=int, default=3000)
    infer_parser.add_argument("--batch-size", type=int, default=8)
    infer_parser.add_argument("--confidence-threshold", type=float, default=0.0)
    infer_parser.add_argument("--device", default=None)

    cluster_parser = _add_command_parser(subparsers, "cluster-farms", help_text="Group poultry-house detections into farm centroids.")
    cluster_parser.add_argument("input_csv")
    cluster_parser.add_argument("output_csv")
    cluster_parser.add_argument("--eps-house", type=float, default=50.0)
    cluster_parser.add_argument("--min-samples", type=int, default=2)
    cluster_parser.add_argument("--eps-merge", type=float, default=100.0)

    centroid_parser = _add_command_parser(subparsers, "centroid-suppress", help_text="Deduplicate nearby detections by centroid clustering.")
    centroid_parser.add_argument("input_csv")
    centroid_parser.add_argument("output_csv")
    centroid_parser.add_argument("--distance-threshold", type=float, default=10.0)
    centroid_parser.add_argument("--geometry-column", default=None)

    mask_nms_parser = _add_command_parser(subparsers, "mask-nms", help_text="Suppress overlapping masks using polygon IoU.")
    mask_nms_parser.add_argument("input_csv")
    mask_nms_parser.add_argument("output_csv")
    mask_nms_parser.add_argument("--iou-threshold", type=float, default=0.5)
    mask_nms_parser.add_argument("--geometry-column", default=None)

    rectangles_parser = _add_command_parser(
        subparsers,
        "create-rectangles",
        help_text="Add rotated rectangle geometry columns derived from polygon masks.",
    )
    rectangles_parser.add_argument("input_csv")
    rectangles_parser.add_argument("output_csv")
    rectangles_parser.add_argument("--geometry-column", default="mask_points_gis")

    merge_poly_parser = _add_command_parser(
        subparsers,
        "merge-polygons",
        help_text="Merge overlapping polygon detections into one highest-confidence record per group.",
    )
    merge_poly_parser.add_argument("input_csv")
    merge_poly_parser.add_argument("output_csv")
    merge_poly_parser.add_argument("--eps", type=float, default=0.0)
    merge_poly_parser.add_argument("--geometry-column", default="mask_points_gis")

    merge_adjacent_parser = _add_command_parser(
        subparsers,
        "merge-adjacent-masks",
        help_text="Merge nearby rotated-rectangle masks by centerline distance.",
    )
    merge_adjacent_parser.add_argument("input_csv")
    merge_adjacent_parser.add_argument("output_csv")
    merge_adjacent_parser.add_argument("--distance-threshold", type=float, default=2.0)
    merge_adjacent_parser.add_argument("--geometry-column", default="rect_mask_points")

    merge_rect_parser = _add_command_parser(
        subparsers,
        "merge-rect-masks",
        help_text="Keep the largest rectangle when rotated-rectangle masks overlap.",
    )
    merge_rect_parser.add_argument("input_csv")
    merge_rect_parser.add_argument("output_csv")
    merge_rect_parser.add_argument("--geometry-column", default="rect_mask_points")

    merge_edge_parser = _add_command_parser(
        subparsers,
        "merge-masks-by-edge",
        help_text="Merge masks whose edges are within a threshold distance.",
    )
    merge_edge_parser.add_argument("input_csv")
    merge_edge_parser.add_argument("output_csv")
    merge_edge_parser.add_argument("--edge-threshold", type=float, default=2.0)
    merge_edge_parser.add_argument("--fill-gap", action=argparse.BooleanOptionalAction, default=True)
    merge_edge_parser.add_argument("--geometry-column", default="rect_mask_points")

    sam_parser = _add_command_parser(subparsers, "sam-refine", help_text="Run SAM refinement over georeferenced detections.")
    sam_parser.add_argument("input_csv")
    sam_parser.add_argument("geotiff_dir")
    sam_parser.add_argument("output_dir")
    sam_parser.add_argument("checkpoint_path")
    sam_parser.add_argument("--model-type", default="vit_h")

    sam_bbox_parser = _add_command_parser(
        subparsers,
        "sam-bbox-refine",
        help_text="Run SAM refinement using projected bounding boxes instead of point prompts.",
    )
    sam_bbox_parser.add_argument("input_csv")
    sam_bbox_parser.add_argument("geotiff_dir")
    sam_bbox_parser.add_argument("output_dir")
    sam_bbox_parser.add_argument("checkpoint_path")
    sam_bbox_parser.add_argument("--model-type", default="vit_h")

    sam2_parser = _add_command_parser(
        subparsers,
        "sam2-visualize",
        help_text="Run SAM2 automatic mask generation on ordinary images and save visualizations.",
    )
    sam2_parser.add_argument("image_dir")
    sam2_parser.add_argument("checkpoint_path")
    sam2_parser.add_argument("model_config")
    sam2_parser.add_argument("--output-dir", default=None)
    sam2_parser.add_argument("--save-images", action="store_true")
    sam2_parser.add_argument("--device", default=None)

    viz_geo_parser = _add_command_parser(
        subparsers,
        "visualize-geotiffs",
        help_text="Overlay detection CSV annotations onto GeoTIFF imagery.",
    )
    viz_geo_parser.add_argument("geotiff_dir")
    viz_geo_parser.add_argument("input_csv")
    viz_geo_parser.add_argument("output_dir")
    viz_geo_parser.add_argument("--window-size", type=int, default=3000)
    viz_geo_parser.add_argument("--stride", type=int, default=3000)
    viz_geo_parser.add_argument("--rectangle-geometry-column", default=None)

    viz_jpg_parser = _add_command_parser(
        subparsers,
        "visualize-jpgs",
        help_text="Overlay detection CSV annotations onto JPG or PNG images.",
    )
    viz_jpg_parser.add_argument("input_csv")
    viz_jpg_parser.add_argument("jpg_dir")
    viz_jpg_parser.add_argument("output_dir")

    train_parser = _add_command_parser(subparsers, "train-yolo", help_text="Train one or more Ultralytics YOLO models.")
    train_parser.add_argument("data_path")
    train_parser.add_argument("project_dir")
    train_parser.add_argument("--models", nargs="+", required=True)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--image-size", type=int, default=640)
    train_parser.add_argument("--optimizer", default="SGD")
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--patience", type=int, default=100)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--wandb-project", default=None)
    train_parser.add_argument("--family", default="auto", choices=["auto", "yolo", "rtdetr"])

    detr_parser = _add_command_parser(
        subparsers,
        "train-detr",
        help_text="Train a HuggingFace DETR model on COCO-style annotations.",
    )
    detr_parser.add_argument("train_image_dir")
    detr_parser.add_argument("train_annotations")
    detr_parser.add_argument("output_dir")
    detr_parser.add_argument("--model-name", default="facebook/detr-resnet-50")
    detr_parser.add_argument("--num-labels", type=int, default=2)
    detr_parser.add_argument("--batch-size", type=int, default=2)
    detr_parser.add_argument("--epochs", type=int, default=1)
    detr_parser.add_argument("--learning-rate", type=float, default=1e-4)
    detr_parser.add_argument("--weight-decay", type=float, default=1e-4)
    detr_parser.add_argument("--val-image-dir", default=None)
    detr_parser.add_argument("--val-annotations", default=None)
    detr_parser.add_argument("--device", default=None)

    predict_detr_parser = _add_command_parser(
        subparsers,
        "predict-detr",
        help_text="Run HuggingFace DETR inference on one image and print JSON detections.",
    )
    predict_detr_parser.add_argument("image_path")
    predict_detr_parser.add_argument("model_path")
    predict_detr_parser.add_argument("--threshold", type=float, default=0.5)
    predict_detr_parser.add_argument("--device", default=None)

    mask2former_parser = _add_command_parser(
        subparsers,
        "train-mask2former",
        help_text="Train a HuggingFace Mask2Former instance segmentation model.",
    )
    mask2former_parser.add_argument("train_image_dir")
    mask2former_parser.add_argument("train_annotations")
    mask2former_parser.add_argument("output_dir")
    mask2former_parser.add_argument("--model-name", default="facebook/mask2former-swin-base-coco-instance")
    mask2former_parser.add_argument("--batch-size", type=int, default=2)
    mask2former_parser.add_argument("--epochs", type=int, default=1)
    mask2former_parser.add_argument("--learning-rate", type=float, default=1e-4)
    mask2former_parser.add_argument("--weight-decay", type=float, default=1e-4)
    mask2former_parser.add_argument("--device", default=None)

    export_parser = _add_command_parser(
        subparsers,
        "export-model",
        help_text="Export an Ultralytics model to ONNX, TFLite, or another supported format.",
    )
    export_parser.add_argument("model_path")
    export_parser.add_argument("--format", default="tflite")
    export_parser.add_argument("--family", default="auto", choices=["auto", "yolo", "rtdetr"])
    export_parser.add_argument("--device", default=None)
    export_parser.add_argument("--image-size", type=int, default=None)

    aggregate_parser = _add_command_parser(
        subparsers,
        "aggregate-counties",
        help_text="Aggregate detection CSV rows into county-level house counts and roof area totals.",
    )
    aggregate_parser.add_argument("input_csv")
    aggregate_parser.add_argument("output_csv")
    aggregate_parser.add_argument("--county-column", default="county")
    aggregate_parser.add_argument("--roof-area-column", default="area_geo")

    merge_shape_parser = _add_command_parser(
        subparsers,
        "merge-overlapping-shapefile",
        help_text="Merge overlapping polygons in a shapefile and keep the largest-area attributes.",
    )
    merge_shape_parser.add_argument("input_shapefile")
    merge_shape_parser.add_argument("output_shapefile")

    dims_shape_parser = _add_command_parser(
        subparsers,
        "process-shapefile-dimensions",
        help_text="Compute polygon area, length, and width statistics in a projected CRS.",
    )
    dims_shape_parser.add_argument("input_shapefile")
    dims_shape_parser.add_argument("output_shapefile")

    filter_shape_parser = _add_command_parser(
        subparsers,
        "filter-shapefile",
        help_text="Filter a shapefile by dimensions, area, and neighbor proximity.",
    )
    filter_shape_parser.add_argument("input_shapefile")
    filter_shape_parser.add_argument("output_shapefile")
    filter_shape_parser.add_argument("--length-min", type=float, default=370.0)
    filter_shape_parser.add_argument("--width-min", type=float, default=39.0)
    filter_shape_parser.add_argument("--width-max", type=float, default=100.0)
    filter_shape_parser.add_argument("--area-min", type=float, default=15000.0)
    filter_shape_parser.add_argument("--area-max", type=float, default=60000.0)
    filter_shape_parser.add_argument("--proximity-m", type=float, default=50.0)
    filter_shape_parser.add_argument("--disable-class-zero-keep", action="store_true")

    farm_shape_parser = _add_command_parser(
        subparsers,
        "identify-farms-shapefile",
        help_text="Identify farm centroids from a filtered poultry-house shapefile.",
    )
    farm_shape_parser.add_argument("input_shapefile")
    farm_shape_parser.add_argument("output_shapefile")
    farm_shape_parser.add_argument("--proximity-m", type=float, default=250.0)
    farm_shape_parser.add_argument("--min-samples", type=int, default=2)

    runoff_parser = _add_command_parser(
        subparsers,
        "runoff-screening",
        help_text="Compute county-level runoff screening volumes from roof area and precipitation normals.",
    )
    runoff_parser.add_argument("county_summary_csv")
    runoff_parser.add_argument("precipitation_csv")
    runoff_parser.add_argument("output_csv")
    runoff_parser.add_argument("--county-column", default="county")
    runoff_parser.add_argument("--roof-area-column", default="roof_area_m2")
    runoff_parser.add_argument("--runoff-coefficient", type=float, default=0.8)
    runoff_parser.add_argument("--nass-farms-column", default="nass_farms")
    runoff_parser.add_argument("--detected-farms-column", default="detected_farms")
    runoff_parser.add_argument("--skip-nass-correction", action="store_true")
    runoff_parser.add_argument("--adjustment-factors-csv", default=None)
    runoff_parser.add_argument("--adjustment-column", default="adjustment_factor")

    pipeline_parser = _add_command_parser(
        subparsers,
        "run-pipeline",
        help_text="Run the end-to-end geospatial pipeline from a JSON config file.",
    )
    pipeline_parser.add_argument("config_path")

    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    with (
        command_log_context(
            raw_argv,
            config=resolve_logging_config(raw_argv),
            program_name="poultry-geo",
        ),
        command_progress_context(raw_argv) as progress,
    ):
        parser = build_parser()
        args = parser.parse_args(raw_argv)
        progress_callback = (
            None
            if progress is None
            else lambda completed, total, detail: progress.update(completed, total, detail=detail)
        )

        if args.command == "slice":
            if args.jpeg and args.format not in {None, "jpg", "jpeg"}:
                parser.error("--jpeg can only be used with JPEG output.")
            output_format = "jpg" if args.jpeg and args.format is None else args.format
            written = slice_geotiff_collection_to_files(
                args.input_geotiff,
                args.output_dir,
                tile_size=args.tile_size,
                stride=args.stride,
                output_format=output_format,
                jpeg_quality=args.jpeg_quality,
                workers=args.workers,
                progress_callback=progress_callback,
            )
            print(f"Wrote {len(written)} slices to {args.output_dir}")
            return 0

        if args.command == "stitch":
            path = stitch_geotiff_tiles(args.input_dir, args.output_geotiff)
            print(f"Stitched GeoTIFF written to {path}")
            return 0

        if args.command == "create-boundaries":
            written = create_boundary_shapefiles_from_geotiffs(
                args.geotiff_dir,
                args.output_dir,
                progress_callback=progress_callback,
            )
            print(f"Created {len(written)} boundary shapefiles in {args.output_dir}")
            return 0

        if args.command == "convert-geotiff":
            written = convert_geotiff_to_png_or_jpg(
                args.input_dir,
                args.output_dir,
                out_ext=args.format,
                jpeg_quality=args.jpeg_quality,
                progress_callback=progress_callback,
            )
            print(f"Converted {len(written)} GeoTIFF files into {args.output_dir}")
            return 0

        if args.command == "sample-rows":
            sampled = sample_random_rows(
                args.input_path,
                args.n_rows,
                args.output_csv,
                seed=args.seed,
            )
            print(f"Wrote {len(sampled)} sampled rows to {args.output_csv}")
            return 0

        if args.command == "merge-csv":
            output = merge_csv_files(args.input_csv_1, args.input_csv_2, args.output_csv)
            print(f"Merged CSV written to {output}")
            return 0

        if args.command == "shapefile-to-csv":
            output = shapefile_to_csv(args.input_shapefile, args.output_csv)
            print(f"CSV written to {output}")
            return 0

        if args.command == "shapefile-to-projected-csv":
            output = shapefile_to_projected_csv(
                args.input_shapefile,
                args.output_csv,
                target_epsg=args.target_epsg,
            )
            print(f"Projected CSV written to {output}")
            return 0

        if args.command == "csv-to-kml":
            output = csv_to_kml_polygons(
                args.input_csv,
                args.output_kml,
                source_epsg=args.source_epsg,
            )
            print(f"KML written to {output}")
            return 0

        if args.command == "export-polygons":
            shapefile = save_polygon_shapefile_from_csv(
                args.input_csv,
                args.output_dir,
                geometry_column=args.geometry_column,
                reference_crs=args.crs,
            )
            print(f"Polygon shapefile written to {shapefile}")
            return 0

        if args.command == "export-bboxes":
            shapefile = save_bbox_shapefile_from_csv(
                args.input_csv,
                args.output_dir,
                reference_crs=args.crs,
            )
            print(f"Bounding-box shapefile written to {shapefile}")
            return 0

        if args.command == "infer-yolo":
            output = run_yolo_inference(
                args.geotiff_dir,
                args.model_path,
                args.output_csv,
                window_size=args.window_size,
                stride=args.stride,
                batch_size=args.batch_size,
                confidence_threshold=args.confidence_threshold,
                device=args.device,
                progress_callback=progress_callback,
            )
            print(f"Inference CSV written to {output}")
            return 0

        if args.command == "cluster-farms":
            df = load_detection_csv(args.input_csv)
            farms = detect_poultry_farms(
                df,
                eps_house=args.eps_house,
                min_samples=args.min_samples,
                eps_merge=args.eps_merge,
            )
            farms.to_csv(args.output_csv, index=False)
            print(f"Farm centroid CSV written to {args.output_csv}")
            return 0

        if args.command == "centroid-suppress":
            df = load_detection_csv(args.input_csv)
            filtered = centroid_clustering_suppression(
                df,
                distance_threshold=args.distance_threshold,
                geometry_column=args.geometry_column,
            )
            filtered.to_csv(args.output_csv, index=False)
            print(f"Deduplicated CSV written to {args.output_csv}")
            return 0

        if args.command == "mask-nms":
            df = load_detection_csv(args.input_csv)
            filtered = mask_nms_by_area(
                df,
                iou_threshold=args.iou_threshold,
                geometry_column=args.geometry_column,
            )
            filtered.to_csv(args.output_csv, index=False)
            print(f"Mask-NMS CSV written to {args.output_csv}")
            return 0

        if args.command == "create-rectangles":
            output = make_rotated_rectangles(
                load_detection_csv(args.input_csv),
                geometry_column=args.geometry_column,
            )
            output.to_csv(args.output_csv, index=False)
            print(f"Rectangle-enriched CSV written to {args.output_csv}")
            return 0

        if args.command == "merge-polygons":
            output = merge_adjacent_polygons(
                load_detection_csv(args.input_csv),
                eps=args.eps,
                geometry_column=args.geometry_column,
            )
            output.to_csv(args.output_csv, index=False)
            print(f"Merged polygon CSV written to {args.output_csv}")
            return 0

        if args.command == "merge-adjacent-masks":
            output = merge_adjacent_masks(
                load_detection_csv(args.input_csv),
                distance_threshold=args.distance_threshold,
                geometry_column=args.geometry_column,
            )
            output.to_csv(args.output_csv, index=False)
            print(f"Adjacent-mask merge CSV written to {args.output_csv}")
            return 0

        if args.command == "merge-rect-masks":
            output = merge_rect_masks(
                load_detection_csv(args.input_csv),
                geometry_column=args.geometry_column,
            )
            output.to_csv(args.output_csv, index=False)
            print(f"Rectangle merge CSV written to {args.output_csv}")
            return 0

        if args.command == "merge-masks-by-edge":
            output = merge_masks_by_edge_proximity(
                load_detection_csv(args.input_csv),
                edge_threshold=args.edge_threshold,
                fill_gap=args.fill_gap,
                geometry_column=args.geometry_column,
            )
            output.to_csv(args.output_csv, index=False)
            print(f"Edge-merged CSV written to {args.output_csv}")
            return 0

        if args.command == "sam-refine":
            written = run_sam_geo_csv_batch(
                args.input_csv,
                args.geotiff_dir,
                args.output_dir,
                args.checkpoint_path,
                model_type=args.model_type,
            )
            print(f"Wrote {len(written)} SAM annotation images to {args.output_dir}")
            return 0

        if args.command == "sam-bbox-refine":
            written = run_sam_on_bbox_csv(
                args.input_csv,
                args.geotiff_dir,
                args.output_dir,
                args.checkpoint_path,
                model_type=args.model_type,
            )
            print(f"Wrote {len(written)} bbox-SAM annotation images to {args.output_dir}")
            return 0

        if args.command == "sam2-visualize":
            written = batch_sam2_visualize(
                args.image_dir,
                args.checkpoint_path,
                args.model_config,
                output_dir=args.output_dir,
                save_images=args.save_images,
                device=args.device,
            )
            print(f"Wrote {len(written)} SAM2 visualization images.")
            return 0

        if args.command == "visualize-geotiffs":
            from poultry_geo.visualization import visualize_detection_csv

            written = visualize_detection_csv(
                args.geotiff_dir,
                args.input_csv,
                args.output_dir,
                window_size=args.window_size,
                stride=args.stride,
                rectangle_geometry_column=args.rectangle_geometry_column,
            )
            print(f"Wrote {len(written)} annotated GeoTIFF overlays to {args.output_dir}")
            return 0

        if args.command == "visualize-jpgs":
            from poultry_geo.visualization import visualize_detection_csv_on_jpgs

            written = visualize_detection_csv_on_jpgs(
                args.input_csv,
                args.jpg_dir,
                args.output_dir,
            )
            print(f"Wrote {len(written)} annotated JPG overlays to {args.output_dir}")
            return 0

        if args.command == "train-yolo":
            runs = train_yolo_models(
                models=args.models,
                data_path=args.data_path,
                project_dir=args.project_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                image_size=args.image_size,
                optimizer=args.optimizer,
                seed=args.seed,
                patience=args.patience,
                device=args.device,
                wandb_project=args.wandb_project,
                family=args.family,
            )
            print(f"Completed {len(runs)} training runs in {args.project_dir}")
            return 0

        if args.command == "train-detr":
            summary = train_hf_detr(
                args.train_image_dir,
                args.train_annotations,
                args.output_dir,
                model_name=args.model_name,
                num_labels=args.num_labels,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                val_image_dir=args.val_image_dir,
                val_annotations=args.val_annotations,
                device=args.device,
            )
            print(f"DETR training summary: {asdict(summary)}")
            return 0

        if args.command == "predict-detr":
            detections = predict_hf_detr(
                args.image_path,
                args.model_path,
                threshold=args.threshold,
                device=args.device,
            )
            print(json.dumps(detections, indent=2))
            return 0

        if args.command == "train-mask2former":
            summary = train_mask2former(
                args.train_image_dir,
                args.train_annotations,
                args.output_dir,
                model_name=args.model_name,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                device=args.device,
            )
            print(f"Mask2Former training summary: {asdict(summary)}")
            return 0

        if args.command == "export-model":
            output = export_ultralytics_model(
                args.model_path,
                export_format=args.format,
                family=args.family,
                device=args.device,
                image_size=args.image_size,
            )
            print(f"Exported model artifact: {output}")
            return 0

        if args.command == "aggregate-counties":
            df = load_detection_csv(args.input_csv)
            summary = aggregate_detections_by_county(
                df,
                county_column=args.county_column,
                roof_area_column=args.roof_area_column,
            )
            summary.to_csv(args.output_csv, index=False)
            print(f"County summary CSV written to {args.output_csv}")
            return 0

        if args.command == "merge-overlapping-shapefile":
            output = merge_overlapping_polygons_with_largest_area_attr(
                args.input_shapefile,
                args.output_shapefile,
            )
            print(f"Merged shapefile written to {output}")
            return 0

        if args.command == "process-shapefile-dimensions":
            output = process_shapefile_with_bbox_dimensions(
                args.input_shapefile,
                args.output_shapefile,
            )
            print(f"Dimension-enriched shapefile written to {output}")
            return 0

        if args.command == "filter-shapefile":
            output = filter_shapefile_by_length_width_area_proximity(
                args.input_shapefile,
                args.output_shapefile,
                length_min=args.length_min,
                width_min=args.width_min,
                width_max=args.width_max,
                area_min=args.area_min,
                area_max=args.area_max,
                proximity_m=args.proximity_m,
                always_keep_class_zero=not args.disable_class_zero_keep,
            )
            print(f"Filtered shapefile written to {output}")
            return 0

        if args.command == "identify-farms-shapefile":
            output = identify_poultry_farms_from_shapefile(
                args.input_shapefile,
                args.output_shapefile,
                proximity_m=args.proximity_m,
                min_samples=args.min_samples,
            )
            print(f"Farm shapefile written to {output}")
            return 0

        if args.command == "runoff-screening":
            county_summary = pd.read_csv(args.county_summary_csv)
            if not args.skip_nass_correction:
                county_summary = apply_nass_farm_correction(
                    county_summary,
                    nass_farms_column=args.nass_farms_column,
                    detected_farms_column=args.detected_farms_column,
                )
            precipitation = pd.read_csv(args.precipitation_csv)
            runoff = compute_county_runoff_screening(
                county_summary,
                precipitation,
                county_column=args.county_column,
                roof_area_column=args.roof_area_column,
                runoff_coefficient=args.runoff_coefficient,
            )
            runoff.to_csv(args.output_csv, index=False)
            print(f"County runoff CSV written to {args.output_csv}")
            if args.adjustment_factors_csv:
                annual_volume_column = (
                    "nass_adjusted_annual_runoff_m3"
                    if "nass_adjusted_annual_runoff_m3" in runoff.columns
                    else "annual_runoff_m3"
                )
                summary = bootstrap_adjusted_runoff(
                    runoff,
                    adjustment_factors=load_adjustment_factors(
                        args.adjustment_factors_csv,
                        column=args.adjustment_column,
                    ),
                    annual_volume_column=annual_volume_column,
                )
                print(f"Bootstrap summary: {asdict(summary)}")
            return 0

        if args.command == "run-pipeline":
            outputs = run_pipeline(load_pipeline_config(args.config_path))
            print(f"Pipeline completed. Summary written to {outputs['summary_json']}")
            return 0

        parser.error(f"Unknown command: {args.command}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
