from __future__ import annotations

import argparse
import json
import sys
import warnings
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
from poultry_geo.updated_train import run_stateful_grid_search
class RichHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show defaults while preserving hand-written help layout."""


class RichArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", RichHelpFormatter)
        super().__init__(*args, **kwargs)


def _resolve_torch_device_alias(device: str | None):
    import torch

    if device is None or device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    normalized = device.strip().lower()
    if normalized == "gpu":
        if not torch.cuda.is_available():
            raise ValueError("Requested device 'gpu', but CUDA is not available.")
        return torch.device("cuda")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested device 'cuda', but CUDA is not available.")
        return torch.device("cuda")
    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(f"Requested device '{device}', but CUDA is not available.")
        return torch.device(normalized)
    if normalized == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Requested device 'mps', but Apple Metal is not available.")
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")
    return torch.device(device)


def _resolve_data_precision(precision: str) -> str:
    return "bf16" if precision == "nf4" else precision


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

    inspect_backbone_parser = _add_command_parser(
        subparsers,
        "inspect-backbone",
        help_text="Inspect adapter and SSL ViT tensor shapes on one NAIP batch.",
        description=dedent(
            """
            Load one batch from the NAIP self-supervised dataloader, adapt n-band imagery
            down to three channels, and run a ViT backbone forward pass.

            Use this to validate shape flow, precision resolution, device placement, and
            patch-grid geometry before training or plugging in SSL losses.
            """
        ).strip(),
        epilog=dedent(
            """
            Examples:
              poultry-geo inspect-backbone tmp/test_jpg --batch-size 2 --crop-size 224 --channels 3 --patch-size 16 --embed-dim 192 --depth 12 --num-heads 3
              poultry-geo inspect-backbone tmp/test_jpg --batch-size 4 --crop-size 224 --channels 4 --precision bf16 --patch-size 16 --embed-dim 384 --depth 8 --num-heads 6 --gradient-checkpointing
            """
        ).strip(),
    )
    inspect_backbone_parser.add_argument("data_root")
    inspect_backbone_parser.add_argument("--batch-size", type=int, default=2)
    inspect_backbone_parser.add_argument("--crop-size", type=int, default=224)
    inspect_backbone_parser.add_argument("--channels", type=int, default=None)
    inspect_backbone_parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16", "nf4"])
    inspect_backbone_parser.add_argument("--patch-size", type=int, default=16)
    inspect_backbone_parser.add_argument("--embed-dim", type=int, required=True)
    inspect_backbone_parser.add_argument("--depth", type=int, required=True)
    inspect_backbone_parser.add_argument("--num-heads", type=int, required=True)
    inspect_backbone_parser.add_argument("--mlp-ratio", type=float, default=4.0)
    inspect_backbone_parser.add_argument("--num-workers", type=int, default=4)
    inspect_backbone_parser.add_argument("--device", default=None, help="Execution device, for example cpu or cuda:0.")
    inspect_backbone_parser.add_argument("--gradient-checkpointing", action="store_true")

    inspect_segmentation_parser = _add_command_parser(
        subparsers,
        "inspect-segmentation",
        help_text="Inspect LoRA-adapted ViT segmentation output shapes on one NAIP batch.",
        description=dedent(
            """
            Load one batch from the NAIP self-supervised dataloader, run the band adapter,
            optional LoRA-augmented ViT backbone, and a lightweight decoder head that
            reconstructs a binary roof mask at full image resolution.

            Use this to validate freezing behavior, LoRA wiring, precision resolution,
            and segmentation output shape before training.
            """
        ).strip(),
        epilog=dedent(
            """
            Examples:
              poultry-geo inspect-segmentation tmp/test_jpg --batch-size 2 --crop-size 256 --channels 3 --patch-size 16 --embed-dim 192 --depth 8 --num-heads 6 --use-lora --freeze-backbone
              poultry-geo inspect-segmentation tmp/test_jpg --batch-size 2 --crop-size 512 --channels 4 --precision bf16 --patch-size 16 --embed-dim 384 --depth 12 --num-heads 6 --use-lora --lora-rank 8 --decoder-hidden-channels 128
            """
        ).strip(),
    )
    inspect_segmentation_parser.add_argument("data_root")
    inspect_segmentation_parser.add_argument("--batch-size", type=int, default=2)
    inspect_segmentation_parser.add_argument("--crop-size", type=int, default=256)
    inspect_segmentation_parser.add_argument("--channels", type=int, default=None)
    inspect_segmentation_parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16", "nf4"])
    inspect_segmentation_parser.add_argument("--patch-size", type=int, default=16)
    inspect_segmentation_parser.add_argument("--embed-dim", type=int, required=True)
    inspect_segmentation_parser.add_argument("--depth", type=int, required=True)
    inspect_segmentation_parser.add_argument("--num-heads", type=int, required=True)
    inspect_segmentation_parser.add_argument("--mlp-ratio", type=float, default=4.0)
    inspect_segmentation_parser.add_argument("--num-workers", type=int, default=4)
    inspect_segmentation_parser.add_argument("--device", default=None, help="Execution device, for example cpu, gpu, cuda:0, or mps.")
    inspect_segmentation_parser.add_argument("--gradient-checkpointing", action="store_true")
    inspect_segmentation_parser.add_argument("--use-lora", action="store_true")
    inspect_segmentation_parser.add_argument("--freeze-backbone", action="store_true")
    inspect_segmentation_parser.add_argument("--lora-rank", type=int, default=4)
    inspect_segmentation_parser.add_argument("--lora-alpha", type=float, default=8.0)
    inspect_segmentation_parser.add_argument("--lora-dropout", type=float, default=0.0)
    inspect_segmentation_parser.add_argument("--decoder-hidden-channels", type=int, default=128)

    train_mim_parser = _add_command_parser(
        subparsers,
        "train-mim",
        help_text="Train a masked image modeling SSL run on NAIP tiles.",
        description=dedent(
            """
            Train a MAE-style masked image modeling objective over NAIP tiles using the
            custom band adapter and RemoteSensingViT backbone.

            This command supports FP32, FP16, BF16, and NF4/QLoRA-style low-memory runs.
            """
        ).strip(),
        epilog=dedent(
            """
            Examples:
              poultry-geo train-mim tmp/test_jpg runs/mim --batch-size 4 --crop-size 224 --channels 3 --patch-size 16 --embed-dim 192 --depth 8 --num-heads 6 --epochs 1 --precision bf16
              poultry-geo train-mim tmp/test_jpg runs/mim-nf4 --batch-size 4 --crop-size 224 --channels 4 --patch-size 16 --embed-dim 192 --depth 8 --num-heads 6 --epochs 1 --precision nf4 --use-lora --freeze-backbone
            """
        ).strip(),
    )
    train_mim_parser.add_argument("data_root")
    train_mim_parser.add_argument("output_dir")
    train_mim_parser.add_argument("--batch-size", type=int, default=4)
    train_mim_parser.add_argument("--crop-size", type=int, default=224)
    train_mim_parser.add_argument("--channels", type=int, default=None)
    train_mim_parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16", "nf4"])
    train_mim_parser.add_argument("--model-name", choices=["S", "B", "L", "H"], help="Standard ViT variant preset.")
    train_mim_parser.add_argument("--patch-size", type=int, default=16)
    train_mim_parser.add_argument("--embed-dim", type=int)
    train_mim_parser.add_argument("--depth", type=int)
    train_mim_parser.add_argument("--num-heads", type=int)
    train_mim_parser.add_argument("--mlp-ratio", type=float, default=4.0)
    train_mim_parser.add_argument("--mask-ratio", type=float, default=0.75)
    train_mim_parser.add_argument("--num-workers", type=int, default=4)
    train_mim_parser.add_argument("--epochs", type=int, default=1)
    train_mim_parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adan", "sophia", "lion"])
    train_mim_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_mim_parser.add_argument("--weight-decay", type=float, default=0.05)
    train_mim_parser.add_argument("--device", default=None, help="Execution device, for example cpu, gpu, cuda:0, or mps.")
    train_mim_parser.add_argument("--gradient-checkpointing", action="store_true")
    train_mim_parser.add_argument("--use-lora", action="store_true")
    train_mim_parser.add_argument("--freeze-backbone", action="store_true")
    train_mim_parser.add_argument("--lora-rank", type=int, default=4)
    train_mim_parser.add_argument("--lora-alpha", type=float, default=8.0)
    train_mim_parser.add_argument("--lora-dropout", type=float, default=0.0)

    train_spark_parser = _add_command_parser(
        subparsers,
        "train-spark",
        help_text="Train a Sparse Masked Modeling (SparK) SSL run on NAIP tiles.",
        description=dedent(
            """
            Train a SparK-style masked image modeling objective using a sparse hierarchical 
            encoder (ResNet50 or CSPDarkNet) and a dense hierarchical decoder.
            """
        ).strip(),
        epilog=dedent(
            """
            Examples:
              poultry-geo train-spark tmp/test_jpg runs/spark --batch-size 4 --backbone-type resnet50 --precision bf16
              poultry-geo train-spark tmp/test_jpg runs/spark-csp --batch-size 4 --backbone-type cspdarknet --mask-ratio 0.75
            """
        ).strip(),
    )
    train_spark_parser.add_argument("data_root")
    train_spark_parser.add_argument("output_dir")
    train_spark_parser.add_argument("--batch-size", type=int, default=4)
    train_spark_parser.add_argument("--crop-size", type=int, default=224)
    train_spark_parser.add_argument("--channels", type=int, default=None)
    train_spark_parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    train_spark_parser.add_argument("--backbone-type", default="resnet50", choices=["resnet50", "cspdarknet", "yolo11l", "yolov26l"])
    train_spark_parser.add_argument("--mask-ratio", type=float, default=0.65)
    train_spark_parser.add_argument("--patch-size", type=int, default=16)
    train_spark_parser.add_argument("--num-workers", type=int, default=4)
    train_spark_parser.add_argument("--epochs", type=int, default=1)
    train_spark_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_spark_parser.add_argument("--device", default=None, help="Execution device, for example cpu, gpu, cuda:0, or mps.")

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

    updated_train_parser = _add_command_parser(
        subparsers,
        "updated-train",
        help_text="Run stateful resume-aware segmentation grid search training.",
    )
    updated_train_parser.add_argument("training_name", help="Run identifier or 'training_name=<name>'.")
    updated_train_parser.add_argument("--config", default="config.yaml", help="Path to the grid-search YAML config.")
    updated_train_parser.add_argument("--epochs", type=int, default=None, help="Override runtime.epochs from the config.")
    updated_train_parser.add_argument("--root-dir", default="trained_models", help="Root directory for saved runs.")

    ssl_grid_parser = _add_command_parser(
        subparsers,
        "ssl-grid-train",
        help_text="Run stateful resume-aware SSL (MIM) grid search training.",
    )
    ssl_grid_parser.add_argument("training_name", help="Run identifier.")
    ssl_grid_parser.add_argument("--config", default="ssl_config.yaml", help="Path to the SSL grid-search YAML config.")
    ssl_grid_parser.add_argument("--epochs", type=int, default=None, help="Override runtime.epochs from the config.")
    ssl_grid_parser.add_argument("--root-dir", default="trained_models", help="Root directory for saved runs.")

    spark_grid_parser = _add_command_parser(
        subparsers,
        "spark-grid-train",
        help_text="Run stateful resume-aware SparK (SSL) grid search training.",
    )
    spark_grid_parser.add_argument("training_name", help="Run identifier.")
    spark_grid_parser.add_argument("--config", default="configs/spark_grid_search.yaml", help="Path to the SparK grid-search YAML config.")
    spark_grid_parser.add_argument("--epochs", type=int, default=None, help="Override runtime.epochs from the config.")

    # Visualization commands
    plot_run_parser = _add_command_parser(
        subparsers,
        "plot-run",
        help_text="Generate professional-grade training plots for a single run.",
    )
    plot_run_parser.add_argument("run_dir", help="Directory containing metrics.json/csv")
    plot_run_parser.add_argument("--output-dir", default="plots", help="Directory to save plots")

    grid_plot_parser = _add_command_parser(
        subparsers,
        "ssl-grid-plot",
        help_text="Generate a professional-grade summary plot for an SSL grid search campaign.",
    )
    grid_plot_parser.add_argument("results_csv", help="Path to grid_search_results.csv")
    grid_plot_parser.add_argument("--output-dir", default="plots", help="Directory to save summary plots")
    spark_grid_parser.add_argument("--root-dir", default="trained_models", help="Root directory for saved runs.")

    plot_parser = _add_command_parser(
        subparsers,
        "plot-training-logs",
        help_text="Generate publication-ready plots from training log files (metrics.json or metrics.csv).",
    )
    plot_parser.add_argument("run_dir", help="Directory containing metrics.json, metrics.csv, or results.csv.")
    plot_parser.add_argument("output_dir", help="Directory to save the generated figures.")

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

        if args.command == "inspect-backbone":
            import torch
            import poultry_geo.naip_data as naip_data
            import poultry_geo.ssl_models as ssl_models

            device = _resolve_torch_device_alias(args.device)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*pin_memory.*no accelerator is found.*",
                    category=UserWarning,
                )
                train_loader, _ = naip_data.get_dataloaders(
                    args.data_root,
                    batch_size=args.batch_size,
                    crop_size=args.crop_size,
                    channels=args.channels,
                    precision=args.precision,
                    num_workers=args.num_workers,
                )
                batch = next(iter(train_loader))
            images = batch["image"]
            images = images.to(device=device, non_blocking=True)

            adapter = ssl_models.BandAdapter(
                in_channels=int(images.shape[1]),
                out_channels=3,
                precision=args.precision,
            ).to(device)
            backbone = ssl_models.RemoteSensingViT(
                image_size=args.crop_size,
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                depth=args.depth,
                num_heads=args.num_heads,
                mlp_ratio=args.mlp_ratio,
                precision=args.precision,
                gradient_checkpointing=args.gradient_checkpointing,
            ).to(device)
            adapter.eval()
            backbone.eval()

            with torch.no_grad():
                adapted = adapter(images)
                tokens = backbone.forward_features(adapted)

            patch_grid = (
                int(adapted.shape[2]) // args.patch_size,
                int(adapted.shape[3]) // args.patch_size,
            )
            print(
                "inspect-backbone "
                f"input_shape={tuple(images.shape)} "
                f"adapted_shape={tuple(adapted.shape)} "
                f"token_shape={tuple(tokens.shape)} "
                f"patch_grid={patch_grid} "
                f"requested_precision={backbone.requested_precision} "
                f"resolved_precision={backbone.resolved_precision} "
                f"device={device}"
            )
            return 0

        if args.command == "inspect-segmentation":
            import torch
            import poultry_geo.naip_data as naip_data
            import poultry_geo.ssl_models as ssl_models

            device = _resolve_torch_device_alias(args.device)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*pin_memory.*no accelerator is found.*",
                    category=UserWarning,
                )
                train_loader, _ = naip_data.get_dataloaders(
                    args.data_root,
                    batch_size=args.batch_size,
                    crop_size=args.crop_size,
                    channels=args.channels,
                    precision=args.precision,
                    num_workers=args.num_workers,
                )
                batch = next(iter(train_loader))
            images = batch["image"]
            images = images.to(device=device, non_blocking=True)

            model = ssl_models.RemoteSensingSegmentationModel(
                in_channels=int(images.shape[1]),
                image_size=args.crop_size,
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                depth=args.depth,
                num_heads=args.num_heads,
                precision=args.precision,
                gradient_checkpointing=args.gradient_checkpointing,
                use_lora=args.use_lora,
                freeze_backbone=args.freeze_backbone,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                decoder_hidden_channels=args.decoder_hidden_channels,
                mlp_ratio=args.mlp_ratio,
            ).to(device)
            model.eval()

            with torch.no_grad():
                outputs = model.forward_with_intermediates(images)

            patch_grid = (
                int(outputs["adapted"].shape[2]) // args.patch_size,
                int(outputs["adapted"].shape[3]) // args.patch_size,
            )
            print(
                "inspect-segmentation "
                f"input_shape={tuple(images.shape)} "
                f"adapted_shape={tuple(outputs['adapted'].shape)} "
                f"token_shape={tuple(outputs['tokens'].shape)} "
                f"mask_shape={tuple(outputs['mask'].shape)} "
                f"patch_grid={patch_grid} "
                f"requested_precision={model.requested_precision} "
                f"resolved_precision={model.resolved_precision} "
                f"use_lora={model.use_lora} "
                f"freeze_backbone={model.freeze_backbone} "
                f"device={device}"
            )
            return 0

        if args.command == "train-mim":
            import poultry_geo.naip_data as naip_data
            import poultry_geo.ssl_models as ssl_models
            import poultry_geo.training as training

            device = _resolve_torch_device_alias(args.device)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*pin_memory.*no accelerator is found.*",
                    category=UserWarning,
                )
                train_loader, val_loader = naip_data.get_dataloaders(
                    args.data_root,
                    batch_size=args.batch_size,
                    crop_size=args.crop_size,
                    channels=args.channels,
                    precision=_resolve_data_precision(args.precision),
                    num_workers=args.num_workers,
                )

            quantization_config = (
                ssl_models.build_nf4_quantization_config()
                if args.precision == "nf4"
                else None
            )
            model_in_channels = args.channels or getattr(getattr(train_loader, "dataset", None), "_expected_channels", 3)
            from poultry_geo.ssl_models import VIT_CONFIGS
            
            embed_dim = args.embed_dim
            depth = args.depth
            num_heads = args.num_heads
            
            if args.model_name:
                config = VIT_CONFIGS[args.model_name]
                embed_dim = config["embed_dim"]
                depth = config["depth"]
                num_heads = config["num_heads"]
            
            if embed_dim is None or depth is None or num_heads is None:
                parser.error("Either --model-name or all of (--embed-dim, --depth, --num-heads) must be provided.")

            model = ssl_models.RemoteSensingMIMModel(
                in_channels=model_in_channels,
                image_size=args.crop_size,
                patch_size=args.patch_size,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                precision=args.precision,
                mask_ratio=args.mask_ratio,
                gradient_checkpointing=args.gradient_checkpointing,
                use_lora=args.use_lora,
                freeze_backbone=args.freeze_backbone,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                mlp_ratio=args.mlp_ratio,
                quantization_config=quantization_config,
            )
            from poultry_geo.optimizers import build_optimizer
            optimizer = build_optimizer(
                args.optimizer,
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            trainer = training.SSLTrainer(
                model,
                train_loader,
                optimizer,
                val_loader=val_loader,
                device=device,
                precision=args.precision,
                progress_callback=progress_callback,
            )
            summary = trainer.fit(args.epochs, args.output_dir)
            print(
                "train-mim "
                f"output_dir={summary.output_dir} "
                f"epochs={summary.epochs} "
                f"train_batches={summary.train_batches} "
                f"val_batches={summary.val_batches} "
                f"final_train_loss={summary.final_train_loss} "
                f"final_val_loss={summary.final_val_loss} "
                f"requested_precision={model.requested_precision} "
                f"resolved_precision={model.resolved_precision} "
                f"device={device}"
            )
            return 0

        if args.command == "train-spark":
            import os
            import torch
            import poultry_geo.naip_data as naip_data
            import poultry_geo.spark as spark
            import poultry_geo.spark_models as spark_models
            import poultry_geo.train_spark_ssl as train_spark_ssl
            import poultry_geo.models as models

            device = _resolve_torch_device_alias(args.device)
            train_loader, val_loader = naip_data.get_dataloaders(
                args.data_root,
                batch_size=args.batch_size,
                crop_size=args.crop_size,
                channels=args.channels,
                precision=_resolve_data_precision(args.precision),
                num_workers=args.num_workers,
            )

            # Build model
            in_channels = args.channels or getattr(getattr(train_loader, "dataset", None), "_expected_channels", 3)
            adapter = models.BandAdapter(in_channels=in_channels, out_channels=3)
            transformer = spark.SparseTransformer(
                adapter=adapter, 
                patch_size=args.patch_size, 
                mask_ratio=args.mask_ratio
            ).to(device)
            model = spark_models.SparKNetwork(
                backbone_type=args.backbone_type, 
                in_channels=3
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            scaler = torch.cuda.amp.GradScaler(enabled=(args.precision != "fp32"))
            config = train_spark_ssl.SparKConfig(
                mask_ratio=args.mask_ratio,
                backbone_type=args.backbone_type,
                precision=args.precision.upper(),
                lr=args.learning_rate,
                batch_size=args.batch_size
            )

            print(f"Starting SparK training on {device}...")
            os.makedirs(args.output_dir, exist_ok=True)
            
            import pandas as pd
            import time
            import json
            import platform
            history = []
            
            # Record initial experiment metadata
            experiment_info = {
                "system": platform.platform(),
                "python_version": platform.python_version(),
                "device": str(device),
                "args": vars(args),
                "start_time": time.time()
            }
            with open(os.path.join(args.output_dir, "experiment_info.json"), "w") as f:
                json.dump(experiment_info, f, indent=4)
            
            for epoch in range(args.epochs):
                model.train()
                epoch_loss = 0
                epoch_start_time = time.time()
                for i, batch in enumerate(train_loader):
                    img = batch["image"].to(device)
                    loss = train_spark_ssl.train_one_step(
                        model, optimizer, scaler, transformer, img, config
                    )
                    epoch_loss += loss
                    if i % 10 == 0:
                        print(f"Epoch {epoch} | Batch {i} | Loss: {loss:.4f}")
                
                epoch_duration = time.time() - epoch_start_time
                avg_loss = epoch_loss / len(train_loader)
                print(f"Epoch {epoch} complete in {epoch_duration:.2f}s. Avg Loss: {avg_loss:.4f}")
                
                # Record metrics
                history.append({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "lr": args.learning_rate,
                    "epoch_duration_seconds": epoch_duration
                })
                pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
                
                # Save checkpoints
                train_spark_ssl.save_spark_checkpoint(model, config, args.output_dir, epoch)

            experiment_info["end_time"] = time.time()
            experiment_info["total_duration_seconds"] = experiment_info["end_time"] - experiment_info["start_time"]
            with open(os.path.join(args.output_dir, "experiment_info.json"), "w") as f:
                json.dump(experiment_info, f, indent=4)

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

        if args.command == "updated-train":
            results = run_stateful_grid_search(
                config_path=args.config,
                training_name=args.training_name,
                epochs=args.epochs,
                root_dir=args.root_dir,
                progress_callback=progress_callback,
            )
            print(results.to_string(index=False))
            return 0

        if args.command == "ssl-grid-train":
            from poultry_geo.ssl_grid_search import run_stateful_ssl_grid_search
            results = run_stateful_ssl_grid_search(
                config_path=args.config,
                training_name=args.training_name,
                epochs=args.epochs,
                root_dir=args.root_dir,
                progress_callback=progress_callback,
            )
            print(results.to_string(index=False))
            return 0

        if args.command == "spark-grid-train":
            from poultry_geo.spark_grid_search import run_stateful_spark_grid_search
            results = run_stateful_spark_grid_search(
                config_path=args.config,
                training_name=args.training_name,
                epochs=args.epochs,
                root_dir=args.root_dir,
                progress_callback=progress_callback,
            )
            print(results.to_string(index=False))
            return 0

        if args.command == "plot-run":
            from poultry_geo.plot_metrics import plot_training_logs
            out_file = plot_training_logs(args.run_dir, args.output_dir)
            print(f"Generated plot written to {out_file}")
            return 0

        if args.command == "ssl-grid-plot":
            from poultry_geo.plot_metrics import plot_grid_search_results
            out_file = plot_grid_search_results(args.results_csv, args.output_dir)
            if out_file:
                print(f"Grid search summary plot written to {out_file}")
            return 0

        parser.error(f"Unknown command: {args.command}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
