"""Build traceable manuscript figures from the frozen AL-MS experiment artifacts.

The script deliberately redraws the recorded CSV metrics and renders OBBs from
the original validation images. It does not use held-test imagery, choose
examples based on a model's outcome, or alter training artifacts.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe -m scripts.make_compag_manuscript_figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image


SEEDS = (7, 17, 42)
BASELINE = "baseline"
ANCHORED = "anchored_ssl"
MAP_KEY = "metrics/mAP50-95(B)"
COLORS = {
    "baseline": "#3B5B92",
    "anchored": "#008B8B",
    "breeder": "#2C7FB8",
    "house": "#D95F02",
    "train": "#4C78A8",
    "validation": "#008B8B",
    "ink": "#20252B",
    "grid": "#D9DEE5",
}
CLASS_NAMES = ("Broiler breeder", "Broiler house")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the parent of scripts/).",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Ultralytics inference device used only for validation visualizations.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Build metric and data-domain plots without regenerating qualitative predictions.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], column: str) -> float:
    value = row[column]
    return float(value) if value else float("nan")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.03,
        )
    plt.close(fig)


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)


def draw_polygons(
    axis: plt.Axes,
    polygons: np.ndarray,
    classes: Iterable[int],
    confidences: Iterable[float] | None = None,
    linewidth: float = 1.5,
) -> None:
    """Draw OBB quadrilaterals using the two class colors used throughout."""
    confidence_values = list(confidences) if confidences is not None else None
    for index, (polygon, class_id) in enumerate(zip(polygons, classes)):
        class_id = int(class_id)
        color = COLORS["breeder"] if class_id == 0 else COLORS["house"]
        closed = np.vstack([polygon, polygon[0]])
        axis.plot(closed[:, 0], closed[:, 1], color=color, linewidth=linewidth)
        if confidence_values is not None:
            confidence = float(confidence_values[index])
            x, y = polygon[0]
            axis.text(
                x,
                y,
                f"{confidence:.2f}",
                color="white",
                fontsize=6.5,
                ha="left",
                va="bottom",
                bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.9, "pad": 0.8},
            )


def label_path_for_image(image_path: Path) -> Path:
    dataset_root = image_path.parents[2]
    return dataset_root / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def read_ground_truth(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    label_path = label_path_for_image(image_path)
    image_width, image_height = Image.open(image_path).size
    polygons: list[np.ndarray] = []
    classes: list[int] = []
    with label_path.open(encoding="utf-8") as handle:
        for line in handle:
            values = line.split()
            if len(values) < 9:
                continue
            classes.append(int(float(values[0])))
            points = np.asarray([float(value) for value in values[1:9]], dtype=float).reshape(4, 2)
            points[:, 0] *= image_width
            points[:, 1] *= image_height
            polygons.append(points)
    if not polygons:
        return np.empty((0, 4, 2), dtype=float), np.empty((0,), dtype=int)
    return np.stack(polygons), np.asarray(classes, dtype=int)


def display_image(axis: plt.Axes, image_path: Path) -> None:
    with Image.open(image_path) as image:
        axis.imshow(np.asarray(image.convert("RGB")))
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#AAB2BD")
        spine.set_linewidth(0.55)


def make_data_domains(root: Path, output_dir: Path) -> None:
    source_image = Path(
        r"E:\NAIP_Dataset_PH\MS_tiles_512_256_jpg\m_3008803_ne_16_060_20230802_20240103\slice_0_23.jpg"
    )
    breeder_image = root / (
        "datasets/poultry_clean_raw_90_10/images/val/"
        "tile_0_4_jpg.rf.000153d254f7fecd9fd37548c59c259b.jpg"
    )
    house_image = root / (
        "datasets/poultry_clean_raw_90_10/images/val/"
        "tile_3_2_jpg.rf.ebc7673e8fa9702c3168fefb2cecea8e.jpg"
    )
    for path in (source_image, breeder_image, house_image):
        if not path.exists():
            raise FileNotFoundError(path)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55))
    titles = (
        "(a) Unlabeled MS NAIP tile",
        "(b) AL reference OBB: breeder",
        "(c) AL reference OBB: house",
    )
    for axis, path, title in zip(axes, (source_image, breeder_image, house_image), titles):
        display_image(axis, path)
        axis.set_title(title, pad=5, fontsize=8.2)
    for axis, target_path in zip(axes[1:], (breeder_image, house_image)):
        polygons, classes = read_ground_truth(target_path)
        draw_polygons(axis, polygons, classes, linewidth=1.4)

    legend_handles = [
        Line2D([0], [0], color=COLORS["breeder"], lw=2, label=CLASS_NAMES[0]),
        Line2D([0], [0], color=COLORS["house"], lw=2, label=CLASS_NAMES[1]),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.035),
    )
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.16, top=0.88, wspace=0.055)
    save_figure(fig, output_dir, "fig_data_domains")


def make_training_dynamics(root: Path, output_dir: Path) -> None:
    run_root = root / "runs/ssl_al_ms_anchored_dinov2_v3"
    ssl_rows = read_csv_rows(run_root / "00_ms_pretraining/metrics.csv")
    epochs = np.asarray([as_float(row, "epoch") + 1 for row in ssl_rows])
    train_total = np.asarray([as_float(row, "train_total") for row in ssl_rows])
    val_total = np.asarray([as_float(row, "val_total") for row in ssl_rows])
    best_index = int(np.nanargmin(val_total))

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.95), gridspec_kw={"wspace": 0.33})
    loss_axis, map_axis = axes
    loss_axis.plot(epochs, train_total, color=COLORS["train"], linewidth=1.7, label="Training")
    loss_axis.plot(
        epochs,
        val_total,
        color=COLORS["validation"],
        linewidth=1.7,
        label="Internal source-group validation",
    )
    loss_axis.axvline(epochs[best_index], color=COLORS["ink"], linewidth=0.8, linestyle="--")
    loss_axis.scatter(
        [epochs[best_index]], [val_total[best_index]], s=26, color=COLORS["validation"], zorder=3
    )
    loss_axis.annotate(
        f"Best {val_total[best_index]:.4f}\nEpoch {int(epochs[best_index])}",
        xy=(epochs[best_index], val_total[best_index]),
        xytext=(epochs[best_index] - 10, val_total[best_index] + 0.04),
        arrowprops={"arrowstyle": "-", "color": COLORS["ink"], "lw": 0.65},
        fontsize=7.2,
        color=COLORS["ink"],
    )
    loss_axis.set_title("(a) Mississippi DINOv2 distillation objective", loc="left")
    loss_axis.set_xlabel("Adaptation epoch")
    loss_axis.set_ylabel("Total loss")
    loss_axis.set_xlim(1, len(epochs))
    style_axis(loss_axis)
    loss_axis.legend(frameon=False, loc="upper right", handlelength=2.2)

    for method, color, label, marker in (
        (BASELINE, COLORS["baseline"], "Matched YOLO11l-OBB baseline", "o"),
        (ANCHORED, COLORS["anchored"], "Domain-anchored DINOv2", "D"),
    ):
        for seed in SEEDS:
            rows = read_csv_rows(run_root / "01_al_finetune" / method / f"seed{seed}" / "results.csv")
            run_epochs = np.asarray([as_float(row, "epoch") for row in rows])
            metric = np.asarray([as_float(row, MAP_KEY) for row in rows])
            map_axis.plot(run_epochs, metric, color=color, alpha=0.35, linewidth=1.0)
            best_index = int(np.nanargmax(metric))
            map_axis.scatter(
                [run_epochs[best_index]],
                [metric[best_index]],
                marker=marker,
                s=23,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )
        map_axis.plot([], [], color=color, linewidth=1.8, label=label)
    map_axis.set_title("(b) Alabama validation mAP50-95", loc="left")
    map_axis.set_xlabel("Supervised fine-tuning epoch")
    map_axis.set_ylabel("OBB mAP50-95")
    map_axis.set_xlim(1, 100)
    map_axis.set_ylim(0.50, 0.89)
    style_axis(map_axis)
    map_axis.legend(frameon=False, loc="lower right", handlelength=2.2)

    save_figure(fig, output_dir, "fig_training_dynamics")


def load_primary_metrics(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    run_root = root / "runs/ssl_al_ms_anchored_dinov2_v3"
    validation_rows = read_csv_rows(run_root / "validation/per_seed_validation.csv")
    validation: dict[str, dict[int, float]] = {BASELINE: {}, ANCHORED: {}}
    for row in validation_rows:
        validation[row["method"]][int(row["seed"])] = as_float(row, "best_val_map50_95")
    test: dict[str, dict[int, float]] = {BASELINE: {}, ANCHORED: {}}
    for method in (BASELINE, ANCHORED):
        for seed in SEEDS:
            with (run_root / "final_test" / method / f"seed{seed}.json").open(encoding="utf-8") as handle:
                report = json.load(handle)
            test[method][seed] = float(report[MAP_KEY])
    return (
        np.asarray([validation[BASELINE][seed] for seed in SEEDS]),
        np.asarray([validation[ANCHORED][seed] for seed in SEEDS]),
        np.asarray([test[BASELINE][seed] for seed in SEEDS]),
        np.asarray([test[ANCHORED][seed] for seed in SEEDS]),
    )


def make_primary_paired_results(root: Path, output_dir: Path) -> None:
    validation_baseline, validation_ssl, test_baseline, test_ssl = load_primary_metrics(root)
    panels = (
        (
            "(a) Alabama validation",
            validation_baseline,
            validation_ssl,
            "Mean delta = +0.00212\n95% paired interval [-0.00789, +0.01212]",
        ),
        (
            "(b) Observed Alabama test",
            test_baseline,
            test_ssl,
            "Mean delta = +0.00437\n95% paired interval [-0.00915, +0.01790]",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw={"wspace": 0.36})
    x = np.arange(len(SEEDS))
    for axis, (title, baseline, ssl, summary) in zip(axes, panels):
        for position, baseline_value, ssl_value in zip(x, baseline, ssl):
            axis.plot([position - 0.11, position + 0.11], [baseline_value, ssl_value], color="#737A84", lw=1.05)
            axis.scatter(position - 0.11, baseline_value, color=COLORS["baseline"], s=34, zorder=3)
            axis.scatter(position + 0.11, ssl_value, color=COLORS["anchored"], marker="D", s=34, zorder=3)
        mean_x = 3.10
        axis.plot([mean_x - 0.11, mean_x + 0.11], [baseline.mean(), ssl.mean()], color=COLORS["ink"], lw=1.1)
        axis.scatter(mean_x - 0.11, baseline.mean(), color=COLORS["baseline"], s=38, zorder=3)
        axis.scatter(mean_x + 0.11, ssl.mean(), color=COLORS["anchored"], marker="D", s=38, zorder=3)
        padding = 0.0026 if "validation" in title else 0.0029
        lower = min(baseline.min(), ssl.min()) - padding
        upper = max(baseline.max(), ssl.max()) + padding
        axis.set_ylim(lower, upper)
        axis.set_xlim(-0.55, 3.65)
        axis.set_xticks([*x, mean_x])
        axis.set_xticklabels(["Seed 7", "Seed 17", "Seed 42", "Mean"])
        axis.set_title(title, loc="left")
        axis.set_ylabel("OBB mAP50-95")
        style_axis(axis)
        axis.text(
            0.02,
            0.03,
            summary,
            transform=axis.transAxes,
            fontsize=7.1,
            color=COLORS["ink"],
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "#BFC6CF", "lw": 0.45, "pad": 2.2},
        )
    legend_handles = [
        Line2D([0], [0], color=COLORS["baseline"], marker="o", lw=0, label="Baseline"),
        Line2D([0], [0], color=COLORS["anchored"], marker="D", lw=0, label="Anchored DINOv2"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    save_figure(fig, output_dir, "fig_primary_paired_results")


def get_predictions(checkpoint: Path, image_paths: list[Path], device: str) -> dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    predictions: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for image_path in image_paths:
        result = model.predict(
            source=str(image_path),
            imgsz=1024,
            conf=0.25,
            iou=0.7,
            device=device,
            verbose=False,
            save=False,
            save_txt=False,
        )[0]
        if result.obb is None or len(result.obb) == 0:
            predictions[image_path] = (
                np.empty((0, 4, 2), dtype=float),
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=float),
            )
            continue
        predictions[image_path] = (
            result.obb.xyxyxyxy.cpu().numpy(),
            result.obb.cls.cpu().numpy().astype(int),
            result.obb.conf.cpu().numpy(),
        )
    return predictions


def make_qualitative_detections(root: Path, output_dir: Path, device: str) -> None:
    breeder_image = root / (
        "datasets/poultry_clean_raw_90_10/images/val/"
        "tile_0_4_jpg.rf.000153d254f7fecd9fd37548c59c259b.jpg"
    )
    house_image = root / (
        "datasets/poultry_clean_raw_90_10/images/val/"
        "tile_3_2_jpg.rf.ebc7673e8fa9702c3168fefb2cecea8e.jpg"
    )
    images = [breeder_image, house_image]
    for path in images:
        if not path.exists():
            raise FileNotFoundError(path)
    run_root = root / "runs/ssl_al_ms_anchored_dinov2_v3/01_al_finetune"
    baseline_predictions = get_predictions(run_root / "baseline/seed42/weights/best.pt", images, device)
    anchored_predictions = get_predictions(run_root / "anchored_ssl/seed42/weights/best.pt", images, device)

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.65))
    column_titles = ("Reference OBB", "Matched baseline", "Anchored DINOv2")
    row_titles = ("Breeder facility", "Dense broiler-house facility")
    for row_index, image_path in enumerate(images):
        ground_truth, classes = read_ground_truth(image_path)
        for column_index, axis in enumerate(axes[row_index]):
            display_image(axis, image_path)
            if column_index == 0:
                draw_polygons(axis, ground_truth, classes, linewidth=1.45)
            elif column_index == 1:
                polygons, predicted_classes, _ = baseline_predictions[image_path]
                draw_polygons(axis, polygons, predicted_classes, linewidth=1.45)
            else:
                polygons, predicted_classes, _ = anchored_predictions[image_path]
                draw_polygons(axis, polygons, predicted_classes, linewidth=1.45)
            if row_index == 0:
                axis.set_title(column_titles[column_index], pad=5)
        axes[row_index, 0].set_ylabel(row_titles[row_index], fontsize=8.4, labelpad=6)

    legend_handles = [
        Line2D([0], [0], color=COLORS["breeder"], lw=2, label=CLASS_NAMES[0]),
        Line2D([0], [0], color=COLORS["house"], lw=2, label=CLASS_NAMES[1]),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.52, 0.0),
    )
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.08, top=0.93, hspace=0.055, wspace=0.035)
    save_figure(fig, output_dir, "fig_qualitative_obb")


def make_pseudo_retention(root: Path, output_dir: Path) -> None:
    retention_path = root / "runs/ctq_al_ms_v1/pseudo_retention_diagnostic_v1/retention_by_seed.csv"
    rows = [row for row in read_csv_rows(retention_path) if row["stage"] == "confidence_nms"]
    rows.sort(key=lambda row: int(row["seed"]))
    seeds = [int(row["seed"]) for row in rows]
    retained_images = np.asarray([as_float(row, "images_with_boxes") for row in rows])
    coverage = np.asarray([as_float(row, "image_coverage") * 100 for row in rows])
    breeders = np.asarray([as_float(row, "class_broiler_breeder") for row in rows])
    houses = np.asarray([as_float(row, "class_broiler_house") for row in rows])
    x = np.arange(len(seeds))

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw={"wspace": 0.37})
    images_axis, boxes_axis = axes
    image_bars = images_axis.bar(x, retained_images, color=COLORS["anchored"], width=0.58)
    for bar, count, percent in zip(image_bars, retained_images, coverage):
        images_axis.text(
            bar.get_x() + bar.get_width() / 2,
            count + 0.22,
            f"{int(count)}\n({percent:.3f}%)",
            ha="center",
            va="bottom",
            fontsize=7.4,
        )
    images_axis.set_ylim(0, 8.5)
    images_axis.set_xticks(x)
    images_axis.set_xticklabels([f"Seed {seed}" for seed in seeds])
    images_axis.set_ylabel("Images with pseudo labels")
    images_axis.set_title("(a) Retained images out of 2,048", loc="left")
    style_axis(images_axis)

    boxes_axis.bar(x, breeders, color=COLORS["breeder"], width=0.58, label=CLASS_NAMES[0])
    house_bars = boxes_axis.bar(
        x, houses, bottom=breeders, color=COLORS["house"], width=0.58, label=CLASS_NAMES[1]
    )
    for bar, count in zip(house_bars, houses):
        boxes_axis.text(
            bar.get_x() + bar.get_width() / 2,
            count + 0.5,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    boxes_axis.text(
        0.02,
        0.94,
        "0 breeder pseudo labels for every seed",
        transform=boxes_axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=COLORS["ink"],
    )
    boxes_axis.set_ylim(0, 23)
    boxes_axis.set_xticks(x)
    boxes_axis.set_xticklabels([f"Seed {seed}" for seed in seeds])
    boxes_axis.set_ylabel("Retained pseudo boxes")
    boxes_axis.set_title("(b) Class composition after confidence/NMS", loc="left")
    style_axis(boxes_axis)
    legend_handles = [
        Line2D([0], [0], color=COLORS["breeder"], lw=6, label=CLASS_NAMES[0]),
        Line2D([0], [0], color=COLORS["house"], lw=6, label=CLASS_NAMES[1]),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.72, -0.035),
    )
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.20, top=0.88)
    save_figure(fig, output_dir, "fig_pseudo_retention")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = root / "compag/figures"
    configure_matplotlib()
    make_data_domains(root, output_dir)
    make_training_dynamics(root, output_dir)
    make_primary_paired_results(root, output_dir)
    make_pseudo_retention(root, output_dir)
    if not args.skip_inference:
        make_qualitative_detections(root, output_dir, args.device)
    print(f"Wrote manuscript figures to {output_dir}")


if __name__ == "__main__":
    main()
