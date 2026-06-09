import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import argparse

def set_academic_style():
    """
    Configures matplotlib with publication-ready, academic journal styling.
    Uses serif fonts, proper font sizing, high DPI, and clean gridlines.
    """
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman', 'serif'],
        'mathtext.fontset': 'stix',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.5,
        'grid.linewidth': 1.0,
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.format': 'png',
    })

def plot_training_logs(run_dir: str | Path, output_dir: str | Path):
    """
    Parses training logs (metrics.json, metrics.csv, results.csv) from a run directory 
    and generates publication-ready matplotlib figures suitable for research papers.
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    set_academic_style()
    
    json_path = run_dir / "metrics.json"
    csv_path = run_dir / "metrics.csv"
    yolo_csv = run_dir / "results.csv"
    
    history = []
    is_yolo = False
    
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
            if "history" in data:
                history = data["history"]
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        history = df.to_dict('records')
    elif yolo_csv.exists():
        df = pd.read_csv(yolo_csv)
        df.columns = [c.strip() for c in df.columns]
        is_yolo = True
    else:
        raise FileNotFoundError(f"No metrics.json, metrics.csv, or results.csv found in {run_dir}")
        
    if not is_yolo and not history:
        raise ValueError("Found metrics file but history was empty.")
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if is_yolo:
        epochs = df['epoch']
        if 'train/loss' in df.columns:
            ax.plot(epochs, df['train/loss'], label='Training Loss', color='#004c99', marker='o', markersize=6)
        if 'val/loss' in df.columns:
            ax.plot(epochs, df['val/loss'], label='Validation Loss', color='#cc0000', marker='s', markersize=6)
        ax.set_title('YOLO Fine-Tuning Convergence')
        ax.set_ylabel('Box/Class/DFL Loss')
    else:
        df_hist = pd.DataFrame(history)
        epochs = df_hist.get('epoch', df_hist.index + 1)
        
        if 'train_loss' in df_hist.columns:
            ax.plot(epochs, df_hist['train_loss'], label='Training Loss', color='#004c99', marker='o', markersize=6)
        if 'val_loss' in df_hist.columns:
            ax.plot(epochs, df_hist['val_loss'], label='Validation Loss', color='#cc0000', marker='s', markersize=6)
        
        ax.set_title('Self-Supervised Pretraining Convergence')
        ax.set_ylabel('Reconstruction Loss (MSE)')
        
    ax.set_xlabel('Training Epoch')
    ax.grid(True)
    
    if len(ax.get_legend_handles_labels()[1]) > 0:
        ax.legend(loc='upper right')
    
    # Save the figure with high resolution
    out_file = output_dir / f"{run_dir.name}_loss_curve.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_grid_search_results(results_csv: str | Path, output_dir: str | Path):
    """
    Generates professional-grade summary visualizations for grid search campaigns.
    Creates a performance comparison across different hyperparameter combinations.
    """
    results_csv = Path(results_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_csv.exists():
        raise FileNotFoundError(f"Grid search results not found: {results_csv}")
        
    df = pd.read_csv(results_csv)
    successful = df[df['status'] == 'ok'].copy()
    
    if successful.empty:
        print("No successful runs found to plot.")
        return None
        
    set_academic_style()
    
    # Sort by best loss
    successful = successful.sort_values('best_loss', ascending=True)
    top_n = min(len(successful), 15)
    successful = successful.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create descriptive labels for the y-axis
    labels = []
    for _, row in successful.iterrows():
        label = f"{row['model_name']} | {row['optimizer']} | lr={row['learning_rate']} | {row['precision']}"
        labels.append(label)
        
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, successful['best_loss'], color='#004c99', edgecolor='black', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Best on top
    ax.set_xlabel('Minimum Reconstruction Loss (MSE)')
    ax.set_title(f'SSL Grid Search: Top {top_n} Experiment Performance')
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    
    # Add loss values to the end of bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (width * 0.01), bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', va='center', fontsize=10, fontweight='bold')

    out_file = output_dir / "grid_search_summary.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Professional visualization of training and grid search metrics.")
    subparsers = parser.add_subparsers(dest="command")
    
    run_parser = subparsers.add_parser("plot-run", help="Plot metrics for a single run.")
    run_parser.add_argument("run_dir", help="Directory containing metrics.json/csv")
    run_parser.add_argument("--output-dir", default="plots", help="Where to save plots")
    
    grid_parser = subparsers.add_parser("plot-grid", help="Plot grid search summary results.")
    grid_parser.add_argument("results_csv", help="Path to grid_search_results.csv")
    grid_parser.add_argument("--output-dir", default="plots", help="Where to save summary plots")
    
    args = parser.parse_args()
    
    if args.command == "plot-run":
        out = plot_training_logs(args.run_dir, args.output_dir)
        print(f"Plot saved to {out}")
    elif args.command == "plot-grid":
        out = plot_grid_search_results(args.results_csv, args.output_dir)
        if out:
            print(f"Summary plot saved to {out}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
