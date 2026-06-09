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
