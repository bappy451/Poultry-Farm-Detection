import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plot_training_logs(run_dir: str | Path, output_dir: str | Path):
    """
    Parses training logs (metrics.json or metrics.csv) from a run directory 
    and generates publication-ready matplotlib figures.
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for metrics.json (used by MIM and Segmentation)
    json_path = run_dir / "metrics.json"
    csv_path = run_dir / "metrics.csv"
    
    history = []
    
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
            if "history" in data:
                history = data["history"]
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        history = df.to_dict('records')
    else:
        # Check YOLO structure
        yolo_csv = run_dir / "results.csv"
        if yolo_csv.exists():
            df = pd.read_csv(yolo_csv)
            df.columns = [c.strip() for c in df.columns]
            
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            if 'train/loss' in df.columns:
                ax.plot(df['epoch'], df['train/loss'], label='Train Loss', color='#1f77b4', linewidth=2)
            if 'val/loss' in df.columns:
                ax.plot(df['epoch'], df['val/loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
            
            ax.set_title('YOLO Training Loss Curve', fontsize=16, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel('Loss', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=12)
            
            out_file = output_dir / f"{run_dir.name}_yolo_loss_curve.png"
            fig.tight_layout()
            fig.savefig(out_file)
            plt.close(fig)
            return out_file
        else:
            raise FileNotFoundError(f"No metrics.json, metrics.csv, or results.csv found in {run_dir}")
            
    if not history:
        raise ValueError("Found metrics file but history was empty.")
        
    df = pd.DataFrame(history)
    
    # Plotting setup
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    epochs = df.get('epoch', df.index + 1)
    
    if 'train_loss' in df.columns:
        ax.plot(epochs, df['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2.5, marker='o')
    if 'val_loss' in df.columns:
        ax.plot(epochs, df['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2.5, marker='s')
        
    ax.set_title('Self-Supervised Pretraining Loss Curve', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Reconstruction Loss', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    out_file = output_dir / f"{run_dir.name}_loss_curve.png"
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    return out_file
