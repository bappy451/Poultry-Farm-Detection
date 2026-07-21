import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set professional scientific plot style
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5, rc={"font.family": "serif"})

OUT_DIR = Path("C:/Users/mza0288/Documents/Poultry-Farm-Detection/compag/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors
COLOR_IMAGENET = "#1f77b4" # Blue
COLOR_SPARK = "#ff7f0e"    # Orange

def plot_convergence():
    # Paths to the CSV files
    baseline_csv = Path("C:/Users/mza0288/Documents/Poultry-Farm-Detection/runs/downstream_scratch/imagenet_baseline/results.csv")
    spark_csv = Path("C:/Users/mza0288/Documents/Poultry-Farm-Detection/runs/downstream_scratch/spark_scratch/results.csv")
    
    if not baseline_csv.exists() or not spark_csv.exists():
        print("Convergence CSVs not found. Skipping convergence plot.")
        return

    # Load data
    df_base = pd.read_csv(baseline_csv)
    df_spark = pd.read_csv(spark_csv)
    
    # Strip whitespace from column names just in case
    df_base.columns = df_base.columns.str.strip()
    df_spark.columns = df_spark.columns.str.strip()

    epochs_base = df_base['epoch']
    map50_base = df_base['metrics/mAP50(B)']
    
    epochs_spark = df_spark['epoch']
    map50_spark = df_spark['metrics/mAP50(B)']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_base, map50_base, label="ImageNet Baseline", color=COLOR_IMAGENET, linewidth=3)
    plt.plot(epochs_spark, map50_spark, label="SparK (Ours)", color=COLOR_SPARK, linewidth=3, linestyle='--')
    
    plt.xlabel("Epoch")
    plt.ylabel("Validation mAP@50")
    plt.title("Training Convergence on Poultry Farm Dataset")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_convergence.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {OUT_DIR / 'fig_convergence.pdf'}")

def plot_cross_dataset():
    # Hardcoded values from the tables
    datasets = ["Poultry Farms", "GRSD (Greenhouse)", "DOTA Benchmark"]
    imagenet_scores = [0.9840, 0.9743, 0.9369]
    spark_scores = [0.9790, 0.9676, 0.9086]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, imagenet_scores, width, label='ImageNet Baseline', color=COLOR_IMAGENET, edgecolor='black')
    rects2 = ax.bar(x + width/2, spark_scores, width, label='SparK (Ours)', color=COLOR_SPARK, edgecolor='black')
    
    ax.set_ylabel('mAP@50')
    ax.set_title('Cross-Dataset Transferability (100% Training Data)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0.85, 1.0)
    ax.legend(loc="lower left")
    
    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12)
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_cross_dataset.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {OUT_DIR / 'fig_cross_dataset.pdf'}")

def plot_data_efficiency():
    # Hardcoded values from the tables
    conditions = ["100% Data (Abundant)", "10% Data (Few-Shot)"]
    imagenet_scores = [0.9840, 0.9778]
    spark_scores = [0.9790, 0.7064]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, imagenet_scores, width, label='ImageNet Baseline', color=COLOR_IMAGENET, edgecolor='black')
    rects2 = ax.bar(x + width/2, spark_scores, width, label='SparK (Ours)', color=COLOR_SPARK, edgecolor='black')
    
    ax.set_ylabel('mAP@50')
    ax.set_title('Data Efficiency Ablation (Poultry Dataset)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0.60, 1.02)
    ax.legend(loc="upper right")
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12)
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_data_efficiency.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {OUT_DIR / 'fig_data_efficiency.pdf'}")

if __name__ == "__main__":
    print("Generating professional figures for COMPAG manuscript...")
    plot_convergence()
    plot_cross_dataset()
    plot_data_efficiency()
    print("All figures generated successfully.")
