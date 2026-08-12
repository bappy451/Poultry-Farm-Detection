import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('fig', exist_ok=True)

# 1. Pre-training metrics plot
try:
    df_metrics = pd.read_csv('../../runs/dino_pretrain_failed/metrics.csv')
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('DINO Loss', color=color)
    ax1.plot(df_metrics['epoch'], df_metrics['train_dino_loss'], color=color, label='DINO Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Prototype Entropy', color=color)
    ax2.plot(df_metrics['epoch'], df_metrics['prototype_entropy'], color=color, linestyle='--', label='Entropy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('DINO Pre-Training Metrics')
    plt.savefig('fig/pretrain_metrics.pdf')
    plt.close()
    print("Saved fig/pretrain_metrics.pdf")
except Exception as e:
    print(f"Failed to generate pretrain plot: {e}")

# 2. Downstream performance bar chart
try:
    # From runs/downstream/comparison_results.csv
    data = {
        'Method': ['ImageNet Baseline', 'DINO Pre-Trained (Ours)'],
        'mAP50': [0.98058, 0.97740],
        'mAP50-95': [0.80181, 0.80083]
    }
    df_downstream = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = [0, 1]
    width = 0.35

    ax.bar([p - width/2 for p in x], df_downstream['mAP50'], width, label='mAP50', color='skyblue')
    ax.bar([p + width/2 for p in x], df_downstream['mAP50-95'], width, label='mAP50-95', color='salmon')

    ax.set_ylabel('Score')
    ax.set_title('Downstream Segmentation Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(df_downstream['Method'])
    ax.legend()
    ax.set_ylim(0, 1.1)

    fig.tight_layout()
    plt.savefig('fig/downstream_comparison.pdf')
    plt.close()
    print("Saved fig/downstream_comparison.pdf")
except Exception as e:
    print(f"Failed to generate downstream plot: {e}")

# 3. Few-shot performance bar chart
try:
    # From runs/downstream_fewshot/comparison_results.csv
    # We saw imagenet_baseline mAP50=0.97781, spark_scratch=0.70638
    data_fs = {
        'Method': ['ImageNet Baseline', 'DINO Scratch (SPARK)'],
        'mAP50': [0.97781, 0.70638],
        'mAP50-95': [0.75993, 0.55951]
    }
    df_fs = pd.DataFrame(data_fs)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([p - width/2 for p in x], df_fs['mAP50'], width, label='mAP50', color='skyblue')
    ax.bar([p + width/2 for p in x], df_fs['mAP50-95'], width, label='mAP50-95', color='salmon')

    ax.set_ylabel('Score')
    ax.set_title('Few-Shot Segmentation Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(df_fs['Method'])
    ax.legend()
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    plt.savefig('fig/fewshot_comparison.pdf')
    plt.close()
    print("Saved fig/fewshot_comparison.pdf")
except Exception as e:
    print(f"Failed to generate few-shot plot: {e}")
