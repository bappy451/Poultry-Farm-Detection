import pandas as pd
import os

base_path = r'runs/segment/trained_models/finetune_results'
models = ['yolo11l_pretrained', 'yolo11l_spark', 'yolo26l_pretrained', 'yolo26l_spark']

results = []

for model in models:
    csv_path = os.path.join(base_path, model, 'results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Strip whitespace from column names
        df.columns = [c.strip() for c in df.columns]
        
        # Get the row with the best mAP50(B)
        best_row_b = df.loc[df['metrics/mAP50(B)'].idxmax()]
        
        # Get the row with the best mAP50(M)
        best_row_m = df.loc[df['metrics/mAP50(M)'].idxmax()]
        
        pb = best_row_b['metrics/precision(B)']
        rb = best_row_b['metrics/recall(B)']
        map50b = best_row_b['metrics/mAP50(B)']
        f1b = 2 * (pb * rb) / (pb + rb) if (pb + rb) > 0 else 0
        
        pm = best_row_m['metrics/precision(M)']
        rm = best_row_m['metrics/recall(M)']
        map50m = best_row_m['metrics/mAP50(M)']
        f1m = 2 * (pm * rm) / (pm + rm) if (pm + rm) > 0 else 0
        
        results.append({
            'Model': model,
            'P(B)': pb,
            'R(B)': rb,
            'F1(B)': f1b,
            'mAP50(B)': map50b,
            'P(M)': pm,
            'R(M)': rm,
            'F1(M)': f1m,
            'mAP50(M)': map50m
        })

summary_df = pd.DataFrame(results)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(summary_df)
