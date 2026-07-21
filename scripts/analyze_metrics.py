import pandas as pd

df = pd.read_csv(r'trained_models\spark-search_new\5d1be34f2942_yolo11l_fp16\training_metrics.csv')
print(f'Total Epochs: {len(df)}')
print(f'Best Loss : {df.loss.min():.6f} @ epoch {int(df.loc[df.loss.idxmin(),"epoch"])}')
print(f'Final Loss: {df.loss.iloc[-1]:.6f} @ epoch {int(df.epoch.iloc[-1])}')
drop = df.loss.iloc[0] - df.loss.iloc[-1]
print(f'Total drop: {drop:.6f}  ({drop/df.loss.iloc[0]*100:.2f}%)')

print('\n--- Epoch-by-epoch ---')
print(f'{"Epoch":>6}  {"Loss":>8}  {"LR":>10}  {"Dur(s)":>8}')
for _, row in df.iterrows():
    print(f'{int(row.epoch):>6}  {row.loss:>8.6f}  {row.lr:>10.6f}  {row.duration:>8.1f}')
