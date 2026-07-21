"""
Detailed epoch-by-epoch comparison between ImageNet baseline and SparK-pretrained YOLO.
Shows convergence speed, peak performance, and final metrics.
"""
import csv, pathlib

def load(path):
    p = pathlib.Path(path)
    with open(p, newline='') as f:
        reader = csv.DictReader(f)
        return [{k.strip(): v.strip() for k, v in row.items()} for row in reader]

img = load(r'runs\downstream\imagenet_baseline\results.csv')
spk = load(r'runs\downstream\spark_pretrained\results.csv')

KEY50    = 'metrics/mAP50(B)'
KEY5095  = 'metrics/mAP50-95(B)'
KEYPREC  = 'metrics/precision(B)'
KEYREC   = 'metrics/recall(B)'
KEYBOX   = 'val/box_loss'
KEYCLS   = 'val/cls_loss'

print("=" * 90)
print("  YOLO11l-OBB  |  ImageNet Baseline  vs  SparK-Pretrained Backbone")
print("  Dataset: Poultry Houses (4,824 train / 702 test)  |  100 epochs each")
print("=" * 90)
print()

# Milestone epochs to show
milestones = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100]

print(f"{'Epoch':>5}  {'IMG mAP50':>10}  {'SPK mAP50':>10}  {'Delta':>8}  {'IMG mAP50-95':>13}  {'SPK mAP50-95':>13}  {'Delta':>8}")
print("-" * 90)

for ep in milestones:
    ir = next((r for r in img if r.get('epoch','').strip() == str(ep)), None)
    sr = next((r for r in spk if r.get('epoch','').strip() == str(ep)), None)
    if ir and sr:
        im50   = float(ir[KEY50])
        sm50   = float(sr[KEY50])
        im5095 = float(ir[KEY5095])
        sm5095 = float(sr[KEY5095])
        d50    = sm50 - im50
        d5095  = sm5095 - im5095
        marker = " <-- SparK leads" if d50 > 0 else (" <-- ImageNet leads" if d50 < 0 else "")
        print(f"{ep:>5}  {im50:>10.4f}  {sm50:>10.4f}  {d50:>+8.4f}  {im5095:>13.4f}  {sm5095:>13.4f}  {d5095:>+8.4f}{marker}")

print()
print("=" * 90)
print("  FINAL RESULTS (Epoch 100)")
print("=" * 90)
print()

ir100 = img[-1]
sr100 = spk[-1]

metrics = [
    ("mAP50",       KEY50),
    ("mAP50-95",    KEY5095),
    ("Precision",   KEYPREC),
    ("Recall",      KEYREC),
    ("Val Box Loss",KEYBOX),
    ("Val Cls Loss",KEYCLS),
]

print(f"  {'Metric':<20}  {'ImageNet':>10}  {'SparK':>10}  {'Delta':>10}  {'Winner':}")
print("  " + "-" * 70)
for label, key in metrics:
    iv = float(ir100.get(key, 0))
    sv = float(sr100.get(key, 0))
    delta = sv - iv
    # For loss, lower is better
    if 'loss' in key.lower():
        winner = "SparK" if delta < 0 else ("ImageNet" if delta > 0 else "Tie")
    else:
        winner = "SparK" if delta > 0 else ("ImageNet" if delta < 0 else "Tie")
    print(f"  {label:<20}  {iv:>10.4f}  {sv:>10.4f}  {delta:>+10.4f}  {winner}")

print()

# Find best epochs
best_img_ep  = max(img, key=lambda r: float(r[KEY50]))
best_spk_ep  = max(spk, key=lambda r: float(r[KEY50]))
best_img_9595 = max(img, key=lambda r: float(r[KEY5095]))
best_spk_9595 = max(spk, key=lambda r: float(r[KEY5095]))

print("=" * 90)
print("  PEAK PERFORMANCE")
print("=" * 90)
print(f"  Best mAP50    ImageNet: {float(best_img_ep[KEY50]):.4f}  @ epoch {best_img_ep['epoch']}")
print(f"  Best mAP50    SparK   : {float(best_spk_ep[KEY50]):.4f}  @ epoch {best_spk_ep['epoch']}")
print(f"  Best mAP50-95 ImageNet: {float(best_img_9595[KEY5095]):.4f}  @ epoch {best_img_9595['epoch']}")
print(f"  Best mAP50-95 SparK   : {float(best_spk_9595[KEY5095]):.4f}  @ epoch {best_spk_9595['epoch']}")

# Convergence: epoch where each first hits 0.95 mAP50
thresh = 0.95
for name, rows in [("ImageNet", img), ("SparK", spk)]:
    hit = next((r for r in rows if float(r[KEY50]) >= thresh), None)
    if hit:
        print(f"  First epoch reaching mAP50 >= {thresh}: {name} @ epoch {hit['epoch']}")
    else:
        print(f"  Never reached mAP50 >= {thresh}: {name}")
