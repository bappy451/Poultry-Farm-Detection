import csv, pathlib

runs = {
    'YOLO-ImageNet (baseline)': r'runs\downstream\imagenet_baseline\results.csv',
    'YOLO-SparK (ours)':        r'runs\downstream\spark_pretrained\results.csv',
    'ViT-B ImageNet':           r'runs\downstream\vit_b_imagenet\results.csv',
    'ViT-B MIM (ours)':         r'runs\downstream\vit_b_mim\results.csv',
    'ViT-B Random':             r'runs\downstream\vit_b_random\results.csv',
    'ViT-S ImageNet':           r'runs\downstream\vit_s_imagenet\results.csv',
    'ViT-S MIM (ours)':         r'runs\downstream\vit_s_mim\results.csv',
    'ViT-S Random':             r'runs\downstream\vit_s_random\results.csv',
}

hdr = "%-28s %6s  %7s  %9s  %9s  %7s  %s"
print(hdr % ("Experiment", "Epochs", "mAP50", "mAP50-95", "Precision", "Recall", "Status"))
print('-' * 90)

for name, path in runs.items():
    p = pathlib.Path(path)
    if not p.exists():
        print("%-28s  NOT FOUND" % name)
        continue
    with open(p, newline='') as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        print("%-28s  EMPTY" % name)
        continue
    header = [h.strip() for h in rows[0]]
    last_row = rows[-1]
    last = {header[i]: last_row[i].strip() for i in range(len(header))}
    ep = last_row[0].strip()

    if 'metrics/mAP50(B)' in last:
        map50   = float(last.get('metrics/mAP50(B)', 0))
        map5095 = float(last.get('metrics/mAP50-95(B)', 0))
        prec    = float(last.get('metrics/precision(B)', 0))
        rec     = float(last.get('metrics/recall(B)', 0))
    else:
        try:
            vals = last_row
            map50   = float(vals[2]) if len(vals) > 2 else 0
            map5095 = float(vals[5]) if len(vals) > 5 else 0
            prec    = float(vals[3]) if len(vals) > 3 else 0
            rec     = float(vals[4]) if len(vals) > 4 else 0
        except Exception:
            map50 = prec = rec = map5095 = 0

    done = "Done" if ep == '100' else ("Ep %s/100" % ep)
    print("%-28s %6s  %7.4f  %9.4f  %9.4f  %7.4f  %s" % (name, ep, map50, map5095, prec, rec, done))
