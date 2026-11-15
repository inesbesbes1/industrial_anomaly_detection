"""Teste predict() sur external/aligned/ -> results/external_aligned.csv + .png"""
import csv, sys
from pathlib import Path
import cv2, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from detection.predict import predict

rows, figs = [], []
for cat in ["bottle", "cable", "transistor"]:
    for f in sorted((ROOT / "external/aligned" / cat).glob("*.png")):
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        r = predict(img, category=cat)
        truth = "good" if f.name.startswith("good") else "broken"
        verdict = "HORS_DOMAINE" if not r["in_domain"] else r["status"]
        ok = (truth == "good" and verdict == "normal") or (truth == "broken" and verdict == "anomalous")
        rows.append(dict(category=cat, file=f.name, truth=truth, verdict=verdict,
                         anomaly_score=round(r["anomaly_score"], 3), domain_score=r["domain_score"],
                         domain_threshold=r["domain_threshold"], correct=ok))
        figs.append((img, r["anomaly_map"], f"{cat}/{f.stem}\n{verdict} s={r['anomaly_score']:.2f} d={r['domain_score']:.2f}"))
        print(rows[-1], flush=True)

out = ROOT / "results/external_aligned.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
n = len(figs); fig, ax = plt.subplots(2, n, figsize=(2.2 * n, 5))
for i, (img, amap, t) in enumerate(figs):
    ax[0, i].imshow(img); ax[0, i].set_title(t, fontsize=6)
    ax[1, i].imshow(amap, cmap="jet", vmin=0, vmax=1)
    for a in ax[:, i]: a.axis("off")
plt.tight_layout(); plt.savefig(ROOT / "results/external_aligned.png", dpi=110)
print("correct:", sum(r["correct"] for r in rows), "/", len(rows))
