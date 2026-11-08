"""Compare les scores bruts de predict() (image par image) avec ceux de evaluate (predictions.csv).

Usage (depuis la racine du projet) :
    python check.py
    python check.py --category bottle --n 20
"""

import argparse
import csv
import random
from pathlib import Path

from detection.predict import predict

ap = argparse.ArgumentParser()
ap.add_argument("--category", default="cable")
ap.add_argument("--n", type=int, default=30, help="nombre d'images à comparer")
args = ap.parse_args()

csv_path = Path("results") / args.category / "predictions.csv"
rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))

# Échantillon aléatoire (graine fixe) pour couvrir plusieurs types de défauts, pas seulement les premiers
random.seed(0)
rows = random.sample(rows, min(args.n, len(rows)))

diffs = []
print(f"{'type':<22}{'evaluate':>10}{'predict':>10}{'écart':>9}")
for r in rows:
    res = predict(r["image_path"], args.category)
    d = res["raw_score"] - float(r["score_raw"])
    diffs.append(d)
    print(f"{r['defect_type']:<22}{float(r['score_raw']):>10.3f}{res['raw_score']:>10.3f}{d:>+9.3f}")

print(f"\nÉcart moyen : {sum(diffs) / len(diffs):+.3f}")
print(f"Écart absolu max : {max(abs(d) for d in diffs):.3f}")