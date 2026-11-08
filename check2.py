import csv
import random

import torch
import torch.nn.functional as F

from detection.predict import get_model, _to_rgb, predict

CAT = "bottle"
model, cfg, dev = get_model(CAT)
print("PRE-PROCESSOR:", model.pre_processor)
print("POST-PROCESSOR:", model.post_processor)

rows = list(csv.DictReader(open(f"results/{CAT}/predictions.csv", encoding="utf-8")))
random.seed(0)
rows = random.sample(rows, 20)[:8]

print(f"\n{'type':<14}{'evaluate':>10}{'predict':>10}{'resize256':>11}")
for r in rows:
    rgb = _to_rgb(r["image_path"])
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
    x256 = F.interpolate(x, size=(256, 256), mode="bilinear", antialias=True, align_corners=False)
    with torch.no_grad():
        s_resized = float(model(x256).pred_score.reshape(-1)[0])
    s_pred = predict(r["image_path"], CAT)["raw_score"]
    print(f"{r['defect_type']:<14}{float(r['score_raw']):>10.3f}{s_pred:>10.3f}{s_resized:>11.3f}")