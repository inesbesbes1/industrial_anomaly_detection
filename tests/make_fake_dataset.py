"""Crée un mini-dataset SYNTHÉTIQUE au format MVTec AD (uniquement pour tester le code).

Usage : python tests/make_fake_dataset.py tests/fake_mvtec
"""
import sys, numpy as np, cv2
from pathlib import Path
root = Path(sys.argv[1]); size = 300
rng = np.random.default_rng(0)
def normal():
    img = np.full((size, size, 3), 30, np.uint8)
    cv2.circle(img, (size//2, size//2), size//3, (90, 140, 70), -1)
    img = img.astype(np.int16) + rng.integers(-8, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
for cat in ["bottle", "cable"]:
    (root/cat/"train/good").mkdir(parents=True, exist_ok=True)
    for i in range(16): cv2.imwrite(str(root/cat/f"train/good/{i:03d}.png"), normal())
    (root/cat/"test/good").mkdir(parents=True, exist_ok=True)
    for i in range(6): cv2.imwrite(str(root/cat/f"test/good/{i:03d}.png"), normal())
    for d in ["broken_large", "contamination"]:
        (root/cat/f"test/{d}").mkdir(parents=True, exist_ok=True)
        (root/cat/f"ground_truth/{d}").mkdir(parents=True, exist_ok=True)
        for i in range(5):
            img = normal(); m = np.zeros((size, size), np.uint8)
            c = tuple(int(v) for v in rng.integers(90, 210, 2)); r = int(rng.integers(12, 30))
            cv2.circle(img, c, r, (240, 240, 240) if d == "contamination" else (0, 0, 0), -1)
            cv2.circle(m, c, r, 255, -1)
            cv2.imwrite(str(root/cat/f"test/{d}/{i:03d}.png"), img)
            cv2.imwrite(str(root/cat/f"ground_truth/{d}/{i:03d}_mask.png"), m)
print("ok")
