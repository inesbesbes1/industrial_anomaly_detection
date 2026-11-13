"""Crée des variantes « prise de vue différente » d'images NORMALES du test MVTec.

Chaque variante simule une seule dérive réaliste (autre caméra / autre éclairage) sur une pièce
intacte : le bon résultat attendu est donc toujours « normal ». On mesure ainsi la robustesse du
modèle au changement de domaine, à intensité croissante.

Usage :
    python external/make_shifted.py                 # bottle, cable, transistor, 4 images chacune
Produit : external/shifted/<categorie>/<transformation>__<image>.png
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_IMAGES = 4


def brightness(im, f):
    return np.clip(im.astype(np.float32) * f, 0, 255).astype(np.uint8)


def warm(im, s):  # lumière plus chaude (plus de rouge, moins de bleu) - image BGR
    out = im.astype(np.float32)
    out[..., 2] *= 1 + s
    out[..., 0] *= 1 - s
    return np.clip(out, 0, 255).astype(np.uint8)


def rotate(im, deg):
    h, w = im.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(im, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def shift(im, frac):
    h, w = im.shape[:2]
    m = np.float32([[1, 0, frac * w], [0, 1, frac * h]])
    return cv2.warpAffine(im, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def zoom(im, f):
    h, w = im.shape[:2]
    ch, cw = int(h / f), int(w / f)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (w, h))


def blur(im, k):
    return cv2.GaussianBlur(im, (k, k), 0)


def jpeg(im, q):
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def noise(im, sigma):
    rng = np.random.default_rng(0)
    return np.clip(im + rng.normal(0, sigma, im.shape), 0, 255).astype(np.uint8)


TRANSFORMS = {
    "lum_+15": lambda im: brightness(im, 1.15),
    "lum_+40": lambda im: brightness(im, 1.40),
    "lum_-30": lambda im: brightness(im, 0.70),
    "chaud_10": lambda im: warm(im, 0.10),
    "chaud_25": lambda im: warm(im, 0.25),
    "rot_5": lambda im: rotate(im, 5),
    "rot_20": lambda im: rotate(im, 20),
    "decal_5pct": lambda im: shift(im, 0.05),
    "decal_15pct": lambda im: shift(im, 0.15),
    "zoom_1.2": lambda im: zoom(im, 1.2),
    "flou_9": lambda im: blur(im, 9),
    "flou_25": lambda im: blur(im, 25),
    "jpeg_q30": lambda im: jpeg(im, 30),
    "bruit_10": lambda im: noise(im, 10),
}


def main():
    for cat in ["bottle", "cable", "transistor"]:
        src = sorted((ROOT / "data" / "MVTecAD" / cat / "test" / "good").glob("*.png"))[:N_IMAGES]
        out = ROOT / "external" / "shifted" / cat
        out.mkdir(parents=True, exist_ok=True)
        for p in src:
            im = cv2.imread(str(p))
            for name, fn in TRANSFORMS.items():
                cv2.imwrite(str(out / f"{name}__{p.stem}.png"), fn(im))
        print(f"[{cat}] {len(src) * len(TRANSFORMS)} variantes -> {out}")


if __name__ == "__main__":
    main()
