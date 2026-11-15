"""Images externes « même angle que MVTec » : bonnes + cassées, pour les 3 catégories.

Sources : photos Wikimedia Commons déjà dans external/<cat>/ (pas de MVTec), recadrées
pour reproduire le cadrage du dataset (objet centré, vue de dessus, même taille).
Les versions « broken » sont obtenues en ajoutant un défaut réaliste sur la photo alignée
(défaut SYNTHÉTIQUE : il n'existe pas de photo web du même objet réellement cassé).

transistor : le BC547 externe est détouré puis posé sur une plaque vide du dataset
(data/MVTecAD/transistor/test/misplaced/002.png) pour retrouver le fond de la ligne.

Usage : python external/make_aligned.py   ->  external/aligned/<cat>/{good,broken}_*.png
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "external"
OUT = EXT / "aligned"
RNG = np.random.default_rng(0)


def read(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def save(img, cat, name):
    d = OUT / cat
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def crop_circle(img, cx, cy, r, size, angle=0.0, margin=1.04):
    """Recadre un carré centré sur l'objet rond (rotation optionnelle) et redimensionne."""
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    h = int(r * margin)
    x0, y0 = int(cx - h), int(cy - h)
    pad = max(0, -x0, -y0, x0 + 2 * h - img.shape[1], y0 + 2 * h - img.shape[0])
    rot = cv2.copyMakeBorder(rot, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    sq = rot[y0 + pad:y0 + pad + 2 * h, x0 + pad:x0 + pad + 2 * h]
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)


def jagged_polygon(cx, cy, r, n=14, jitter=0.35):
    a = np.sort(RNG.uniform(0, 2 * np.pi, n))
    rr = r * (1 + RNG.uniform(-jitter, jitter, n))
    return np.stack([cx + rr * np.cos(a), cy + rr * np.sin(a)], 1).astype(np.int32)


# ------------------------------------------------------------------ bottle
def bottle():
    srcs = {"glass_mouth_196855": (157, 177, 92), "glass_mouth_196850": (241, 168, 80)}
    for name, (cx, cy, r) in srcs.items():
        img = read(EXT / "bottle" / f"{name}.jpg")
        good = crop_circle(img, cx, cy, r, 900, margin=1.08)
        save(good, "bottle", f"good_{name}")
        # broken_large : un éclat du goulot manque -> on voit le fond clair à travers
        br = good.copy()
        # éclat sur le bord extérieur du goulot (en haut à droite) : le verre manque, fond clair visible
        poly = jagged_polygon(735, 165, 120)
        cv2.fillPoly(br, [poly], (226, 228, 224))
        cv2.polylines(br, [poly], True, (60, 80, 65), 5)
        save(br, "bottle", f"broken_{name}")
        # contamination : dépôt sombre à l'intérieur du goulot
        ct = good.copy()
        spot = np.zeros(ct.shape[:2], np.float32)
        cv2.fillPoly(spot, [jagged_polygon(560, 600, 45, jitter=0.5)], 1.0)
        spot = cv2.GaussianBlur(spot, (21, 21), 0)[..., None]
        ct = (ct * (1 - spot) + np.array([70, 45, 20]) * spot).astype(np.uint8)
        save(ct, "bottle", f"broken_contamination_{name}")


# ------------------------------------------------------------------ cable
def cable():
    img = read(EXT / "cable" / "power_cable_us_3core.jpg")
    # coupe du câble centrée (509,355) r~270 ; rotation -> conducteur vert en haut comme MVTec
    good = crop_circle(img, 509, 355, 272, 1024, angle=-30, margin=1.02)
    save(good, "cable", "good_power_cable_us_3core")

    hsv = cv2.cvtColor(good, cv2.COLOR_RGB2HSV)
    copper = cv2.inRange(hsv, (0, 40, 120), (25, 200, 255))
    disk = np.zeros_like(copper)
    cv2.circle(disk, (512, 512), 380, 255, -1)
    copper &= disk
    copper = cv2.morphologyEx(copper, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(copper)
    blobs = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:3]

    # missing_wire : les brins de cuivre d'un conducteur ont disparu (on voit l'isolant)
    mw = good.copy()
    i = blobs[0]
    m = cv2.dilate((lab == i).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    ring = cv2.dilate(m.astype(np.uint8), np.ones((40, 40), np.uint8)).astype(bool) & ~m
    mw[m] = np.median(good[ring], 0).astype(np.uint8)
    save(mw, "cable", "broken_missing_wire")

    # cut_outer_insulation : entaille dans la gaine extérieure
    cu = good.copy()
    poly = np.array([[150, 420], [235, 380], [250, 520], [165, 560]], np.int32)
    cv2.fillPoly(cu, [poly], (35, 30, 30))
    cv2.polylines(cu, [poly], True, (190, 120, 90), 3)
    save(cu, "cable", "broken_cut_insulation")


# ------------------------------------------------------------------ transistor
def transistor():
    src = read(EXT / "transistor" / "bc547_photo.jpg")
    board = read(ROOT / "data/MVTecAD/transistor/test/misplaced/002.png")
    # détourage : tout ce qui n'est pas le fond blanc
    crop = src[240:575, 330:540]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    mask = (gray < 185).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    s = 2.3
    obj = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    msk = cv2.resize(mask, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)

    def paste(o, m, x, y, angle=0.0):
        out = board.copy()
        if angle:
            c = (o.shape[1] / 2, o.shape[0] / 2)
            M = cv2.getRotationMatrix2D(c, angle, 1.0)
            o = cv2.warpAffine(o, M, (o.shape[1], o.shape[0]))
            m = cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST)
        h, w = m.shape
        reg = out[y:y + h, x:x + w]
        mb = cv2.GaussianBlur(m.astype(np.float32), (5, 5), 0)[..., None]
        out[y:y + h, x:x + w] = (o * mb + reg * (1 - mb)).astype(np.uint8)
        return out

    x0 = 512 - obj.shape[1] // 2
    save(paste(obj, msk, x0, 150), "transistor", "good_bc547")
    # misplaced : transistor tourné et décalé
    save(paste(obj, msk, x0 + 120, 170, angle=35), "transistor", "broken_misplaced_bc547")
    # cut_lead : la patte de droite est coupée à mi-hauteur
    m2 = msk.copy()
    hh, ww = m2.shape
    m2[int(hh * 0.58):, int(ww * 0.62):] = 0
    save(paste(obj, m2, x0, 150), "transistor", "broken_cut_lead_bc547")


if __name__ == "__main__":
    bottle(); cable(); transistor()
    print("OK ->", OUT)
