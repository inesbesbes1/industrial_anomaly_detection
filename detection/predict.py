"""Fonction d'inférence - c'est le point de contact avec la Personne B.

    from detection.predict import predict

    result = predict("photo.png", category="bottle")
    result["status"]          # "normal" ou "anomalous"
    result["anomaly_score"]   # float dans [0, 1], seuil de décision = 0.5
    result["threshold"]       # 0.5
    result["anomaly_map"]     # np.ndarray float32 (H, W) dans [0, 1], même taille que l'image d'entrée
                              # pixel suspect  <=>  anomaly_map >= result["pixel_threshold"] (0.5)

Usage en ligne de commande :
    python -m detection.predict --image chemin/image.png --category cable --out overlay.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import checkpoint_path, load_config, normalize, quiet

_CACHE: dict[str, Any] = {}


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model(category: str, config: str | Path | None = None):
    """Charge (une seule fois) le modèle PatchCore d'une catégorie."""
    key = f"{config}:{category}"
    if key not in _CACHE:
        from anomalib.models import Patchcore

        quiet()
        cfg = load_config(config)
        ckpt = checkpoint_path(cfg, category)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {ckpt}. Entraînez-le avec `python -m detection.train --category {category}`"
            )
        dev = _device()
        model = Patchcore.load_from_checkpoint(str(ckpt), map_location=dev).to(dev)
        model.eval()
        # On récupère les valeurs brutes et on normalise nous-mêmes (voir common.normalize)
        model.post_processor.enable_normalization = False
        _CACHE[key] = (model, cfg, dev)
    return _CACHE[key]


def _to_rgb(image) -> np.ndarray:
    """Accepte un chemin, une image PIL ou un tableau numpy RGB (H, W, 3)."""
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255 if arr.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
    return arr


def _map_to_original(amap: np.ndarray, cfg: dict, h: int, w: int) -> np.ndarray:
    """Remet la carte (taille modèle) à la taille de l'image d'origine."""
    size = tuple(cfg["model"].get("image_size", (256, 256)))
    crop = cfg["model"].get("center_crop")
    if crop:  # la carte ne couvre que la zone centrale recadrée
        full = np.zeros(size, dtype=np.float32)
        y0, x0 = (size[0] - crop[0]) // 2, (size[1] - crop[1]) // 2
        full[y0:y0 + crop[0], x0:x0 + crop[1]] = cv2.resize(amap, (crop[1], crop[0]))
        amap = full
    return cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)


def predict(image, category: str, config: str | Path | None = None) -> dict:
    """Analyse une image et renvoie le dictionnaire du contrat d'interface A -> B."""
    import torch

    model, cfg, dev = get_model(category, config)
    rgb = _to_rgb(image)
    h, w = rgb.shape[:2]
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(dev)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x)
    dt = (time.perf_counter() - t0) * 1000

    pp = model.post_processor
    raw_score = float(out.pred_score.reshape(-1)[0])
    raw_map = out.anomaly_map.detach().cpu().float().numpy().reshape(out.anomaly_map.shape[-2:])
    img_thr, pix_thr = float(pp.image_threshold), float(pp.pixel_threshold)
    score = float(normalize(raw_score, float(pp.image_min), float(pp.image_max), img_thr))
    amap = normalize(raw_map, float(pp.pixel_min), float(pp.pixel_max), pix_thr)
    amap = _map_to_original(amap, cfg, h, w)

    return {
        "category": category,
        "status": "anomalous" if raw_score >= img_thr else "normal",
        "anomaly_score": round(score, 4),
        "threshold": 0.5,
        "anomaly_map": amap.astype(np.float32),
        "pixel_threshold": 0.5,
        "raw_score": round(raw_score, 4),
        "raw_threshold": round(img_thr, 4),
        "image_size": [h, w],
        "inference_time_ms": round(dt, 1),
    }


def to_json(result: dict) -> dict:
    """Version sérialisable (sans la carte) - utile pour les logs ou le débogage."""
    return {k: v for k, v in result.items() if k != "anomaly_map"}


def main() -> None:
    from .visualize import draw_contours, overlay_heatmap

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="chemin de sortie de l'image superposée (optionnel)")
    args = ap.parse_args()

    res = predict(args.image, args.category, args.config)
    print(json.dumps(to_json(res), indent=2, ensure_ascii=False))
    if args.out:
        rgb = _to_rgb(args.image)
        vis = draw_contours(overlay_heatmap(rgb, res["anomaly_map"]), res["anomaly_map"] >= 0.5, (255, 255, 255))
        cv2.imwrite(args.out, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"Superposition enregistrée : {args.out}")


if __name__ == "__main__":
    main()
