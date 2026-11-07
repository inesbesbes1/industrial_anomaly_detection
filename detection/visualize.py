"""Outils de visualisation : heatmaps, superpositions, contours.

Utilisables aussi par la Personne B dans l'interface Streamlit :

    from detection.visualize import overlay_heatmap
    vis = overlay_heatmap(image_rgb, result["anomaly_map"])
"""

from __future__ import annotations

import cv2
import numpy as np


def load_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_like_model(img: np.ndarray, image_size=(256, 256), center_crop=None) -> np.ndarray:
    """Applique à une image d'affichage le même redimensionnement / recadrage que le modèle."""
    h, w = image_size
    out = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if center_crop:
        ch, cw = center_crop
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        out = out[y0:y0 + ch, x0:x0 + cw]
    return out


def colorize(anomaly_map: np.ndarray) -> np.ndarray:
    """Carte dans [0, 1] -> image RGB (palette JET : bleu = normal, rouge = anormal)."""
    a = np.clip(np.asarray(anomaly_map, dtype=np.float32), 0, 1)
    color = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def overlay_heatmap(image_rgb: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpose la heatmap sur l'image (la carte est redimensionnée à la taille de l'image)."""
    h, w = image_rgb.shape[:2]
    amap = np.asarray(anomaly_map, dtype=np.float32)
    if amap.shape != (h, w):
        amap = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(image_rgb, 1 - alpha, colorize(amap), alpha, 0)


def draw_contours(image_rgb: np.ndarray, mask: np.ndarray, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Dessine le contour d'un masque binaire sur l'image."""
    h, w = image_rgb.shape[:2]
    m = np.asarray(mask).astype(np.uint8)
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = image_rgb.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out
