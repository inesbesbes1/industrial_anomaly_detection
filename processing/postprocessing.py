"""Module de post-traitement OpenCV (Personne B).
Transforme la carte d'anomalie en régions géométriques labellisées pour le LLM.
"""

from __future__ import annotations
import cv2
import numpy as np


def determine_position(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> str:
    """Détermine la zone spatiale du défaut (ex: 'haut-droite', 'centre', etc.)."""
    cx, cy = x + w / 2, y + h / 2
    
    vertical = "haut" if cy < img_h / 3 else ("bas" if cy > 2 * img_h / 3 else "centre")
    horizontal = "gauche" if cx < img_w / 3 else ("droite" if cx > 2 * img_w / 3 else "centre")
    
    if vertical == "centre" and horizontal == "centre":
        return "centre"
    return f"{vertical}-{horizontal}".strip("-")


def extract_regions(
    anomaly_map: np.ndarray,
    threshold: float = 0.5,
    min_area_pixels: int = 30
) -> list[dict]:
    """Seuille l'anomaly_map et extrait les composantes connexes."""
    img_h, img_w = anomaly_map.shape[:2]
    total_pixels = img_h * img_w

    # 1. Binarisation
    mask = (anomaly_map >= threshold).astype(np.uint8) * 255

    # 2. Nettoyage morphologique (supprime le bruit de quelques pixels)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 3. Composantes connexes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

    regions = []
    # stats[0] est toujours le fond (background)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area_pixels:
            continue

        region_mask = (labels == i)
        max_intensity = float(np.max(anomaly_map[region_mask]))
        area_pct = round((area / total_pixels) * 100, 2)

        regions.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "area_pct": area_pct,
            "position": determine_position(x, y, w, h, img_w, img_h),
            "max_intensity": round(max_intensity, 2)
        })

    # Tri par criticité (intensité max * surface)
    regions.sort(key=lambda r: r["max_intensity"] * r["area_pct"], reverse=True)
    return regions


def build_llm_payload(predict_result: dict) -> dict:
    """Construit le payload JSON conforme au contrat d'interface pour le LLM."""
    amap = predict_result.get("anomaly_map")
    pix_thr = predict_result.get("pixel_threshold", 0.5)

    if amap is not None:
        regions = extract_regions(amap, threshold=pix_thr)
    else:
        regions = []

    return {
        "category": predict_result.get("category", "inconnue"),
        "status": predict_result.get("status", "inconnu"),
        "anomaly_score": round(float(predict_result.get("anomaly_score", 0.0)), 3),
        "threshold": float(predict_result.get("threshold", 0.5)),
        "regions_count": len(regions),
        "regions": regions
    }