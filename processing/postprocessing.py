"""Module de post-traitement OpenCV avec sévérité et masque binaire (Personne B)."""

from __future__ import annotations
import cv2
import numpy as np
import math
from processing.severity import evaluate_severity_and_decision


def determine_position(cx: float, cy: float, img_w: int, img_h: int) -> str:
    """Détermine la zone spatiale précise du défaut via le centroïde réel (centre de masse)."""
    center_x, center_y = img_w / 2, img_h / 2
    dx = cx - center_x
    dy = cy - center_y
    radius = min(img_w, img_h) / 2
    distance = math.sqrt(dx**2 + dy**2)
    # 1. Seuil très strict : centre uniquement si vraiment au milieu (< 8% du rayon)
    if distance < 0.08 * radius:
        return "centre"
    # 2. Orientation vectorielle basée sur le centroïde réel
    vertical = "haut" if dy < -0.08 * radius else ("bas" if dy > 0.08 * radius else "")
    horizontal = "gauche" if dx < -0.08 * radius else ("droite" if dx > 0.08 * radius else "")
    if vertical and horizontal:
        return f"{vertical}-{horizontal}"
    elif vertical:
        return vertical
    elif horizontal:
        return horizontal
    return "centre"


def _dominant_quadrant_from_pixels(region_mask: np.ndarray, img_w: int, img_h: int) -> str:
    """Pour les grandes régions (arcs, croissants), trouve le quadrant avec le plus de pixels anomaux."""
    ys, xs = np.where(region_mask)
    if len(xs) == 0:
        return "centre"

    cx_half = img_w / 2
    cy_half = img_h / 2

    # Comptage par quadrant
    q = {"haut-gauche": 0, "haut-droite": 0, "bas-gauche": 0, "bas-droite": 0}
    q["haut-gauche"] = int(np.sum((xs < cx_half) & (ys < cy_half)))
    q["haut-droite"] = int(np.sum((xs >= cx_half) & (ys < cy_half)))
    q["bas-gauche"]  = int(np.sum((xs < cx_half) & (ys >= cy_half)))
    q["bas-droite"]  = int(np.sum((xs >= cx_half) & (ys >= cy_half)))

    dominant = max(q, key=q.get)
    total = len(xs)
    # Si un quadrant domine vraiment (> 40 %), on retourne sa position
    if q[dominant] / total > 0.40:
        return dominant
    # Sinon : moitié gauche ou droite ?
    left = q["haut-gauche"] + q["bas-gauche"]
    right = q["haut-droite"] + q["bas-droite"]
    top = q["haut-gauche"] + q["haut-droite"]
    bottom = q["bas-gauche"] + q["bas-droite"]
    h = "gauche" if left > right * 1.2 else ("droite" if right > left * 1.2 else "")
    v = "haut" if top > bottom * 1.2 else ("bas" if bottom > top * 1.2 else "")
    if v and h:
        return f"{v}-{h}"
    return v or h or "centre"

def get_binary_mask(anomaly_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Génère le masque binaire nettoyé par morphologie mathématique."""
    mask = (anomaly_map >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def extract_regions(
    anomaly_map: np.ndarray,
    threshold: float = 0.5,
    min_area_pixels: int = 30
) -> list[dict]:
    """Seuille l'anomaly_map et extrait les composantes connexes."""
    img_h, img_w = anomaly_map.shape[:2]
    total_pixels = img_h * img_w

    clean_mask = get_binary_mask(anomaly_map, threshold)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

    regions = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area_pixels:
            continue

        region_mask = (labels == i)
        max_intensity = float(np.max(anomaly_map[region_mask]))
        area_pct = round((area / total_pixels) * 100, 2)

        # Pour les grandes régions (arcs, croissants), on compte les pixels par quadrant
        # Pour les petites régions ponctuelles, le centroïde suffit
        if area_pct > 5.0:
            position = _dominant_quadrant_from_pixels(region_mask, img_w, img_h)
        else:
            position = determine_position(centroids[i][0], centroids[i][1], img_w, img_h)

        regions.append({
            "region_id": i,
            "bbox": [int(x), int(y), int(w), int(h)],
            "area_pct": area_pct,
            "position": position,
            "max_intensity": round(max_intensity, 2)
        })

    regions.sort(key=lambda r: r["max_intensity"] * r["area_pct"], reverse=True)
    return regions


def build_llm_payload(predict_result: dict) -> dict:
    """Construit le payload JSON enrichi avec l'évaluation de sévérité."""
    amap = predict_result.get("anomaly_map")
    pix_thr = predict_result.get("pixel_threshold", 0.5)

    regions = extract_regions(amap, threshold=pix_thr) if amap is not None else []
    assessment = evaluate_severity_and_decision(predict_result, regions)

    payload = {
        "category": predict_result.get("category", "inconnue"),
        "status": predict_result.get("status", "inconnu"),
        "anomaly_score": round(float(predict_result.get("anomaly_score", 0.0)), 3),
        "threshold": float(predict_result.get("threshold", 0.5)),
        "severity": assessment["severity"],
        "decision": assessment["decision"],
        "decision_label": assessment["decision_label"],
        "confidence": assessment["confidence"],
        "primary_actions": assessment["actions"],
        "regions_count": len(regions),
        "regions": regions
    }
    payload["_assessment"] = assessment
    return payload