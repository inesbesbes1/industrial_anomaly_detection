"""Tests unitaires du post-traitement OpenCV (Personne B)."""

import numpy as np
import pytest

from processing.postprocessing import (
    determine_position,
    extract_regions,
    build_llm_payload
)


def test_determine_position():
    # Image 300x300 : le centre est entre 100 et 200
    assert determine_position(10, 10, 20, 20, 300, 300) == "haut-gauche"
    assert determine_position(140, 140, 20, 20, 300, 300) == "centre"
    assert determine_position(250, 250, 20, 20, 300, 300) == "bas-droite"


def test_extract_regions_empty():
    # Carte d'anomalie entièrement saine (zéros partout)
    amap = np.zeros((100, 100), dtype=np.float32)
    regions = extract_regions(amap, threshold=0.5)
    assert len(regions) == 0


def test_extract_regions_single_defect():
    # Une tache anormale de 40x40 pixels dans le coin supérieur droit
    amap = np.zeros((300, 300), dtype=np.float32)
    amap[20:60, 220:260] = 0.92

    regions = extract_regions(amap, threshold=0.5, min_area_pixels=20)
    assert len(regions) == 1
    r = regions[0]
    assert r["position"] == "haut-droite"
    assert r["max_intensity"] == 0.92
    assert r["area_pct"] > 0.0


def test_build_llm_payload():
    amap = np.zeros((200, 200), dtype=np.float32)
    amap[80:120, 80:120] = 0.85

    mock_predict = {
        "category": "bottle",
        "status": "anomalous",
        "anomaly_score": 0.81,
        "threshold": 0.5,
        "anomaly_map": amap,
        "pixel_threshold": 0.5
    }

    payload = build_llm_payload(mock_predict)
    assert payload["category"] == "bottle"
    assert payload["status"] == "anomalous"
    assert payload["anomaly_score"] == 0.81
    assert payload["regions_count"] == 1
    assert payload["regions"][0]["position"] == "centre"
    print("Tous les tests de processing sont passés avec succès !")


if __name__ == "__main__":
    test_determine_position()
    test_extract_regions_empty()
    test_extract_regions_single_defect()
    test_build_llm_payload()