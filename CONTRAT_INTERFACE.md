# Contrat d'interface A → B

Document à donner à la Personne B. Il décrit **exactement** ce que le module de détection renvoie.

## Appel

```python
from detection.predict import predict

result = predict(image, category="bottle")   # ou "cable"
```

`image` peut être :
- un chemin (`"photo.png"`),
- une image PIL (par exemple `Image.open(fichier_uploadé)` dans Streamlit),
- un tableau numpy RGB `(H, W, 3)`.

Le modèle est chargé une seule fois, au premier appel, puis gardé en mémoire. Dans Streamlit, vous pouvez quand
même envelopper l'appel dans `@st.cache_resource` si vous voulez.

## Ce que renvoie `predict()`

| Clé | Type | Description |
|---|---|---|
| `category` | `str` | `"bottle"` ou `"cable"` |
| `status` | `str` | `"normal"` ou `"anomalous"` |
| `anomaly_score` | `float` | score normalisé dans **[0, 1]** |
| `threshold` | `float` | toujours **0.5** : `status == "anomalous"` si et seulement si `anomaly_score >= 0.5` |
| `anomaly_map` | `np.ndarray` float32 `(H, W)` | carte d'anomalie dans [0, 1], **même taille que l'image d'entrée** |
| `pixel_threshold` | `float` | **0.5** : un pixel est suspect si `anomaly_map >= 0.5` |
| `raw_score` | `float` | score brut du modèle (pour info / débogage) |
| `raw_threshold` | `float` | seuil brut (pour info) |
| `image_size` | `[H, W]` | taille de l'image d'entrée |
| `inference_time_ms` | `float` | temps d'inférence |

> `to_json(result)` renvoie le même dictionnaire **sans** `anomaly_map` (sérialisable en JSON).

### Interprétation du score

`anomaly_score` est normalisé de sorte que **0.5 = seuil de décision**. En pratique :
- environ 0 à 0.4 : clairement normal,
- environ 0.4 à 0.6 : proche du seuil, **cas incertain** (bon déclencheur pour une recommandation de « vérification manuelle »),
- environ 0.6 à 1 : clairement anormal.

Ces plages sont indicatives. Pour le LLM, il suffit de dire que la proximité avec 0.5 traduit l'incertitude.

## Ce que fait B à partir de là (post-traitement)

```python
import cv2, numpy as np

mask = (result["anomaly_map"] >= result["pixel_threshold"]).astype(np.uint8)
n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
# stats[i] = x, y, w, h, aire  -> bbox, area_pct, position ("top-left"...), max_intensity
```

Puis B construit le JSON envoyé au LLM (`regions`, etc.).

## Visualisation prête à l'emploi

```python
from detection.visualize import overlay_heatmap, draw_contours, colorize

vis = overlay_heatmap(image_rgb, result["anomaly_map"])                 # heatmap superposée
vis = draw_contours(vis, result["anomaly_map"] >= 0.5, (255, 255, 255))  # contour des zones suspectes
heat = colorize(result["anomaly_map"])                                  # heatmap seule
```

## Faux résultat pour développer avant que le modèle soit prêt

```python
import numpy as np

def fake_predict(image, category="bottle"):
    h, w = 900, 900
    amap = np.zeros((h, w), np.float32)
    amap[300:380, 500:600] = 0.9      # une zone suspecte
    return {"category": category, "status": "anomalous", "anomaly_score": 0.87,
            "threshold": 0.5, "anomaly_map": amap, "pixel_threshold": 0.5,
            "raw_score": 3.1, "raw_threshold": 2.2, "image_size": [h, w], "inference_time_ms": 120.0}
```

## Fichiers nécessaires

```
detection/                  (dossier de code)
config.yaml
models/patchcore_bottle.ckpt
models/patchcore_cable.ckpt
```

Gardez cette structure : `detection/`, `config.yaml` et `models/` doivent être dans le **même dossier**, car les
chemins sont résolus à partir de là. Les dépendances sont dans `requirements.txt` (`anomalib==2.7.0`).

## Limite à connaître pour la démo

Le modèle a appris l'apparence **des images MVTec** : même éclairage, même cadrage, même fond. Une photo prise
au téléphone ou trouvée sur internet sera très probablement classée `anomalous`, même si l'objet est intact,
parce qu'elle ne ressemble pas aux images d'entraînement. Pour la démo, utilisez les images de test MVTec.
C'est aussi un bon point à mentionner dans l'interface (« modèle valable pour les conditions de prise de vue
de la ligne de production »).
