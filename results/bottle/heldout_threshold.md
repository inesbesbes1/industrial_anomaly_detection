# Seuil hors échantillon - bottle

200 tirages aléatoires 50/50 (stratifiés par type de défaut, graine 42). Test complet : 20 normales, 63 défectueuses.

| Protocole | F1 image | Fausses alarmes | Défauts manqués |
|---|---|---|---|
| Seuil Anomalib (37.87), choisi sur tout le test | 100.0 % | 0.0 % (0/20) | 0.0 % (0/63) |
| Seuil choisi sur une moitié, mesuré sur l'autre (moyenne ± écart-type) | 100.0 ± 0.0 % | 0.0 ± 0.0 % (≈ 0.0/20) | 0.0 ± 0.0 % (≈ 0.0/63) |

Seuil moyen retrouvé sur les moitiés : 33.93 ± 1.11 (seuil Anomalib : 37.87).

## Taux de détection par type de défaut (moitié non utilisée pour le seuil)

| Type | Taux de détection (moyenne ± écart-type) |
|---|---|
| broken_large | 100.0 ± 0.0 % |
| broken_small | 100.0 ± 0.0 % |
| contamination | 100.0 ± 0.0 % |

Lecture : l'AUROC ne dépend pas du seuil et reste inchangé. Un écart important entre les deux lignes du tableau mesure l'optimisme dû au choix du seuil sur les images de test.
Les écarts-types reflètent la petite taille des moitiés : ils sont indicatifs.
