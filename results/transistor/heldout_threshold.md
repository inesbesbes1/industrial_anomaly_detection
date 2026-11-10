# Seuil hors échantillon - transistor

200 tirages aléatoires 50/50 (stratifiés par type de défaut, graine 42). Test complet : 60 normales, 40 défectueuses.

| Protocole | F1 image | Fausses alarmes | Défauts manqués |
|---|---|---|---|
| Seuil Anomalib (37.68), choisi sur tout le test | 96.3 % | 3.3 % (2/60) | 2.5 % (1/40) |
| Seuil choisi sur une moitié, mesuré sur l'autre (moyenne ± écart-type) | 93.6 ± 3.9 % | 4.4 ± 4.8 % (≈ 2.6/60) | 6.2 ± 6.1 % (≈ 2.5/40) |

Seuil moyen retrouvé sur les moitiés : 38.32 ± 1.75 (seuil Anomalib : 37.68).

## Taux de détection par type de défaut (moitié non utilisée pour le seuil)

| Type | Taux de détection (moyenne ± écart-type) |
|---|---|
| bent_lead | 91.4 ± 9.9 % |
| cut_lead | 99.4 ± 3.4 % |
| damaged_case | 99.4 ± 3.4 % |
| misplaced | 85.1 ± 16.1 % |

Lecture : l'AUROC ne dépend pas du seuil et reste inchangé. Un écart important entre les deux lignes du tableau mesure l'optimisme dû au choix du seuil sur les images de test.
Les écarts-types reflètent la petite taille des moitiés : ils sont indicatifs.
