# Seuil hors échantillon - cable

200 tirages aléatoires 50/50 (stratifiés par type de défaut, graine 42). Test complet : 58 normales, 92 défectueuses.

| Protocole | F1 image | Fausses alarmes | Défauts manqués |
|---|---|---|---|
| Seuil Anomalib (42.67), choisi sur tout le test | 96.8 % | 6.9 % (4/58) | 2.2 % (2/92) |
| Seuil choisi sur une moitié, mesuré sur l'autre (moyenne ± écart-type) | 95.5 ± 1.7 % | 9.5 ± 6.0 % (≈ 5.5/58) | 3.1 ± 2.4 % (≈ 2.8/92) |

Seuil moyen retrouvé sur les moitiés : 42.09 ± 0.68 (seuil Anomalib : 42.67).

## Taux de détection par type de défaut (moitié non utilisée pour le seuil)

| Type | Taux de détection (moyenne ± écart-type) |
|---|---|
| bent_wire | 100.0 ± 0.0 % |
| cable_swap | 87.2 ± 11.8 % |
| combined | 100.0 ± 0.0 % |
| cut_inner_insulation | 100.0 ± 0.0 % |
| cut_outer_insulation | 100.0 ± 0.0 % |
| missing_cable | 100.0 ± 0.0 % |
| missing_wire | 100.0 ± 0.0 % |
| poke_insulation | 87.2 ± 12.3 % |

Lecture : l'AUROC ne dépend pas du seuil et reste inchangé. Un écart important entre les deux lignes du tableau mesure l'optimisme dû au choix du seuil sur les images de test.
Les écarts-types reflètent la petite taille des moitiés : ils sont indicatifs.
