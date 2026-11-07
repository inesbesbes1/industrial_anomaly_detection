# Analyse des erreurs - cable

- Images de test : 150 (58 normales, 92 défectueuses)
- Seuil (score brut) : 42.6672
- Faux positifs : **4** (images normales signalées comme défectueuses)
- Faux négatifs : **2** (défauts non détectés)

## Par type de défaut

| Type | Images | Taux de détection | AUROC pixel | Score moyen (brut) |
|---|---|---|---|---|
| bent_wire | 13 | 100.0 % | 98.6 % | 66.083 |
| cable_swap | 12 | 91.7 % | 96.7 % | 43.865 |
| combined | 11 | 100.0 % | 97.0 % | 62.182 |
| cut_inner_insulation | 14 | 100.0 % | 98.4 % | 51.824 |
| cut_outer_insulation | 10 | 100.0 % | 93.3 % | 52.455 |
| missing_cable | 12 | 100.0 % | 99.5 % | 64.991 |
| missing_wire | 10 | 100.0 % | 99.5 % | 53.992 |
| poke_insulation | 10 | 90.0 % | 98.7 % | 46.041 |

- Type le moins bien détecté : **poke_insulation** (90.0 % détectés)
- Type le moins bien localisé : **cut_outer_insulation**

## Liste des erreurs

| Type d'erreur | Fichier | Catégorie de défaut | Score brut |
|---|---|---|---|
| FP | 057.png | good | 48.1822 |
| FP | 039.png | good | 45.1888 |
| FP | 030.png | good | 44.6024 |
| FP | 013.png | good | 43.0709 |
| FN | 006.png | cable_swap | 40.8140 |
| FN | 002.png | poke_insulation | 41.3989 |

## Pistes d'interprétation (à rédiger dans le rapport)

- Faux négatifs : défauts petits ou peu contrastés, proches de l'apparence normale ?
- Faux positifs : variations normales (reflets, position du câble, bruit) absentes du train ?
- Comparer les figures `examples_<type>.png` : la heatmap pointe-t-elle au bon endroit ?
- Effet du seuil : voir `score_hist.png` (chevauchement des deux distributions).
