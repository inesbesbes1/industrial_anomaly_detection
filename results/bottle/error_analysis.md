# Analyse des erreurs - bottle

- Images de test : 83 (20 normales, 63 défectueuses)
- Seuil (score brut) : 37.8718
- Faux positifs : **0** (images normales signalées comme défectueuses)
- Faux négatifs : **0** (défauts non détectés)

## Par type de défaut

| Type | Images | Taux de détection | AUROC pixel | Score moyen (brut) |
|---|---|---|---|---|
| broken_large | 20 | 100.0 % | 98.3 % | 62.021 |
| broken_small | 22 | 100.0 % | 99.4 % | 59.337 |
| contamination | 21 | 100.0 % | 97.3 % | 55.992 |

- Type le moins bien détecté : **broken_large** (100.0 % détectés)
- Type le moins bien localisé : **contamination**


## Pistes d'interprétation (à rédiger dans le rapport)

- Faux négatifs : défauts petits ou peu contrastés, proches de l'apparence normale ?
- Faux positifs : variations normales (reflets, position du câble, bruit) absentes du train ?
- Comparer les figures `examples_<type>.png` : la heatmap pointe-t-elle au bon endroit ?
- Effet du seuil : voir `score_hist.png` (chevauchement des deux distributions).
