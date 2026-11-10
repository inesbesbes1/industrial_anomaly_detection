# Analyse des erreurs - transistor

- Images de test : 100 (60 normales, 40 défectueuses)
- Seuil (score brut) : 37.6774
- Faux positifs : **2** (images normales signalées comme défectueuses)
- Faux négatifs : **1** (défauts non détectés)

## Par type de défaut

| Type | Images | Taux de détection | AUROC pixel | Score moyen (brut) |
|---|---|---|---|---|
| bent_lead | 10 | 100.0 % | 98.3 % | 49.454 |
| cut_lead | 10 | 100.0 % | 97.2 % | 47.701 |
| damaged_case | 10 | 100.0 % | 98.6 % | 48.583 |
| misplaced | 10 | 90.0 % | 86.5 % | 50.189 |

- Type le moins bien détecté : **misplaced** (90.0 % détectés)
- Type le moins bien localisé : **misplaced**

## Liste des erreurs

| Type d'erreur | Fichier | Catégorie de défaut | Score brut |
|---|---|---|---|
| FP | 014.png | good | 44.9057 |
| FP | 045.png | good | 39.6366 |
| FN | 002.png | misplaced | 36.2201 |

## Pistes d'interprétation (à rédiger dans le rapport)

- Faux négatifs : défauts petits ou peu contrastés, proches de l'apparence normale ?
- Faux positifs : variations normales (reflets, position du câble, bruit) absentes du train ?
- Comparer les figures `examples_<type>.png` : la heatmap pointe-t-elle au bon endroit ?
- Effet du seuil : voir `score_hist.png` (chevauchement des deux distributions).
