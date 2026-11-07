# Plan des slides : partie A (modèle de détection)

Environ 6 slides, 5 à 6 minutes d'oral.

## 1. Pourquoi la détection d'anomalies ?
- Les défauts sont rares et imprévisibles : impossible d'étiqueter tous les cas.
- On apprend **la normalité**, et on signale ce qui s'en écarte.
- Visuel : une image normale à côté d'une image défectueuse (`dataset_samples_*.png`).

## 2. Le dataset MVTec AD
- Bottle et Cable : nombre d'images train/test, types de défauts (tableau `dataset_stats.md`).
- Le train ne contient **que des images normales**.

## 3. PatchCore en une slide
- Schéma : image → Wide ResNet-50 gelé → features de patchs → banque mémoire (coreset 10 %)
  → distance au plus proche voisin → carte d'anomalie + score.
- Message clé : **pas d'entraînement par gradient, un seul passage sur les images normales.**

## 4. Résultats quantitatifs
- Tableau : AUROC image, AUROC pixel, F1 pour Bottle et Cable.
- Courbe ROC + histogramme des scores (une catégorie suffit).

## 5. Localisation : exemples
- `best_detections.png` : la heatmap tombe sur le défaut.
- 1 ou 2 types de défauts intéressants (`examples_<type>.png`).

## 6. Limites et erreurs
- `errors_fp_fn.png` : un faux positif et un faux négatif commentés.
- Types de défauts les plus difficiles (`per_defect.png`).
- Changement de domaine (photos hors MVTec), anomalies logiques.
- (Bonus) Ablation : coreset, backbone, PaDiM.

**Transition vers B** : « Le modèle produit un score et une carte, mais un inspecteur a besoin d'une explication
claire. C'est le rôle du module LLM. »
