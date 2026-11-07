# Partie A : modèle de détection (PatchCore / Anomalib)

Projet **Industrial Defect Analysis and Intelligent Quality Assistant** : partie de la Personne A.

Ce dossier couvre toutes les tâches de A dans la répartition :

| # | Tâche | Où |
|---|---|---|
| 1 | Dataset MVTec AD (Bottle, Cable) | `detection/prepare_data.py` |
| 2 | Prétraitement | `config.yaml` (section `model`) + `detection/common.py` |
| 3 | Entraînement PatchCore | `detection/train.py` |
| 4 | Évaluation (AUROC image/pixel, F1, seuil) | `detection/evaluate.py` |
| 5 | Analyse des erreurs | `detection/error_analysis.py` |
| 6 | Fonction d'inférence `predict()` | `detection/predict.py` (à donner à B) |
| 7 | Rapport : méthode + résultats | `rapport/sections_rapport_A.md` |
| 8 | Soutenance | `rapport/plan_slides_A.md`, `rapport/questions_jury.md` |
| Bonus | Ablation + comparaison PaDiM | `detection/ablation.py` |

---

## 1. Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

- Python **3.10 à 3.13**.
- **GPU fortement conseillé.** Sur CPU, l'entraînement marche aussi mais prend nettement plus de temps,
  surtout l'étape « Selecting Coreset Indices ». Si vous n'avez pas de GPU, utilisez Google Colab
  (`notebooks/colab_partie_A.ipynb`).
- Au premier lancement, les poids ImageNet du backbone `wide_resnet50_2` sont téléchargés automatiquement
  (environ 270 Mo), il faut donc une connexion internet.

## 2. Vérifier que tout marche (2 minutes)

Avant les vraies données, lancez le test de bout en bout sur un faux mini-dataset :

```bash
bash tests/smoke_test.sh
```

Il doit se terminer par `SMOKE TEST OK`. Les résultats de ce test n'ont aucune valeur scientifique :
ils servent juste à vérifier l'installation.

## 3. Lancer sur les vraies données

Placez `bottle.tar.xz` et `cable.tar.xz` à la racine de ce dossier (ou indiquez leur dossier), puis :

```bash
bash scripts/run_all.sh                    # Linux / macOS / Colab
scripts\run_all.bat                        # Windows
bash scripts/run_all.sh ~/Downloads        # si les archives sont ailleurs
```

Ou étape par étape :

```bash
python -m detection.prepare_data           # extraction + statistiques du dataset
python -m detection.train                  # un modèle par catégorie -> models/
python -m detection.evaluate               # métriques + figures -> results/
python -m detection.error_analysis         # FP / FN + exemples -> results/<cat>/figures
```

Chaque commande accepte `--category bottle` ou `--category cable` pour ne traiter qu'une catégorie.

## 4. Ce qui est produit

```
models/patchcore_bottle.ckpt, models/patchcore_cable.ckpt     <- à partager avec B
results/
  dataset_stats.md                     tableau du dataset (pour le rapport)
  results_table.md                     tableau des résultats (pour le rapport)
  figures/dataset_samples_<cat>.png    exemples du dataset
  <cat>/metrics.json                   toutes les métriques
  <cat>/predictions.csv                score et prédiction de chaque image de test
  <cat>/error_analysis.md              résumé des erreurs
  <cat>/figures/
      roc.png  score_hist.png  confusion.png  per_defect.png  scores_by_type.png
      errors_fp_fn.png  best_detections.png  examples_<type_de_defaut>.png
```

## 5. Ce qu'il faut donner à la Personne B

1. Le dossier `detection/` (le code).
2. Les deux fichiers `models/patchcore_*.ckpt` (via Drive, ils sont trop gros pour Git).
3. Le fichier `CONTRAT_INTERFACE.md`, qui explique exactement ce que renvoie `predict()`.

En attendant que les modèles soient prêts, B peut développer avec un faux résultat (exemple dans le contrat).

## 6. Hyperparamètres (`config.yaml`)

| Paramètre | Valeur | Rôle |
|---|---|---|
| `backbone` | `wide_resnet50_2` | réseau pré-entraîné ImageNet qui extrait les features (celui du papier) |
| `layers` | `layer2`, `layer3` | couches intermédiaires : assez locales pour localiser, assez sémantiques pour comprendre |
| `coreset_sampling_ratio` | `0.1` | fraction des patchs gardés dans la banque mémoire |
| `num_neighbors` | `9` | voisins utilisés pour repondérer le score |
| `image_size` | `256 x 256` | taille d'entrée |
| `center_crop` | `null` | `[224, 224]` pour reproduire exactement le papier (mais coupe les bords) |
| `val_split_mode` | `same_as_test` | voir la remarque sur le seuil ci-dessous |

Pour tester d'autres valeurs sans modifier le fichier : `python -m detection.train --coreset 0.01`.

## 7. Remarques importantes pour le rapport et l'oral

- **L'AUROC ne dépend pas du seuil** : c'est la métrique principale du benchmark MVTec.
- **Le seuil** est choisi automatiquement par Anomalib : c'est la valeur qui maximise le F1 sur le jeu de validation.
  Avec `same_as_test` (protocole standard des articles), la validation est le jeu de test : le F1 est donc
  légèrement optimiste. Pour un F1 plus rigoureux, mettez `val_split_mode: from_test` (la moitié du test sert à
  choisir le seuil, l'autre moitié à évaluer).
- **Les scores** : `metrics.json` utilise les scores **bruts** (distance au plus proche voisin dans la banque
  mémoire). `predict()` renvoie un score **normalisé** dans [0, 1] où **0.5 = seuil**, plus simple pour B.
- **Le F1 image** de `metrics.json` peut différer très légèrement de `anomalib_official_metrics`. Anomalib compare
  avec `score > seuil`, alors que nous utilisons `score >= seuil`, ce qui est cohérent avec la façon dont le seuil
  F1 optimal est calculé. La seule image concernée est celle dont le score est exactement égal au seuil. Les
  AUROC sont identiques.

## 8. Bonus : étude d'ablation

```bash
python -m detection.ablation --category bottle
python -m detection.ablation --category cable --only patchcore_ref padim_r18
```

Ce script compare PatchCore (coreset 1 %, 10 %, 25 %, backbone ResNet-18, couche 3 seule) et PaDiM.
Il produit `results/<cat>/ablation/ablation.md`. Comptez le temps de plusieurs entraînements.

## 9. Problèmes fréquents

| Problème | Solution |
|---|---|
| Erreur de DataLoader sous Windows | `num_workers: 0` dans `config.yaml` |
| `CUDA out of memory` | baisser `train_batch_size` / `eval_batch_size` à 4 ou 8 |
| Entraînement très lent | normal sur CPU, passer sur Colab GPU |
| `Catégorie introuvable` | lancer d'abord `python -m detection.prepare_data` |
| Erreurs d'import Anomalib | vérifier que `anomalib==2.7.0` est bien installé (`pip show anomalib`) |
