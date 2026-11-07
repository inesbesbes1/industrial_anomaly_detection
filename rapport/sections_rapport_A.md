# Sections du rapport : partie A (brouillon)

> Brouillon à adapter avec vos mots. Les passages **[À COMPLÉTER]** se remplissent à partir des fichiers
> générés dans `results/`. Les figures citées sont produites automatiquement par le code.

---

## X. Détection d'anomalies par PatchCore

### X.1 Principe : apprendre la normalité

En contrôle qualité industriel, les produits défectueux sont rares et leurs défauts très variés. Il est donc
difficile de constituer un jeu de données étiqueté qui couvre tous les cas possibles. Plutôt que d'apprendre à
reconnaître des défauts connus (classification supervisée), nous apprenons uniquement **à quoi ressemble un
produit normal**. Toute région qui s'écarte de cette normalité est ensuite considérée comme suspecte. Ce cadre,
appelé détection d'anomalies *one-class*, permet en principe de détecter des défauts jamais vus pendant
l'entraînement.

Nous utilisons **PatchCore** (Roth et al., CVPR 2022), via la bibliothèque open source **Anomalib**.

### X.2 Fonctionnement de PatchCore

PatchCore se déroule en quatre étapes.

**1. Extraction de features de patchs.** Chaque image est passée dans un réseau convolutif pré-entraîné sur
ImageNet (Wide ResNet-50-2), qui n'est **pas ré-entraîné**. On récupère les cartes de caractéristiques de
deux couches intermédiaires (`layer2` et `layer3`). Les couches trop profondes sont évitées, car elles sont
spécialisées dans la classification ImageNet et perdent l'information spatiale fine. Chaque position de ces
cartes décrit un petit patch de l'image. Pour rendre la description robuste à de petits décalages, chaque
feature est moyennée avec celles de son voisinage (3×3). Les features de `layer3` sont ensuite
ré-échantillonnées à la résolution de `layer2` et concaténées.

**2. Banque mémoire.** Les features de **tous les patchs de toutes les images normales** d'entraînement sont
stockées dans une banque mémoire. Il n'y a ni descente de gradient ni époques : un seul passage sur les données
suffit.

**3. Sous-échantillonnage par coreset.** La banque complète est très grande (environ 1 000 patchs par image
pour plus de 200 images). PatchCore n'en garde qu'un sous-ensemble représentatif, choisi par un algorithme
glouton (*k-center greedy*) : on ajoute à chaque fois le patch le plus éloigné de ceux déjà retenus, ce qui
couvre au mieux l'espace des features normales. Nous gardons 10 % des patchs. La mémoire et le temps
d'inférence diminuent fortement, sans perte notable de performance.

**4. Score d'anomalie.** Pour une image de test, on extrait ses features de patchs de la même manière. Pour
chaque patch, on cherche son **plus proche voisin** dans la banque mémoire : plus la distance est grande, plus
le patch est anormal.
- La **carte d'anomalie** (localisation) est formée de ces distances, remises à la taille de l'image et
  lissées par un filtre gaussien.
- Le **score image** est la distance du patch le plus anormal, légèrement repondérée selon la densité de la
  banque autour de son plus proche voisin (`num_neighbors = 9`).

**Seuil de décision.** Une image est déclarée défectueuse si son score dépasse un seuil. Anomalib choisit
automatiquement ce seuil : c'est la valeur qui maximise le score F1 sur le jeu de validation. Les scores sont
ensuite normalisés pour que le seuil corresponde à 0,5.

> Figure suggérée : schéma du pipeline (image → backbone → features de patchs → banque mémoire / coreset →
> plus proche voisin → carte d'anomalie + score).

### X.3 Prétraitement et paramètres

| Paramètre | Valeur |
|---|---|
| Taille d'entrée | 256 × 256 (redimensionnement), normalisation ImageNet |
| Backbone | Wide ResNet-50-2 pré-entraîné ImageNet (gelé) |
| Couches | layer2 + layer3 |
| Taux de coreset | 10 % |
| Voisins pour la repondération | 9 |
| Seuil | F1 maximal sur la validation (adaptatif) |
| Bibliothèque | Anomalib 2.7.0 (PyTorch, Lightning) |
| Matériel | **[À COMPLÉTER : GPU / CPU utilisé]** |

Un modèle distinct est entraîné pour chaque catégorie (Bottle, Cable), comme dans le protocole standard de
MVTec AD.

---

## Y. Protocole expérimental

### Y.1 Données

Nous utilisons deux catégories de **MVTec AD** :
- **Bottle** : bouteilles vues de dessus, avec des défauts de surface et de structure.
- **Cable** : sections de câbles à trois conducteurs, avec des défauts structurels (coupures, isolant
  endommagé, fils manquants ou inversés).

L'entraînement utilise **uniquement des images normales**. Le test contient des images normales et des images
défectueuses, avec des masques de vérité terrain au pixel près.

**[À COMPLÉTER : coller ici le tableau de `results/dataset_stats.md`]**

> Figures : `results/figures/dataset_samples_bottle.png`, `dataset_samples_cable.png`

### Y.2 Métriques

- **AUROC image** : aire sous la courbe ROC calculée sur les scores image. Elle mesure la capacité du modèle à
  classer une image défectueuse plus haut qu'une image normale, **indépendamment du seuil**. 1,0 correspond à
  une séparation parfaite, 0,5 au hasard. C'est la métrique de référence sur MVTec AD.
- **AUROC pixel** : même principe, mais au niveau de chaque pixel, en comparant la carte d'anomalie au masque de
  vérité terrain. Elle mesure la qualité de la **localisation**.
- **Précision, rappel, F1, matrice de confusion** : calculés au seuil choisi. Ils montrent le comportement du
  système une fois la décision prise (fausses alarmes, défauts manqués).

**Remarque méthodologique.** Nous suivons le protocole courant des articles : le seuil est choisi sur le jeu de
test lui-même. Le F1 obtenu est donc une borne optimiste. L'AUROC, elle, ne dépend d'aucun seuil.
**[Optionnel : si vous relancez avec `val_split_mode: from_test`, comparez les deux F1 ici.]**

---

## Z. Résultats

### Z.1 Résultats globaux

**[À COMPLÉTER : coller le tableau « Résultats globaux » de `results/results_table.md`]**

Points à commenter :
- Comparer vos AUROC avec celles rapportées dans l'article PatchCore pour Bottle et Cable
  (**[À COMPLÉTER : valeurs du tableau de l'article]**). Les écarts éventuels peuvent venir de la taille
  d'entrée (pas de recadrage 224), de la version d'Anomalib, etc.
- Quelle catégorie est la plus difficile, et pourquoi ?

> Figures : `results/<cat>/figures/roc.png`, `score_hist.png`, `confusion.png`

L'histogramme des scores (`score_hist.png`) montre la séparation entre images normales et défectueuses. Le
chevauchement des deux distributions explique directement les erreurs au seuil choisi.

### Z.2 Résultats par type de défaut

**[À COMPLÉTER : coller les tableaux « détail par type de défaut » de `results/results_table.md`]**

> Figures : `results/<cat>/figures/per_defect.png`, `scores_by_type.png`

À discuter :
- Les défauts **visibles localement** (contamination, casse, coupure d'isolant) se prêtent bien à une méthode
  par patchs.
- Pour Cable, certaines anomalies sont **« logiques »**. Par exemple, dans `cable_swap`, chaque conducteur a
  une apparence normale, mais ils ne sont pas à la bonne place. Une méthode qui compare des patchs locaux
  peut avoir plus de mal à les détecter. **[À vérifier dans vos résultats et à commenter.]**

### Z.3 Analyse qualitative et analyse des erreurs

> Figures : `best_detections.png`, `examples_<type>.png`, `errors_fp_fn.png` (dans `results/<cat>/figures/`)

**[À COMPLÉTER à partir de `results/<cat>/error_analysis.md`]** :
- Nombre de faux positifs et de faux négatifs.
- Pour chaque erreur importante : qu'est-ce qui la provoque ? (défaut très petit ou peu contrasté, variation
  normale absente du train, reflet…)
- La carte d'anomalie pointe-t-elle au bon endroit, y compris pour les erreurs ? (une image mal classée peut
  quand même avoir une localisation correcte)

### Z.4 (Bonus) Étude d'ablation

**[À COMPLÉTER : tableau `results/<cat>/ablation/ablation.md`]**

À commenter : effet du taux de coreset sur la performance, la mémoire et la vitesse ; backbone léger
(ResNet-18) contre Wide ResNet-50 ; PatchCore contre PaDiM.

---

## Limites (pour la discussion / conclusion)

- **Changement de domaine** : le modèle ne connaît que les conditions de prise de vue de MVTec (éclairage,
  cadrage, fond). Une image prise dans d'autres conditions risque d'être signalée comme anormale.
- **Un modèle par catégorie** : un nouveau produit nécessite un nouvel entraînement, mais il suffit d'images
  normales et l'entraînement est rapide.
- **Seuil** : il est choisi sur des données étiquetées. En production, il faudrait le calibrer sur un lot de
  validation représentatif.
- **Anomalies logiques** : elles sont plus difficiles à détecter pour une approche qui compare des patchs locaux.

## Référence

K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, P. Gehler. *Towards Total Recall in Industrial Anomaly
Detection.* CVPR 2022. arXiv:2106.08265.
