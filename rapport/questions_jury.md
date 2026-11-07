# Questions probables du jury (partie A) et éléments de réponse

**1. Pourquoi ne pas faire une classification supervisée ?**
Les défauts sont rares et très variés. Un classifieur ne reconnaît que les classes vues pendant l'entraînement.
La détection d'anomalies n'a besoin que d'images normales, qui sont abondantes, et peut signaler des défauts
jamais vus.

**2. PatchCore « s'entraîne »-t-il vraiment ?**
Pas au sens de la descente de gradient. Le backbone reste gelé. L'« entraînement » consiste à extraire les
features des images normales et à construire la banque mémoire (avec sous-échantillonnage coreset).

**3. Pourquoi les couches layer2 et layer3, et pas la dernière couche ?**
Les couches profondes sont spécialisées dans les classes d'ImageNet et ont une résolution spatiale trop faible.
Les couches intermédiaires gardent assez de détails pour localiser, tout en étant assez sémantiques pour être
robustes.

**4. À quoi sert le coreset ?**
À réduire la taille de la banque mémoire, donc la mémoire utilisée et le temps de recherche du plus proche
voisin. L'algorithme glouton garde des patchs qui couvrent bien toute la diversité des features normales, donc
on perd très peu en performance.

**5. Qu'est-ce que l'AUROC, et pourquoi l'utiliser ?**
C'est la probabilité qu'une image défectueuse tirée au hasard ait un score plus élevé qu'une image normale tirée
au hasard. Elle ne dépend pas du seuil, ce qui permet de comparer des modèles indépendamment du réglage.

**6. Différence entre AUROC image et AUROC pixel ?**
L'AUROC image évalue la **détection** (cette image est-elle défectueuse ?). L'AUROC pixel évalue la
**localisation** (où est le défaut ?), en comparant la carte d'anomalie au masque de vérité terrain.

**7. Comment le seuil est-il choisi ? N'est-ce pas de la triche ?**
On prend le seuil qui maximise le F1 sur la validation. Avec le protocole standard, la validation est le test,
donc le F1 est optimiste. C'est pour cela que l'AUROC, qui ne dépend pas du seuil, est la métrique principale.
On peut aussi réserver une partie du test pour fixer le seuil (`val_split_mode: from_test`).

**8. Pourquoi un modèle par catégorie ?**
La « normalité » d'une bouteille n'a rien à voir avec celle d'un câble. C'est le protocole standard de MVTec AD.

**9. Que se passe-t-il avec une photo prise au téléphone ?**
Elle sera probablement jugée anormale. Le modèle compare l'image aux conditions de prise de vue de MVTec
(éclairage, fond, cadrage), et une photo différente s'écarte de cette normalité. En industrie, la caméra est
fixe et l'éclairage contrôlé, donc ce problème ne se pose pas sur la ligne.

**10. Quelles sont les limites de PatchCore ?**
- Anomalies « logiques » (par exemple des éléments normaux mais mal placés), difficiles à voir localement.
- Mémoire et temps de recherche qui augmentent avec la banque.
- Sensibilité au changement de domaine.

**11. Pourquoi PatchCore plutôt qu'un autre modèle ?**
C'est l'une des méthodes de référence sur MVTec AD : très performante, sans entraînement par gradient, et
disponible dans Anomalib. (Bonus : comparaison avec PaDiM dans l'étude d'ablation.)

**12. Comment le LLM utilise-t-il vos sorties ?**
`predict()` renvoie le statut, un score normalisé (seuil = 0,5) et la carte d'anomalie. B en extrait les zones
suspectes et le LLM rédige le rapport **à partir de ces données uniquement** : il ne décide jamais lui-même de
la présence d'un défaut.
