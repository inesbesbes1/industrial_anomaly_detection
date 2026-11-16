# Base de Connaissances Normative — Sentinex Vision QC

## CATÉGORIE : BOTTLE (Verre & Emballage)

### Norme EN ISO 7458 — Résistance aux chocs des bouteilles en verre
- Tout éclat ou fissure visible sur le goulot, le corps ou le fond constitue un défaut CRITIQUE.
- Tolérance zéro pour les défauts de type contamination visible (corps étranger, résidu).
- Défaut broken_large (> 5 mm) : REBUT IMMÉDIAT. Risque de blessure opérateur et contamination produit.
- Défaut broken_small (< 5 mm) : REBUT. Risque de micro-éclats dans le produit.
- Défaut contamination : REBUT. Non-conformité alimentaire ou pharmaceutique.
- Action requise : Éjecter la pièce, alerter le responsable ligne, purger les 10 pièces précédentes.

### Norme ISO 9001:2015 — Système de Management de la Qualité
- Toute non-conformité doit être enregistrée avec un ID unique et horodatée.
- Le rapport de non-conformité doit inclure : ID pièce, type de défaut, score d'anomalie, action corrective.

---

## CATÉGORIE : CABLE (Faisceaux Électriques)

### Norme IPC-A-620E — Exigences et Acceptation des Assemblages de Câblage
- Classe 3 (Électronique critique) : Tolérance zéro sur les fils dénudés ou coupés.
- Défaut cut_inner_insulation : REBUT. Risque de court-circuit et incendie.
- Défaut cut_outer_insulation : REBUT. Protection électrique compromise.
- Défaut missing_wire : REBUT IMMÉDIAT. Connexion électrique incomplète.
- Défaut missing_cable : REBUT IMMÉDIAT. Sous-ensemble incomplet.
- Défaut bent_wire : REVIEW. Inspection visuelle complémentaire requise.
- Défaut cable_swap : REVIEW. Vérification de la continuité électrique obligatoire.
- Défaut poke_insulation : REVIEW. Test de rigidité diélectrique recommandé.
- Action requise pour REBUT : Isolation immédiate de la pièce, traçabilité du lot, analyse des causes.

### Norme UL 758 — Câbles d'Équipement Électrique
- Épaisseur d'isolation minimale : 0.4 mm. Tout amincissement visible → REVIEW obligatoire.

---

## CATÉGORIE : LEATHER (Maroquinerie & Sellerie)

### Norme EN ISO 17235 — Cuir : Détermination de la Souplesse
- Défaut cut : REBUT. Toute coupure traversante compromet la résistance mécanique.
- Défaut fold : REVIEW. Pli permanent réduisant l'esthétique. Acceptable pour usage intérieur uniquement.
- Défaut poke : REBUT. Perforation visible non intentionnelle.
- Défaut glue : REVIEW. Résidu adhésif. Nettoyage possible si < 5 mm².
- Défaut color : REVIEW. Variation chromatique. Comparaison avec l'échantillon de référence colorimétrique.
- Action requise REBUT : Mise à l'écart pour découpe en pièces secondaires.

### Norme Automobile VDA 6.1 — Qualité Sellerie
- Surface visible (Classe A) : Tolérance zéro défaut > 1 mm. REBUT systématique.
- Surface non visible (Classe B) : Défauts < 3 mm tolérés avec REVIEW.

---

## CATÉGORIE : TRANSISTOR (Composants Électroniques)

### Norme IPC-A-610H — Acceptabilité des Assemblages Électroniques
- Défaut bent_lead : REBUT si déviation > 10° par rapport à l'axe. Risque de mauvais contact.
- Défaut cut_lead : REBUT IMMÉDIAT. Connexion électrique impossible.
- Défaut damaged_case : REBUT. Boîtier fissuré = composant inutilisable.
- Défaut misplaced : REBUT IMMÉDIAT. Composant absent ou inversé = court-circuit garanti.
- Action requise : Vérification de l'orientation par test électrique avant rebut définitif.

### Norme JEDEC JESD22 — Tests de Fiabilité Composants
- Composant endommagé visuellement : Isoler et ne pas soumettre aux tests électriques (risque ESD).