| variante | modele | backbone | couches | coreset | auroc_image | auroc_pixel | f1_image | temps_entrainement_s | ms_par_image | taille_banque | ms_forward_cpu |
|---|---|---|---|---|---|---|---|---|---|---|---|
| patchcore_coreset_1pct | Patchcore | wide_resnet50_2 | layer2+layer3 | 0.01 | 1.0 | 0.985 | 1.0 | 685.4 | 2100.0 | 2140 | 638.2 |
| patchcore_layer3_only | Patchcore | wide_resnet50_2 | layer3 | 0.1 | 1.0 | 0.9787 | 1.0 | 322.7 | 1430.1 | 5350 | 1160.4 |
