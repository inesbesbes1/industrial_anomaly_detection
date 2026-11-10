## Résultats globaux

| Catégorie | AUROC image | AUROC pixel | F1 image | Précision | Rappel | Fausses alarmes | Temps / image |
|---|---|---|---|---|---|---|---|
| bottle | 100.0 % | 98.6 % | 100.0 % | 100.0 % | 100.0 % | 0.0 % | 1451 ms |
| cable | 98.7 % | 98.5 % | 96.8 % | 95.7 % | 97.8 % | 6.9 % | 2107 ms |
| transistor | 99.4 % | 97.3 % | 96.3 % | 95.1 % | 97.5 % | 3.3 % | 1391 ms |


### bottle : détail par type de défaut

| Type de défaut | Images | AUROC image | Taux de détection | AUROC pixel |
|---|---|---|---|---|
| broken_large | 20 | 100.0 % | 100.0 % | 98.3 % |
| broken_small | 22 | 100.0 % | 100.0 % | 99.4 % |
| contamination | 21 | 100.0 % | 100.0 % | 97.3 % |

### cable : détail par type de défaut

| Type de défaut | Images | AUROC image | Taux de détection | AUROC pixel |
|---|---|---|---|---|
| bent_wire | 13 | 100.0 % | 100.0 % | 98.6 % |
| cable_swap | 12 | 93.8 % | 91.7 % | 96.7 % |
| combined | 11 | 100.0 % | 100.0 % | 97.0 % |
| cut_inner_insulation | 14 | 99.4 % | 100.0 % | 98.4 % |
| cut_outer_insulation | 10 | 99.5 % | 100.0 % | 93.3 % |
| missing_cable | 12 | 100.0 % | 100.0 % | 99.5 % |
| missing_wire | 10 | 99.8 % | 100.0 % | 99.5 % |
| poke_insulation | 10 | 96.9 % | 90.0 % | 98.7 % |

### transistor : détail par type de défaut

| Type de défaut | Images | AUROC image | Taux de détection | AUROC pixel |
|---|---|---|---|---|
| bent_lead | 10 | 99.7 % | 100.0 % | 98.3 % |
| cut_lead | 10 | 99.8 % | 100.0 % | 97.2 % |
| damaged_case | 10 | 99.8 % | 100.0 % | 98.6 % |
| misplaced | 10 | 98.2 % | 90.0 % | 86.5 % |
