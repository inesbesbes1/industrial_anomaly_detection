#!/usr/bin/env bash
# Lance toute la partie A d'un coup (Linux / macOS / Colab).
# Usage : bash scripts/run_all.sh [dossier_des_archives]
set -e
cd "$(dirname "$0")/.."
ARCH=${1:-.}

echo "=== 1/4 Préparation des données ==="
python -m detection.prepare_data --archives-dir "$ARCH"

echo "=== 2/4 Entraînement PatchCore ==="
python -m detection.train

echo "=== 3/4 Évaluation ==="
python -m detection.evaluate

echo "=== 4/4 Analyse des erreurs ==="
python -m detection.error_analysis

echo ""
echo "Terminé. Résultats : results/results_table.md et results/<categorie>/"
