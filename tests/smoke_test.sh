#!/usr/bin/env bash
# Test de bout en bout sur un faux dataset (environ 2 minutes sur CPU).
# Vérifie que tout le pipeline tourne AVANT de lancer sur les vraies données.
set -e
cd "$(dirname "$0")/.."
CFG=tests/config_test.yaml
python tests/make_fake_dataset.py tests/fake_mvtec
python -m detection.train --config $CFG
python -m detection.evaluate --config $CFG
python -m detection.error_analysis --config $CFG
python -m detection.predict --config $CFG --category bottle \
  --image tests/fake_mvtec/bottle/test/contamination/000.png --out tests/_out/overlay_test.png
echo "SMOKE TEST OK"
