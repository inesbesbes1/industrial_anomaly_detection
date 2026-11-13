#!/usr/bin/env bash
# Enchaîne : (attente fin bottle) -> train cable -> train transistor -> evaluate -> error_analysis
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

# 1. attendre la fin de l'entraînement bottle (lancé séparément)
until grep -qE "checkpoint :|Traceback" logs/train_bottle.log 2>/dev/null; do sleep 30; done
if grep -q Traceback logs/train_bottle.log; then echo "ECHEC train bottle"; exit 1; fi
echo "$(date +%H:%M) bottle OK"

for cat in cable transistor; do
  echo "$(date +%H:%M) debut train $cat"
  $PY -u -m detection.train --category $cat > logs/train_$cat.log 2>&1 || { echo "ECHEC train $cat"; exit 1; }
  echo "$(date +%H:%M) $cat OK"
done

echo "$(date +%H:%M) debut evaluate"
$PY -u -m detection.evaluate > logs/evaluate.log 2>&1 || { echo "ECHEC evaluate"; exit 1; }
echo "$(date +%H:%M) debut error_analysis"
$PY -u -m detection.error_analysis > logs/error_analysis.log 2>&1 || { echo "ECHEC error_analysis"; exit 1; }
echo "$(date +%H:%M) CHAINE TERMINEE"
