#!/usr/bin/env bash
# Full ML pipeline: dataset -> train -> evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."
EXPERIMENT="${1:-v1_resnet18}"

PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
export PYTHONPATH="$(pwd)"

echo "== 1/3 Preparing dataset =="
$PY ml/data/prepare.py --config ml/configs/train_v1.yaml

echo "== 2/3 Training ($EXPERIMENT) =="
$PY ml/training/trainer.py --config ml/configs/train_v1.yaml --name "$EXPERIMENT" 2>&1 | tee "train_${EXPERIMENT}.log"

echo "== 3/3 Evaluating on test split =="
$PY ml/evaluation/evaluate.py --config ml/configs/train_v1.yaml --experiment "$EXPERIMENT"

echo "Done. Artifacts in models/experiments/$EXPERIMENT"
