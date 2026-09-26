#!/usr/bin/env bash
# Assemble a lean, self-contained deployment context and build the image.
#
# Why staging: .dockerignore is shared by every build in this repo (compose
# builds mount artifacts instead of baking them), so instead of fighting
# ignore-pattern semantics we copy exactly what the deployment image needs
# into deploy/_scratch/ and build from there. Deterministic and obvious.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="deploy/_scratch"
rm -rf "$OUT"
mkdir -p "$OUT"/{ml,backend,frontend,models/experiments/v1_resnet18}

# ---- app code ----
cp -r ml/*.py ml/configs "$OUT/ml/"
rm -rf "$OUT/ml/__pycache__"
cp -r backend/app backend/alembic backend/alembic.ini "$OUT/backend/"
rm -rf "$OUT/backend/app/__pycache__" "$OUT/backend/app"/*/__pycache__
cp -r frontend/src frontend/public frontend/package.json frontend/package-lock.json* \
      frontend/index.html frontend/tsconfig.json frontend/vite.config.ts "$OUT/frontend/" 2>/dev/null || true

# ---- trained artifacts (must exist; run scripts/train.sh otherwise) ----
for f in best.pth metadata.json history.json test_metrics.json confusion_matrix.png; do
  src="models/experiments/v1_resnet18/$f"
  if [ -f "$src" ]; then cp "$src" "$OUT/models/experiments/v1_resnet18/"; else echo "WARN: $src missing"; fi
done

# ---- deployment files ----
cp deploy/Dockerfile "$OUT/Dockerfile"

echo "Staged $(du -sh "$OUT" | cut -f1) at $OUT"
docker build -t surfacespec:latest "$OUT"
echo "Built surfacespec:latest"
