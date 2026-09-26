# Single-container deployment image (HF Spaces / any Docker host).
# Self-contained: frontend build + trained model + API in one container.
#
# Build (from repo root, after training):
#   docker build -f deploy/Dockerfile -t surfacespec .
# Requires models/experiments/v1_resnet18/best.pth to exist (train first).

# ---- stage 1: python deps -------------------------------------------------
FROM python:3.12-slim AS deps
ENV PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1 \
    && pip install fastapi==0.115.5 uvicorn[standard]==0.32.1 python-multipart==0.0.17 \
        sqlalchemy==2.0.36 alembic==1.14.0 pydantic==2.9.2 pydantic-settings==2.6.1 \
        python-dotenv==1.0.1 numpy==1.26.4 pillow==11.0.0

# ---- stage 2: frontend build ----------------------------------------------
FROM node:20-alpine AS frontend-build
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- stage 3: final image --------------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/backend
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY ml/ ml/
COPY backend/app/ backend/app/
COPY backend/alembic.ini backend/alembic.ini
COPY backend/alembic/ backend/alembic/
COPY --from=frontend-build /web/dist /app/frontend_dist

# Trained artifacts must exist locally (train via scripts/train.sh first).
COPY models/ models/

ENV VI_DATABASE_URL=sqlite:////app/data/inspection.db \
    VI_UPLOAD_DIR=/app/data/storage/uploads \
    VI_FRONTEND_DIST=/app/frontend_dist \
    VI_MODEL_CHECKPOINT=/app/models/experiments/v1_resnet18/best.pth \
    VI_MODEL_VERSION=v1.0.0 \
    VI_MODEL_EXPERIMENT=v1_resnet18 \
    VI_CORS_ORIGINS="*"

RUN mkdir -p /app/data/storage/uploads /app/data/storage/results

# HF Spaces expects the app on port 7860; harmless elsewhere.
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "7860"]
