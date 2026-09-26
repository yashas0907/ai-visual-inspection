# Backend: FastAPI + ML inference (CPU torch)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (minimal; opencv-headless needs libgl only on some hosts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps (torch CPU wheels first, then app deps)
COPY backend/requirements.txt requirements-backend.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1 torchvision==0.20.1
RUN pip install -r requirements-backend.txt pyarrow==18.1.0

# Application code (ml/ + backend/ only; dataset & artifacts mounted or baked via compose)
COPY ml/ ml/
COPY backend/ backend/

ENV PYTHONPATH=/app:/app/backend \
    VI_DATABASE_URL=sqlite:////app/backend/data/inspection.db \
    VI_UPLOAD_DIR=/app/backend/storage/uploads

# Persistent-state dirs (named volumes mount onto these in compose)
RUN mkdir -p /app/backend/data /app/backend/storage/uploads /app/backend/storage/results

EXPOSE 8000

WORKDIR /app/backend

# Runs migrations then starts uvicorn
CMD ["sh", "-c", "python -m alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1"]
