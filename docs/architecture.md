# Architecture

## System overview

```text
                    ┌──────────────────────────────────────────────┐
                    │                Frontend (React/TS/Vite)      │
                    │  Dashboard · Inspect · Result · History ·   │
                    │  Analytics · Model Info                     │
                    └───────────────┬──────────────────────────────┘
                                    │ HTTP /api (fetch, JSON + files)
                    ┌───────────────▼──────────────────────────────┐
                    │              Backend (FastAPI)                │
                    │  routes/  inspections · platform · storage   │
                    │  services/ inspection · image · model ·      │
                    │            registry · severity               │
                    │  schemas/ (Pydantic)  core/ (config, exc.)   │
                    └───┬───────────────┬───────────────┬──────────┘
                        │               │               │
             ┌──────────▼─────┐ ┌───────▼──────┐ ┌──────▼────────────┐
             │   SQLite /     │ │  Storage     │ │  ML runtime        │
             │   PostgreSQL   │ │  (uploads +  │ │  DefectPredictor   │
             │   SQLAlchemy   │ │  results)    │ │  + GradCAM         │
             │   Alembic      │ │  filesystem  │ │  (torch, CPU)      │
             └────────────────┘ └──────────────┘ └───────────────────┘
                        trained artifacts: models/experiments/<exp>/best.pth
```

## Inference request flow

```text
POST /api/inspections (multipart)
  → header pre-check (content-type family)
  → image_service.validate_and_sanitize
        size cap → extension allow-list → magic-byte sniff →
        format/extension agreement → PIL verify() → decode to grayscale
  → image_service.store_image  (server-generated filename; re-encode JPEG)
  → predictor.predict_pil     (resize 224² → normalize → forward → softmax)
  → GradCAM.generate           (backward pass → heatmap → bbox+coverage)
  → confidence_tier            (policy: ≥.80 high / ≥.60 medium / <.60 low)
  → assess_severity            (pure business rules; see severity.md)
  → build_overlay              (evidence overlay JPEG)
  → persist Inspection + Prediction (single transaction)
  → 201 JSON response
```

Failures map to precise HTTP codes: 413 oversize, 422 invalid image /
mismatched content, 409 duplicate feedback, 404 not found, 503 model missing.

## Key decisions & rationale

**Why classification + Grad-CAM rather than detection/segmentation?**
The accessible dataset provides class labels (no boxes/masks). A classifier
with Grad-CAM evidence is honest about what the data supports; a detector
trained on fabricated boxes would not be. See `docs/explainability.md`.

**Why images on the filesystem, not BLOBs?**
The DB stores POSIX-relative paths (`uploads/…jpg`, `results/…jpg`) under
`backend/storage/`. Serving images through the DB would bloat rows, slow
backups, and complicate caching. The storage route resolves paths strictly
inside the storage root (traversal-safe).

**Model lifecycle.** The predictor is a process-level singleton loaded at
startup (once), avoiding per-request initialization (~seconds of overhead).
Concurrent requests serialize on the torch forward pass (GIL); the FastAPI
threadpool keeps I/O (uploads, DB) concurrent. For CPU deployment this is
the correct, simple concurrency strategy; scale-out is per-worker processes.

**Model versioning.** Every `Prediction` row stores the `model_version` that
produced it; `Feedback` stores the version whose prediction it judges. This
makes analytics comparable across retrains and enables drift analysis.

**Migrations.** Alembic owns the schema in production (`alembic upgrade head`
runs in the container entrypoint). `Base.metadata.create_all` remains only
as a dev/test convenience at startup.

**MLOps.** Each training run writes an experiment directory
(`best.pth`, `last.pth`, `history.json`, `metadata.json`,
`test_metrics.json`, `confusion_matrix.png`) — fully reproducible via
`scripts/train.sh` + YAML config + seed. The backend reads those artifacts to
build the model registry row; no metrics are ever hand-entered.

**Human-in-the-loop.** Feedback rows are immutable, one per inspection, and
carry `review_status` (pending/approved/rejected). `scripts/export_feedback.py`
exports approved rows for dataset curation; nothing auto-retrains. The
model team decides at training time what enters the next run.
