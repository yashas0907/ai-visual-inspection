# AI Visual Inspection & Defect Detection Platform

![CI](https://github.com/yashas0907/ai-visual-inspection/actions/workflows/ci.yml/badge.svg)

**Live demo:** https://surfacespec.onrender.com — full platform on free-tier
hosting via a torch-free ONNX serving backend (~159MB RSS; predictions and
Grad-CAM heatmaps verified identical to the torch backend). First request
after ~15 min idle takes ~30–60s while the free instance wakes.

An end-to-end industrial quality-inspection platform for **hot-rolled steel
strip surface defects**. Upload a surface image → the system classifies the
defect type (6 NEU defect classes), explains *where* the model looked
(Grad-CAM), derives operational **severity** through transparent business
rules, persists everything, and lets an inspector **correct the model** —
feeding a gated human-in-the-loop workflow.

This is a portfolio-grade engineering project, not a notebook demo: typed
Python throughout, tested backend, migrations, Docker, CI, documented
architecture decisions and honest limitations.

---

## 1. Overview

Manual surface inspection on a rolling line is slow, fatiguing and
subjective. This platform simulates how a CV model can be deployed *under
human supervision* rather than as an autonomous judge:

- **Model layer** — ResNet18 (transfer learning) classifies the defect.
- **Evidence layer** — Grad-CAM heatmap + approximate bbox show why/where.
- **Policy layer** — configurable confidence tiers + a rule-based severity
  engine keep business logic out of the model.
- **Feedback layer** — inspectors confirm/correct predictions; feedback is
  stored, versioned and exportable for the next training run after review.

## 2. Problem statement

Classify the defect type visible on a hot-rolled steel strip surface image
and turn that prediction into an actionable, auditable quality decision.

## 3. Why it matters

Surface defects cause downstream rejects, claims and line stoppages. A
deployed assistant must therefore be **explainable** (inspectors need to see
where the model looked), **honest about uncertainty** (confidence tiers +
review flags), and **improvable** (structured human feedback). This project
demonstrates exactly that pipeline, end to end.

## 4. Features

- Upload with hard validation (magic bytes, size caps, format agreement,
  re-encoding) and graceful error contracts
- 6-class defect classification with per-class probabilities
- Grad-CAM evidence overlay + approximate defect region + coverage metric
- Configurable confidence tiers (high / medium / low → review)
- Transparent severity engine (defect risk × confidence tier × evidence
  coverage → minor/major/critical + recommended action)
- Inspection history with filtering/search/pagination
- Analytics dashboard computed **only** from stored rows (defect
  distribution, severity distribution, confidence histogram, correction
  rate, latency, per-day volume)
- Human-in-the-loop feedback (confirm/correct + notes), gated review
  workflow before feedback can become training data
- Model registry: every prediction tied to a model version; metrics come
  from real evaluation artifacts
- Model info page with genuine test metrics, training provenance and the
  full severity rubric

## 5. Architecture

```text
React/TS/Vite frontend
        │ /api
FastAPI backend ─── services ─── severity engine (pure rules)
   │            └── image validation/storage (filesystem, traversal-safe)
   ├── SQLite/PostgreSQL via SQLAlchemy + Alembic migrations
   └── ML runtime (singleton predictor + GradCAM, CPU torch)
```

Details & rationale: [`docs/architecture.md`](docs/architecture.md)

## 6. ML pipeline

```text
scripts/train.sh
 ├─ ml/data/prepare.py        download mirror → stratified splits (seed 42)
 ├─ ml/training/trainer.py    ResNet18 transfer learning, warmup+cosine,
 │                            early stopping on val macro-F1, checkpoints,
 │                            history.json + metadata.json per experiment
 └─ ml/evaluation/evaluate.py test-set accuracy/precision/recall/F1 (macro
                              + per-class), confusion matrix PNG, ROC-AUC
                              (macro OVR), CPU latency, model size
```

Configs: `ml/configs/train_v1.yaml` (no magic numbers in code).
Experiments: `models/experiments/<name>/` — `best.pth`, `last.pth`,
`history.json`, `metadata.json`, `test_metrics.json`, `confusion_matrix.png`.

## 7. Dataset

**NEU Surface Defects Database (NEU-DET), classification split** — 1440
grayscale 200×200 images, 6 balanced classes (240 each), downloaded from a
verified Hugging Face mirror. Splits: 1080/216/144 train/val/test
(class-stratified, seed 42). Normalization stats (mean 0.5088, std 0.2095)
computed from the training split.

License caveats, preprocessing details and known limitations:
[`docs/dataset.md`](docs/dataset.md)

## 8. Model

ResNet18 with a 1-channel stem (grayscale), ImageNet-initialized trunk,
fine-tuned end-to-end. Why 18 layers, not more: 1080 training images,
low-res grayscale texture — capacity is not the bottleneck; small models
also keep CPU inference fast (this matters for the latency metrics we
report honestly).

## 9. Evaluation

Genuine numbers produced by this repo on the held-out 144-image test split
(regenerate with `scripts/train.sh`; see
`models/experiments/v1_resnet18/test_metrics.json` after running):

- accuracy, macro precision/recall/F1, per-class P/R/F1
- confusion matrix artifact (PNG)
- macro one-vs-rest ROC-AUC
- mean CPU latency per image + fp32 model size

**No benchmark numbers are claimed that this pipeline did not produce.**
Metrics appear in the UI only after `evaluate.py` writes them.

## 10. Explainability

Grad-CAM heatmap over the last residual stage, upsampled to image size,
with an approximate bbox (90th-percentile threshold + largest connected
component) and coverage ratio feeding the severity engine.
**Limitations are documented explicitly** (evidence ≠ proof, receptive-field
blur, no ground-truth boxes → no IoU claims):
[`docs/explainability.md`](docs/explainability.md)

## 11. Human-in-the-loop workflow

```text
Model: inclusion — 61% (medium)
Inspector: ✗ incorrect → correct label: rolled-in_scale  [+ note]
   └→ Feedback row: immutable, one per inspection, review_status=pending
        └→ Maintainer approves via DB/review step
             └→ scripts/export_feedback.py → manifest for dataset curation
                  └→ next training run (human decision, never automatic)
```

Feedback stores image reference, original prediction, confidence, model
version, human label, timestamp, notes. The production model is **never**
retrained automatically from user feedback.

## 12. Backend

FastAPI + Pydantic v2 schemas + service layer + SQLAlchemy 2.0 ORM +
Alembic. Structured error envelope, logging, startup-safe degraded mode
(app boots without model artifacts; `/api/health` reports it).

## 13. Frontend

React 18 + TypeScript + Vite (no UI framework — hand-rolled dark industrial
theme). Pages: Dashboard, New Inspection, Result (image + overlay +
probabilities + severity breakdown + feedback), History (filters/search),
Analytics, Model Info. Build output ~195 KB JS (61 KB gzip).

## 14. Database

Entities: `inspections`, `predictions` (model_version, confidence, severity
components, evidence), `feedback` (immutable, review-gated), `model_versions`
(registry with genuine metrics). Images are **not** BLOBs — filesystem
storage with traversal-safe relative-path references. Alembic migration
`0001` defines the schema; SQLite for dev, PostgreSQL-ready via env.

## 15. API

[`docs/api.md`](docs/api.md) + interactive Swagger at `/api/docs`.

## 16. Docker & cloud deployment

- `backend/Dockerfile` — Python 3.12-slim, CPU torch, runs Alembic then
  uvicorn. Trained artifacts are **mounted** (`./models:/app/models:ro`) —
  images stay lean and datasets never enter the image. Includes a
  container-level healthcheck (`/api/health`).
- `frontend/nginx.conf` + `backend/frontend.Dockerfile` — static build
  served by nginx with `/api` proxied to the backend service.
- `docker-compose.yml` — one command stack with healthcheck + persistent
  volumes (DB + uploads separated from app code).
- Verified end-to-end in Docker: `docker compose up --build` → backend
  healthy, migrations applied, model served, real inspection + overlay
  through the nginx proxy. Frontend at `http://localhost:8090`, API at
  `http://localhost:8000`.
- GPU: training supports CUDA automatically (`torch.cuda.is_available()`);
  install the CUDA torch wheel instead of the CPU one and run the same
  scripts. The default path documented here is CPU (the tested path).

### Single-container cloud deployment (torch-free ONNX serving)

`deploy/` contains a self-contained deployment: React build + **ONNX**
model + API served from **one origin** (SPA mode via `VI_FRONTEND_DIST`,
unit-tested in `backend/tests/test_spa_mode.py`).

Two serving backends, switchable via `VI_INFERENCE_BACKEND`:
- `torch` — training-grade backend (development default)
- `onnx` — torch-free serving: `ml/export_onnx.py` exports the ResNet18 with
  (logits, layer4 activations) and verifies **bit-identical** predictions and
  **correlation-1.0** Grad-CAM heatmaps; the serving image needs no torch
  (~159MB RSS vs ~550MB, verified with `import torch` blocked)

```bash
# export the serving model (after training) and verify parity:
python ml/export_onnx.py

# always-on free cloud via Render (needs the GitHub repo pushed):
#   render.com -> New Blueprint -> pick the repo -> Apply (render.yaml does the rest)

# single-container on any Docker host:
deploy/build_deploy.sh        # or deploy/build_deploy.ps1 on Windows
docker run -p 7860:7860 surfacespec   # full UI + API on http://localhost:7860
```

Deployment tiers & rationale: [`deploy/README.md`](deploy/README.md)

## 17. Setup

Prerequisites: Python 3.12+, Node 20+, ~600 MB disk for the venv.

```bash
git clone <repo>
cd ai-visual-inspection
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt -r ml/requirements.txt pyarrow==18.1.0

copy .env.example .env      # (Linux: cp)
npm install --prefix frontend
```

## 18. Usage

```powershell
# 1) Train + evaluate (downloads dataset automatically, seed-reproducible)
scripts\train.ps1                     # or: bash scripts/train.sh

# 2) Run backend + frontend dev servers
scripts\dev.ps1                      # backend :8000 (docs at /api/docs), frontend :5173

# Docker alternative (requires trained artifacts in ./models):
docker compose up --build            # frontend :8090, backend :8000
```

## 19. Testing

```bash
# 78 tests: validation, severity rules, confidence tiers, DB constraints,
# API integration (upload/feedback/analytics/security), ML pipeline,
# Grad-CAM sanity, SPA mode, ONNX backend parity (vs torch)
cd backend
PYTHONPATH=..;..\\backend python -m pytest tests -q     # PowerShell syntax shown in scripts
```

Covered: image validation (all rejection paths), path traversal,
duplicate feedback, cascade deletes, unique constraints, prediction
formatting, tiering boundaries, severity band edges, analytics math
(correction rate from real rows).

## 20. Limitations

- Dataset contains **only defective** surfaces → the platform classifies
  defect type; it cannot declare a surface defect-free (no OK class).
- Softmax confidence is not calibrated probability (documented policy, not
  a guarantee).
- Grad-CAM localization is approximate evidence, not detection; no
  localization metrics are claimed (no ground-truth boxes in this split).
- Single-domain data (one hot-rolling line); transfer to other mills is
  untested.
- Concurrency: CPU inference serializes through one process; scale
  horizontally per worker if throughput matters.
- Feedback review/approval is a DB workflow + export script; no admin UI
  yet.

## 21. Future improvements

- Calibrated confidence (temperature scaling on a held-out set) and
  precision/recall curves per class in the UI
- Detection model trained on the annotated NEU-DET detection split (real
  boxes) with IoU/mAP reporting
- Active-learning queue: rank low-confidence + corrected samples for
  priority labeling
- Admin UI for feedback review/approval; one-click training-set rebuild
- PostgreSQL + S3-compatible storage backend; JWT auth for inspectors
- Model drift monitoring (confidence distribution shifts across versions)
- Drift/out-of-distribution rejection (e.g., Mahalanobis on features)

## License

MIT for the code. The dataset is used for educational purposes under its
upstream academic-use terms (not redistributed here).
