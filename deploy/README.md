# Deployment options (three tiers)

## 1. ALWAYS-ON free cloud (Render) — recommended, torch-free ONNX image

**Requirements:** GitHub repo pushed (the 45MB ONNX model is committed) +
free render.com account (no credit card).

**One-time setup:**
1. Push the project to GitHub (see docs/go_live_guide.md Part 2).
2. https://render.com → **Get Started** → sign up **with GitHub** (grant repo access).
3. Dashboard → **New +** → **Blueprint** → select your `ai-visual-inspection`
   repo → Render reads `render.yaml` → **Apply**.
4. Wait for the build (~10-15 min, free tier) → you get
   `https://surfacespec.onrender.com` — always on.

**Why it fits the free tier:** Render free = 512MB RAM. The torch CPU
runtime alone needs ~450MB. The `deploy/Dockerfile.cloud` image serves with
**ONNX Runtime instead of torch — no torch in the image at all — and runs at
~159MB RSS** (verified). Predictions and Grad-CAM explainability are
**bit-identical** (verified: argmax 25/25, softmax delta 0.0, heatmap
correlation 1.0; see `ml/export_onnx.py`).

**Free-tier behavior (know these):**
- Sleeps after ~15 min without traffic → first visitor waits ~30-60s while
  it wakes (later requests are fast)
- SQLite + uploads are ephemeral: history resets on each restart/redeploy
  (fine for a demo; the README says so)
- Every `git push` auto-redeploys the app

**The ONNX port — what to tell interviewers:** training uses torch (the
research stack), serving uses ONNX Runtime (the deployment stack) — a
standard industry pattern that cut serving memory ~4x and unlocked free
hosting. The Grad-CAM heatmap stays exact because for ResNet18 the hooked
gradient flows only through avgpool→fc, reducing Grad-CAM to CAM
(relu(fc_weights[class] @ layer4_features)) — and the export ships both the
activations and fc weights so one forward pass reproduces the identical
heatmap. The equivalence was verified numerically, not assumed.

## 2. On-demand tunnel from your laptop ($0, no accounts)

```powershell
scripts\share_demo.ps1     # starts app + prints a public https://... URL
```

Your laptop is the server while it runs. URL changes each launch.

## 3. Self-hosted Docker (compose, LAN/vps/GPU)

```bash
docker compose up --build   # frontend :8090, backend :8000 (torch image)
```

## Artifact map

| File | Purpose |
|---|---|
| `ml/export_onnx.py` | torch → ONNX export + numerical parity verification |
| `models/onnx/resnet18_neu.onnx` (+fc_weights.npz) | committed serving artifacts (45MB) |
| `deploy/Dockerfile.cloud` | torch-free serving image (Render-ready) |
| `render.yaml` | Render blueprint (free plan, health check, auto-deploy) |
| `deploy/Dockerfile` | single-container torch image (any host) |
| `deploy/push_to_space.py` | HF Spaces deploy (requires HF PRO since 2025 policy) |
| `deploy/torchfree_boot.py` | local proof: boots the app with `import torch` blocked |
