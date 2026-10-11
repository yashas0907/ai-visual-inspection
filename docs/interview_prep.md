# Interview Defense — SurfaceSpec (AI Visual Inspection Platform)

Every answer below is a fact of THIS build (verifiable in the repo), not
generic textbook talk. Read once before the interview; re-read the numbers
table the morning of.

---

## 1. Project & framing

**Q: What does the system do?**
Classifies surface defects on hot-rolled steel strip images (6 NEU classes),
shows *where* the model looked (Grad-CAM overlay), converts the prediction
into an operational decision (severity + action) via transparent rules, and
lets inspectors confirm/correct predictions — feeding a gated retraining
pipeline. Live: https://surfacespec.onrender.com

**Q: Why classification, not detection or segmentation?**
The accessible dataset split has image-level labels only — no boxes/masks.
A detector trained on invented boxes would be dishonest. A classifier +
evidence heatmap is what the data actually supports. (The annotated
detection variant of NEU-DET exists on IEEE DataPort — listed as future work.)

**Q: Why does this problem matter?**
Surface defects cause downstream rejects and claims in rolling mills;
human visual inspection is slow, fatiguing and inconsistent. The platform's
angle is *deployment discipline*: explainability, uncertainty handling and
human-in-the-loop — the parts that make factories actually trust CV.

## 2. Dataset (numbers to know cold)

| Fact | Value |
|---|---|
| Dataset | NEU Surface Defects Database (NEU-DET), Song & Yan 2013 |
| Images | 1,440 grayscale 200×200 — 6 classes × 240 (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches) |
| Splits | 1,080 / 216 / 144 (train/val/test), class-stratified, seed 42 |
| Normalization | mean 0.5088, std 0.2095 (computed from the training split) |
| Honest limitation | every image contains a defect — no "OK surface" class |

**Q: Where did the data come from and can I legally use it?**
Official channel is IEEE DataPort (account, no explicit redistribution
license). I used a public Hugging Face mirror, documented provenance in
`docs/dataset.md`, never committed the dataset, and the README's license
notice restricts derived weights to educational use.

**Q: Why do you say 100% test accuracy is not the headline?**
144 test images, balanced, from one mill — a curated benchmark. Production
distributions are imbalanced and messier. The metric is genuine but the
*caveat* is the insight.

## 3. Model & training

**Q: Why ResNet18?**
1,080 training images, low-res grayscale texture — capacity isn't the
bottleneck. Smaller model → less overfit risk and 24ms CPU inference, which
is a *product* feature, not a footnote.

**Q: Training recipe?**
ImageNet-initialized trunk; first conv re-initialized for 1-channel
grayscale; AdamW lr 3e-4, wd 1e-4, batch 32; 2-epoch linear warmup → cosine
decay; early stopping on val macro-F1 (patience 8) — stopped at epoch 20,
best checkpoint epoch 12, ~48 min CPU. Augmentation: flips, ±15° rotation,
±5% translate — deliberately NO photometric distortion (it can destroy
texture-based defect signatures).

**Q: What metrics did you produce?**
accuracy / macro precision / recall / F1 = 1.0; macro ROC-AUC (OVR) = 1.0;
perfect 144/144 confusion matrix; 24.2 ms/image CPU latency; 42.6MB fp32.
All from `ml/evaluation/evaluate.py` on the untouched test split.

**Q: Your ROC-AUC is hand-implemented — why should I trust it?**
I cross-validated it against sklearn on random, perfect, and tied-score
cases — identical to 4 decimals. (True story: the first version had a
descending-vs-ascending rank bug that returned 0.0 on perfect predictions;
that's *why* I added the cross-check.)

## 4. Explainability

**Q: How does your Grad-CAM work?**
Forward hook keeps a detached copy of layer4's output and registers a
tensor-hook on the live graph tensor (torchvision's full-backward-hook is
incompatible with ResNet's in-place residual adds — real RuntimeError I
hit and fixed). Channel weights = spatial mean of d(score)/d(activation);
weighted sum → ReLU → normalize → upsample; bbox = 90th-percentile threshold
+ largest connected component; coverage feeds the severity engine.

**Q: Limitations?**
Evidence, not proof. Receptive-field blur spreads heat; heatmaps are
class-conditional (a wrong prediction shows a confidently wrong map);
no ground-truth boxes → I claim **no localization metrics** — inventing an
IoU would be fabrication.

**Q: Why did Grad-CAM survive the ONNX port exactly?**
For this architecture, layer4's output feeds only avgpool → fc, both linear.
So d(score)/d(activation) is spatially constant (= fc weight row / 49) and
Grad-CAM reduces *exactly* to CAM: `relu(Σ_k W_fc[c,k] · A^k)`. I export
(logits, activations) + fc weights and compute that in serving — verified:
argmax 25/25, softmax delta 0.0, heatmap correlation 1.0.

## 5. Serving architecture (the strongest story)

**Q: Why ONNX Runtime in production?**
The torch CPU runtime alone is ~450MB RAM and overflows free hosting
(512MB). Serving with onnxruntime (no torch in the image at all) runs the
*whole platform* at ~159MB — measured, and proven by booting the app with
`import torch` blocked and passing all 43 end-to-end checks. Training keeps
torch; serving gets ONNX — the standard research-vs-deployment split.

**Q: Frontend + backend on one URL — how?**
Single container: FastAPI serves the built React bundle from the same origin
(`VI_FRONTEND_DIST`), so no CORS, no second container. That's also why it
fits free plans (one web service).

**Q: Why not Vercel?**
Vercel is static + short-lived serverless functions; this app needs a
persistent Python process (onnxruntime) and a database file. I deploy
containers to container hosts (Render/Railway/Cloud Run). Choosing hosting
to match architecture is the lesson.

**Q: Tell me about a production bug you debugged.**
Live site returned 500 only for >255-character path components, only on
Linux. Windows tolerates them, so local tests never caught it. Root cause:
`Path.is_file()` propagates non-ENOENT `OSError`s — on Linux that's
`ENAMETOOLONG` — unhandled → my 500 envelope. Fix: treat any stat failure as
404; wrote a regression test that monkeypatches `is_file` to raise
ENAMETOOLONG so Windows CI reproduces the Linux failure forever.

## 6. Policy layers

**Q: Severity — where does it come from?**
A pure, unit-tested function (`backend/app/services/severity.py`), not the
model and not an LLM: defect risk (patches 1 · scratches/rolled-in_scale 2 ·
crazing/pitted/inclusion 3) + confidence tier (high +0, medium +1, low +2)
+ evidence coverage (≤10% +0, 10–30% +1, >30% +2) → ≤3 minor / 4–5 major /
≥6 critical, each with a recommended action. The full rubric is exposed in
the UI so the decision is auditable.

**Q: Why does LOW confidence RAISE severity?**
Conservative direction: uncertain evidence still demands attention;
under-reacting costs more in steel QC than a manual check.

**Q: Is your 0.93 confidence a 93% probability?**
No — softmax output, not calibrated. That's why it drives a *policy*
(high/medium/low tiers, `<0.60` flagged for review) instead of pretending to
be certainty. Temperature-scaling calibration is listed as future work.

**Q: How does feedback become training data?**
It doesn't, automatically — by design. Feedback rows are immutable,
one-per-inspection, review-status gated (`pending/approved/rejected`);
`scripts/export_feedback.py` emits approved rows as a manifest a human
maintainer curates into the next run. Auto-retraining from user input
invites feedback loops and adversarial labels.

## 7. Engineering

**Q: Upload security?**
Size cap → extension allow-list → magic-byte sniff → extension/format-family
agreement → PIL `verify()` → decode → re-encode to canonical grayscale JPEG
(defangs polyglots) → dimension cap (decompression bombs). Stored filenames
are server-generated (sha+uuid); originals are display metadata only → path
traversal impossible by construction; the serving route re-verifies strict
path containment.

**Q: Why are images on the filesystem, not BLOBs?**
DB rows stay small, backups sane, serving cacheable; the DB stores relative
paths. BLOBs would bloat every analytics query for zero benefit.

**Q: Concurrency?**
Model is a startup-loaded singleton; torch/ORT forward passes serialize
(GIL) while FastAPI's threadpool keeps I/O concurrent. Scale-out = more
worker processes. Honest limit: single-instance SQLite.

**Q: What breaks at 10× scale?**
SQLite → Postgres (one env var), uploads → object storage (S3-compatible),
inference → a queue + worker pool, CDN for the SPA. None of that is needed
for the current product stage — knowing *when* to add it is the skill.

**Q: Testing philosophy?**
80 automated tests (validation paths, severity band edges, tier boundaries,
DB constraints/cascades, API integration, SPA mode, ONNX↔torch parity) plus
a separate 43-check *user-POV* script that exercises the running system the
way a person does — including abuse cases — and was run against localhost,
Docker and the live deployment.

## 8. Numbers cheat-sheet

| What | Value |
|---|---|
| Images / classes | 1,440 / 6×240 |
| Splits | 1080 / 216 / 144 |
| Norm stats | 0.5088 / 0.2095 |
| Training | 20 epochs (best 12), early stop, ~48 min CPU |
| Test metrics | acc / P / R / F1 / AUC = 1.0 (144 imgs) |
| Torch latency | 24.2 ms/img CPU |
| ONNX serving RAM | ~159 MB whole app (vs ~550MB torch) |
| ONNX parity | argmax 25/25 · Δsoftmax 0.0 · CAM corr 1.0 |
| Live inspection | ~4s end-to-end on free-tier CPU |
| Tests | 80 automated + 43 user-POV checks (all green) |
| Stack | PyTorch · ONNX Runtime · FastAPI · React/TS · SQLAlchemy/Alembic · Docker · Render |
