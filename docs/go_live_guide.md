# Go-Live Guide (Beginner Friendly)

Two steps, in this order:

1. **PART 1 — GitHub** (the portfolio repo; everything else deploys from it)
2. **PART 2 — Render** (always-on free hosting, auto-deploys from GitHub)

Do them once; after that every `git push` updates your live site automatically.

---

# PART 1 — GitHub (the portfolio repo)

### Step 1 — Account
https://github.com → **Sign up** (if you don't have one) → verify email.

### Step 2 — Create an EMPTY repository
1. Top-right **+** → **New repository**
2. Repository name: `ai-visual-inspection`
3. Description: `AI Visual Inspection & Defect Detection Platform — PyTorch + FastAPI + React + ONNX + Docker`
4. Select **Public**
5. ⚠️ Do NOT tick **Add a README**, **Add .gitignore**, or **Choose a license**
   (the project already has all three — ticking them causes push conflicts)
6. Click **Create repository**
7. Copy the URL shown at the top of the next page — it looks like:
   `https://github.com/YOURNAME/ai-visual-inspection.git`

### Step 3 — Tell git who you are (one time ever)
Open **PowerShell** and run (use the same email as your GitHub account):

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### Step 4 — Push the project
In the SAME PowerShell window:

```powershell
cd "C:\Users\phata\OneDrive\Documents\Default Project\ai-visual-inspection"
git init
git add .
git commit -m "AI Visual Inspection & Defect Detection Platform (PyTorch + FastAPI + React + ONNX)"
git branch -M main
git remote add origin https://github.com/YOURNAME/ai-visual-inspection.git
git push -u origin main
```

A browser window may pop up the first time → **Sign in with GitHub** → Authorize.

⚠️ IMPORTANT: **add `.env` to the push check** — the project's `.gitignore`
already excludes it automatically, plus the dataset, model checkpoints
(torch), your venv, the database, uploads and logs. What DOES get pushed
includes one 45MB file — `models/onnx/resnet18_neu.onnx` — that is
intentional: Render needs it to serve your model without retraining.

### Step 5 — Verify
1. Refresh the repo page → all folders visible (`backend/`, `ml/`,
   `frontend/`, `docs/`, `deploy/`, ...)
2. Click the **Actions** tab → a workflow called **CI** starts → wait for a
   green ✅ (runs 78 tests + frontend build, ~5-10 min).
   Green CI on a portfolio repo = instant credibility.
3. Settings → verify no `.env` file appears in the repo (it shouldn't).

---

# PART 2 — Render (always-on free hosting)

### Why it works when others don't
Free hosting gives 512MB RAM; the torch ML library alone needs ~450MB.
This project ships a **torch-free ONNX serving mode** (~159MB measured) with
predictions and Grad-CAM heatmaps **verified identical** to the torch
backend. `render.yaml` + `deploy/Dockerfile.cloud` encode all of this.

### Step 1 — Sign up (no credit card)
https://render.com → **Get Started** → **Sign up with GitHub** → authorize.

### Step 2 — Create the service
1. Dashboard → **New +** → **Blueprint**
2. Select your `ai-visual-inspection` repo → Render reads `render.yaml`
3. Click **Apply**
4. Wait for the build (~10-15 min first time)

### Step 3 — Open your live app
Render gives you: **https://surfacespec.onrender.com** — always on:

- Full UI + API + Swagger docs (`/api/docs`)
- Auto-redeploys on every future `git push`
- Health monitoring at `/api/health`

### Step 4 — Verify (60 seconds)
1. Open the URL → Dashboard loads
2. New Inspection → upload any image from `data\processed\test\`
3. See prediction + heatmap overlay + severity → submit feedback
4. Analytics shows your data; Model Info shows the real test metrics

### Free-tier behavior (expected, not bugs)
| Behavior | Explanation |
|---|---|
| First visit after idle is slow (~30-60s) | Free services sleep after ~15 min; they wake on the next visit |
| History occasionally resets | Free tier has no persistent disk (SQLite is ephemeral) — README documents this |
| Build takes ~10 min after each push | It's installing dependencies + building the frontend fresh |

---

# Bonus: instant demo tunnel from your laptop

For a live demo RIGHT NOW without any account (URL is temporary, laptop must
stay on):

```powershell
.\scripts\share_demo.ps1
```

---

# Troubleshooting

| Problem | Fix |
|---|---|
| `git push` says rejected / non-fast-forward | You ticked "Add README" on GitHub → delete the repo, recreate WITHOUT ticks, push again |
| GitHub login popup doesn't appear | Run `git config --global credential.helper manager`, then push again |
| Render build fails at `npm install` | Check the Actions tab on GitHub passed first; then paste me the Render build log |
| Render service says "unhealthy" | Open its **Logs** tab, search for `ERROR`, send it to me |
| Space stays "Build failed" on Render | The 45MB ONNX file wasn't pushed → Part 1 Step 5 |
