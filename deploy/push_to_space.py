"""One-command deployment to Hugging Face Spaces (needs HF PRO in 2025+).

HF policy change (2025+): ALL dynamic hosting (Gradio AND Docker Spaces)
requires a PRO subscription; free accounts get Static Spaces only. This
script therefore only works with a PRO token. For a free live demo use
scripts/share_demo.ps1 (cloudflared tunnel) instead, and for portfolio
hosting push the repo to GitHub.

Pipeline (when used with PRO): stage a lean folder (code + pre-built
frontend + 43MB model) -> upload via the Hub API (LFS handled
automatically, no local Docker needed) -> HF builds and runs it.

Usage (token is read from the environment — never hard-code it):

    # PowerShell
    $env:HF_TOKEN="hf_xxx"                     # Settings > Access Tokens (write)
    $env:HF_USERNAME="your-hf-username"
    python deploy/push_to_space.py --name surfacespec

Result: https://<username>-<name>.hf.space (first build ~5-10 min)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: SurfaceSpec - AI Visual Inspection
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: gradio
app_port: 7860
pinned: false
license: mit
short_description: AI defect detection for steel surfaces with explainability + human-in-the-loop
---

# SurfaceSpec — AI Visual Inspection & Defect Detection Platform

Upload a hot-rolled steel strip surface image → the model classifies one of
6 NEU defect types, shows **where** it looked (Grad-CAM evidence overlay),
derives operational **severity** via transparent business rules, and lets
you confirm or correct the prediction (human-in-the-loop).

- ResNet18 (ImageNet init, 1-channel stem) · test accuracy/F1/ROC-AUC = 1.0 on the 144-image holdout
- ~25 ms inference per image on CPU · full result in ~0.4 s
- FastAPI + React + SQLAlchemy · every prediction stamped with its model version

Try it: **New Inspection** → drag a steel surface image (any NEU defect
texture works) → review the prediction, Grad-CAM overlay and severity
breakdown → leave inspector feedback → watch Analytics and Model Info.

Note: free-tier storage is ephemeral (history resets when the Space
restarts). The full source, training pipeline and tests live on GitHub —
see the project README.
"""

# Installed by HF inside the Gradio base image (Python 3.10).
# torch CPU wheels come from the pytorch index via --extra-index-url.
SPACE_REQUIREMENTS = """\
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.5.1+cpu
torchvision==0.20.1+cpu
fastapi==0.115.5
uvicorn[standard]==0.32.1
python-multipart==0.0.17
sqlalchemy==2.0.36
alembic==1.14.0
pydantic==2.9.2
pydantic-settings==2.6.1
python-dotenv==1.0.1
numpy==1.26.4
pillow==11.0.0
"""

# Space entry point: env config + uvicorn(FastAPI) on the port HF probes.
SPACE_APP_PY = '''\
"""Space entry point: serves the SurfaceSpec FastAPI app (API + React SPA)
on port 7860. The Gradio SDK runtime simply runs this file with Python.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))          # ml package
sys.path.insert(0, str(ROOT / "backend"))  # app package

STATE = Path("/tmp/surfacespec")
STATE.mkdir(parents=True, exist_ok=True)
(STATE / "storage" / "uploads").mkdir(parents=True, exist_ok=True)
(STATE / "storage" / "results").mkdir(parents=True, exist_ok=True)

os.environ.setdefault("VI_DATABASE_URL", f"sqlite:///{STATE / 'inspection.db'}")
os.environ.setdefault("VI_UPLOAD_DIR", str(STATE / "storage" / "uploads"))
os.environ.setdefault("VI_MODEL_CHECKPOINT", str(ROOT / "models" / "experiments" / "v1_resnet18" / "best.pth"))
os.environ.setdefault("VI_MODEL_VERSION", "v1.0.0")
os.environ.setdefault("VI_MODEL_EXPERIMENT", "v1_resnet18")
os.environ.setdefault("VI_FRONTEND_DIST", str(ROOT / "frontend_dist"))

import uvicorn  # noqa: E402

from app.main import app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
'''


def _force_rmtree(path: Path, retries: int = 5, wait_s: float = 2.0) -> None:
    """Windows-safe recursive delete.

    OneDrive/AV often hold short locks on __pycache__ files inside the
    project folder, so: clear read-only bits, retry with backoff.
    """
    import stat
    import time

    def _onexc(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except PermissionError:
            pass  # retried on the next pass

    for attempt in range(retries):
        if not path.exists():
            return
        try:
            shutil.rmtree(path, onexc=_onexc)
            return
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(wait_s)
            else:
                raise


def stage(out_dir: Path) -> Path:
    """Assemble exactly what the Space runtime needs."""
    if out_dir.exists():
        _force_rmtree(out_dir)

    dist_src = REPO_ROOT / "frontend" / "dist"
    if not (dist_src / "index.html").is_file():
        sys.exit("ERROR: frontend/dist missing — run `npm run build` in frontend/ first.")

    exp = REPO_ROOT / "models/experiments/v1_resnet18"
    if not (exp / "best.pth").exists():
        sys.exit("ERROR: models/experiments/v1_resnet18/best.pth missing — run scripts/train first.")

    # (out_dir / "frontend").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "ml", out_dir / "ml",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO_ROOT / "backend/app", out_dir / "backend/app",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO_ROOT / "backend/alembic", out_dir / "backend/alembic",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(REPO_ROOT / "backend/alembic.ini", out_dir / "backend/alembic.ini")
    shutil.copytree(dist_src, out_dir / "frontend_dist")

    (out_dir / "models/experiments/v1_resnet18").mkdir(parents=True)
    for f in ["best.pth", "metadata.json", "history.json", "test_metrics.json", "confusion_matrix.png"]:
        if (exp / f).exists():
            shutil.copy(exp / f, out_dir / "models/experiments/v1_resnet18" / f)

    (out_dir / "app.py").write_text(SPACE_APP_PY, encoding="utf-8")
    (out_dir / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")
    (out_dir / "README.md").write_text(SPACE_README, encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy to a free HF Gradio-SDK Space")
    parser.add_argument("--name", default="surfacespec", help="Space name (lowercase, hyphens)")
    parser.add_argument("--stage-only", action="store_true", help="Only stage the deploy folder, don't upload")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    username = os.environ.get("HF_USERNAME")
    if not args.stage_only and (not token or not username):
        sys.exit("Set HF_TOKEN (write-access token) and HF_USERNAME env vars first.\n"
                 "Create a token at https://huggingface.co/settings/tokens")

    stage_dir = stage(REPO_ROOT / "deploy/_scratch")
    total_mb = sum(f.stat().st_size for f in stage_dir.rglob("*") if f.is_file()) / 1e6
    print(f"staged {total_mb:.1f} MB at {stage_dir}")

    if args.stage_only:
        print("--stage-only given; stopping here.")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_id = f"{username}/{args.name}"
    print(f"creating/ensuring space {repo_id} (gradio sdk, free cpu) ...")
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio",
                    private=False, exist_ok=True)

    print("uploading (LFS handled automatically; ~43MB upload) ...")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(stage_dir),
        commit_message="Deploy SurfaceSpec: FastAPI + React SPA + model v1.0.0",
    )
    url = f"https://{username}-{args.name}.hf.space"
    print("deployed! (first build takes ~5-10 min on free CPU)")
    print(f"  space: https://huggingface.co/spaces/{repo_id}")
    print(f"  app:   {url}")
    print(f"  build logs: https://huggingface.co/spaces/{repo_id}/logs")


if __name__ == "__main__":
    main()
