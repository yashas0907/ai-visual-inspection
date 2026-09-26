"""Boot the app with torch import BLOCKED — proves the ONNX serving path is
torch-free. If this boots and serves inspections, the cloud image (which has
no torch installed) works identically.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Block torch/torchvision entirely: any import attempt raises ImportError.
sys.modules["torch"] = None
sys.modules["torchvision"] = None

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

os.environ["VI_INFERENCE_BACKEND"] = "onnx"
os.environ["VI_MODEL_CHECKPOINT"] = str(ROOT / "models" / "onnx" / "resnet18_neu.onnx")
os.environ["VI_DATABASE_URL"] = f"sqlite:///{ROOT / 'backend' / 'inspection.db'}"
os.environ["VI_UPLOAD_DIR"] = str(ROOT / "backend" / "storage" / "uploads")

import uvicorn  # noqa: E402

from app.main import app  # noqa: E402  (must not import torch)

try:
    import torch  # noqa: F401  — sanity: must fail
    print("FATAL: torch import unexpectedly succeeded")
    sys.exit(3)
except ImportError:
    print("torch import blocked — serving ONNX backend torch-free")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
