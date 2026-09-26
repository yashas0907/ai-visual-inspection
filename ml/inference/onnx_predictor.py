"""Torch-free ONNX serving backend (deployment path).

Imports ONLY numpy/PIL/onnxruntime — never torch — so the cloud deployment
image stays ~200MB instead of ~2GB and fits free hosting RAM tiers.

Math: the exported graph returns (logits, layer4 activations) in one
forward pass; the heatmap is computed as relu(sum_k W_fc[c,k] * A^k) —
EXACTLY equal to the torch Grad-CAM for this architecture (verified:
argmax 25/25, softmax delta 0.0, heatmap correlation 1.0; see
ml/export_onnx.py::verify_onnx_parity and docs/explainability.md).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from ml.config import DataConfig
from ml.explainability.localization import ExplainabilityResult, summarize_heatmap
from ml.inference.constants import IMAGE_SIZE, MEAN, STD
from ml.inference.policy import Prediction, load_pil_image  # noqa: F401 (re-export convenience)


class OnnxDefectPredictor:
    """Same interface as ml.inference.predictor.DefectPredictor (torch-free)."""

    def __init__(self, onnx_path: str | Path, model_version: str, data_cfg: DataConfig | None = None):
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(str(onnx_path))
        data_cfg = data_cfg or DataConfig()

        self.model_version = model_version
        self.classes = list(data_cfg.classes)
        self.image_size = IMAGE_SIZE
        self.stats = {"inferences": 0, "errors": 0, "total_latency_ms": 0.0}

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        fc_npz = onnx_path.parent / "fc_weights.npz"
        if not fc_npz.exists():
            raise FileNotFoundError(f"{fc_npz} missing (export with ml/export_onnx.py)")
        payload = np.load(fc_npz, allow_pickle=True)
        self.fc_weights = payload["fc_weights"].astype(np.float32)  # (C, 512)
        stored_classes = [str(c) for c in payload["classes"].tolist()]
        if stored_classes != self.classes:
            raise ValueError(f"class mismatch: onnx {stored_classes} vs config {self.classes}")

    # -- preprocessing (identical to the torch path) -----------------------

    def preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.convert("L").resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - MEAN[0]) / STD[0]
        return arr[None, None, :, :]  # (1, 1, H, W)

    # -- inference -----------------------------------------------------------

    def run(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, feats = self.session.run(None, {"image": x})
        return logits[0], feats[0]  # (C,), (512, 7, 7)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def postprocess(self, logits: np.ndarray) -> tuple[str, int, float, dict[str, float]]:
        probs = self._softmax(logits.astype(np.float64))
        idx = int(probs.argmax())
        conf = float(probs[idx])
        prob_dict = {c: round(float(p), 4) for c, p in zip(self.classes, probs)}
        return self.classes[idx], idx, conf, prob_dict

    def predict_pil(self, image: Image.Image):
        t0 = time.perf_counter()
        x = self.preprocess(image)
        logits, _ = self.run(x)
        label, idx, conf, probs = self.postprocess(logits)
        latency = (time.perf_counter() - t0) * 1000
        self.stats["inferences"] += 1
        self.stats["total_latency_ms"] += latency
        return Prediction(
            label=label,
            label_index=idx,
            confidence=round(conf, 4),
            probabilities=probs,
            latency_ms=round(latency, 2),
            model_version=self.model_version,
        )

    def predict_path(self, path: str | Path):
        return self.predict_pil(load_pil_image(Path(path)))


class OnnxGradCAM:
    """Same interface as ml.explainability.gradcam.GradCAM (torch-free).

    One session call yields logits + layer4 activations; the CAM-equivalent
    heatmap uses the fc weight row of the predicted class.
    """

    def __init__(self, predictor: OnnxDefectPredictor):
        self.predictor = predictor

    def generate(self, image: Image.Image) -> tuple[ExplainabilityResult, int, float, dict[str, float]]:
        p = self.predictor
        x = p.preprocess(image)
        logits, feats = p.run(x)
        label, idx, conf, probs = p.postprocess(logits)

        cam = np.maximum(
            np.einsum("khw,k->hw", feats, p.fc_weights[idx]), 0.0
        )  # (7, 7): sum_k W_fc[idx,k] * A^k_ij — identical to torch Grad-CAM here
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (p.image_size, p.image_size), Image.BILINEAR
        )
        heatmap = np.asarray(cam_img, dtype=np.float32) / 255.0
        return summarize_heatmap(heatmap, label), idx, conf, probs


def load_export_meta(onnx_path: str | Path) -> dict:
    meta_path = Path(onnx_path).parent / "export_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))
