"""Model inference: load a checkpoint once, serve predictions (torch backend).

The predictor owns the whole inference pipeline:
validation -> preprocessing -> forward pass -> postprocessing ->
confidence tiering. It is deliberately framework-free (pure torch) so the
FastAPI service can wrap it without import cycles.

Serving policy (tiering) and image validation live in ml/inference/policy.py
and are re-exported here for backward compatibility. A torch-free ONNX
backend (ml/inference/onnx_predictor.py) reuses the same policy module.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from PIL import Image

from ml.config import DataConfig, TrainConfig
from ml.data.dataset import build_transforms
from ml.training.model import build_model
from ml.inference.policy import (  # re-exported for compatibility
    IMAGE_EXTENSIONS,
    ConfidenceTiers,
    ImageValidationError,
    confidence_tier,
    load_pil_image,
    validate_image_file,
)

logger = logging.getLogger(__name__)

__all__ = [
    "IMAGE_EXTENSIONS",
    "ConfidenceTiers",
    "ImageValidationError",
    "confidence_tier",
    "validate_image_file",
    "load_pil_image",
    "Prediction",
    "DefectPredictor",
]


@dataclass(frozen=True)
class Prediction:
    label: str
    label_index: int
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    model_version: str


class DefectPredictor:
    """Holds the trained model; ``predict`` runs the full pipeline."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        data_cfg: DataConfig,
        train_cfg: TrainConfig,
        model_version: str,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cpu")
        self.model_version = model_version
        self.classes: Sequence[str] = list(data_cfg.classes)
        self.data_cfg = data_cfg
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model: nn.Module = build_model(train_cfg, num_classes=len(self.classes))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.transform = build_transforms(data_cfg, train=False)
        # Failure hook for monitoring: count inference errors without crashing.
        self.stats = {"inferences": 0, "errors": 0, "total_latency_ms": 0.0}

    # -- core ---------------------------------------------------------------

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image).unsqueeze(0)  # (1, C, H, W)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device))

    def postprocess(self, logits: torch.Tensor) -> tuple[str, int, float, dict[str, float]]:
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax().item())
        conf = float(probs[idx].item())
        prob_dict = {c: round(float(p), 4) for c, p in zip(self.classes, probs.tolist())}
        return self.classes[idx], idx, conf, prob_dict

    def predict_pil(self, image: Image.Image) -> Prediction:
        t0 = time.perf_counter()
        x = self.preprocess(image)
        logits = self.forward(x)
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

    def predict_path(self, path: str | Path) -> Prediction:
        image = load_pil_image(Path(path))
        return self.predict_pil(image)
