"""Model lifecycle: load once at startup, register metadata, expose predictor.

The ML model is treated as an application dependency. It is loaded a single
time (lifespan) and shared across requests through a singleton accessor;
inference holds the GIL-bound forward pass, so FastAPI's threadpool handles
concurrent requests while the model itself runs serially — documented in
docs/architecture.md#concurrency.

Two interchangeable backends behind the same interface (VI_INFERENCE_BACKEND):
- "torch":  ml.inference.predictor.DefectPredictor + GradCAM (training env)
- "onnx":   ml.inference.onnx_predictor.OnnxDefectPredictor + OnnxGradCAM
            (torch-FREE: the serving image needs no torch at all — 4x less RAM,
             verified bit-identical outputs; see ml/export_onnx.py)

Imports of backend modules are lazy so a torch-free deployment image never
imports torch transitively.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_LOCK = threading.Lock()
_predictor = None
_gradcam = None


class ModelNotLoadedError(RuntimeError):
    pass


def load_model() -> None:
    """Load the configured backend + explainer. Idempotent; thread-safe."""
    global _predictor, _gradcam
    with _LOCK:
        if _predictor is not None:
            return
        settings = get_settings()
        ckpt = Path(settings.model_checkpoint)
        if not ckpt.exists():
            logger.warning("model artifact missing at %s; /api/inspections will 503", ckpt)
            raise FileNotFoundError(str(ckpt))

        if settings.inference_backend.lower() == "onnx":
            from ml.config import DataConfig
            from ml.inference.onnx_predictor import OnnxDefectPredictor, OnnxGradCAM

            _predictor = OnnxDefectPredictor(ckpt, settings.model_version, DataConfig())
            _gradcam = OnnxGradCAM(_predictor)
            logger.info("ONNX backend %s loaded from %s", settings.model_version, ckpt)
        else:
            from ml.config import DataConfig, TrainConfig
            from ml.explainability.gradcam import GradCAM
            from ml.inference.predictor import DefectPredictor

            _predictor = DefectPredictor(
                checkpoint_path=ckpt,
                data_cfg=DataConfig(),
                train_cfg=TrainConfig(),
                model_version=settings.model_version,
            )
            _gradcam = GradCAM(_predictor)
            logger.info("torch backend %s loaded from %s", settings.model_version, ckpt)


def get_predictor():
    if _predictor is None:
        raise ModelNotLoadedError("model not loaded")
    return _predictor


def get_gradcam():
    if _gradcam is None:
        raise ModelNotLoadedError("gradcam not initialized")
    return _gradcam


def model_ready() -> bool:
    return _predictor is not None


def get_confidence_tiers():
    from ml.inference.policy import ConfidenceTiers

    s = get_settings()
    return ConfidenceTiers(high=s.confidence_high, medium=s.confidence_medium)


def load_experiment_metrics() -> dict:
    """Read genuine test metrics produced by ml/evaluation (no invention)."""
    settings = get_settings()
    ckpt = Path(settings.model_checkpoint)
    if settings.inference_backend.lower() == "onnx":
        # models/onnx/x.onnx -> metrics live in models/experiments/<exp>/
        exp_dir = ckpt.parent.parent / "experiments" / settings.model_experiment
    else:
        # models/experiments/<exp>/best.pth -> metrics live alongside it
        exp_dir = ckpt.parent
    payload: dict = {}
    for key, filename in [
        ("test_metrics", "test_metrics.json"),
        ("training_metadata", "metadata.json"),
        ("training_history", "history.json"),
    ]:
        path = exp_dir / filename
        if path.exists():
            payload[key] = json.loads(path.read_text(encoding="utf-8"))
    return payload
