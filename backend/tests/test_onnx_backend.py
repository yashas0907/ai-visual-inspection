"""ONNX serving backend tests (skip when artifacts are absent).

Verifies the torch-free backend is a drop-in replacement:
- identical class predictions vs the torch predictor on real test images
- heatmap correlation ~1.0 vs torch Grad-CAM (the CAM equivalence)
- same Prediction/ExplainabilityResult contracts the API depends on
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1].parent))  # repo root for ml package

from ml.config import DataConfig, TrainConfig  # noqa: E402
from ml.explainability.gradcam import GradCAM  # noqa: E402
from ml.inference.onnx_predictor import OnnxDefectPredictor, OnnxGradCAM  # noqa: E402
from ml.inference.predictor import DefectPredictor  # noqa: E402

REPO = HERE.parents[1]  # backend/tests -> backend -> ai-visual-inspection (repo root)
ONNX_PATH = REPO / "models" / "onnx" / "resnet18_neu.onnx"
TORCH_PATH = REPO / "models" / "experiments" / "v1_resnet18" / "best.pth"
TEST_DIR = REPO / "data" / "processed" / "test"

pytestmark = pytest.mark.skipif(
    not ONNX_PATH.exists() or not TORCH_PATH.exists(),
    reason="ONNX/torch artifacts not exported (run ml/export_onnx.py + training)",
)


@pytest.fixture(scope="module")
def torch_predictor():
    return DefectPredictor(TORCH_PATH, DataConfig(), TrainConfig(), "v-test")


@pytest.fixture(scope="module")
def onnx_predictor():
    return OnnxDefectPredictor(ONNX_PATH, "v-test")


def _sample_images(n: int = 3) -> list[Path]:
    imgs = sorted(TEST_DIR.rglob("*.jpg"))
    assert imgs, "test split missing — run ml/data/prepare.py"
    step = max(1, len(imgs) // n)
    return imgs[::step][:n]


class TestOnnxPredictor:
    def test_prediction_format(self, onnx_predictor):
        p = onnx_predictor.predict_pil(Image.new("L", (200, 200), 128))
        assert p.label in DataConfig.classes
        assert 0 <= p.label_index < 6
        assert 0.0 <= p.confidence <= 1.0
        assert abs(sum(p.probabilities.values()) - 1.0) < 0.01
        assert p.model_version == "v-test"
        assert p.latency_ms >= 0

    def test_identical_argmax_vs_torch(self, torch_predictor, onnx_predictor):
        for img_path in _sample_images():
            img = Image.open(img_path).convert("L")
            t = torch_predictor.predict_pil(img)
            o = onnx_predictor.predict_pil(img)
            assert t.label == o.label, f"{img_path.name}: torch={t.label} onnx={o.label}"

    def test_probabilities_close(self, torch_predictor, onnx_predictor):
        img = Image.open(_sample_images(1)[0]).convert("L")
        t = torch_predictor.predict_pil(img)
        o = onnx_predictor.predict_pil(img)
        for cls in DataConfig.classes:
            assert abs(t.probabilities[cls] - o.probabilities[cls]) < 0.01

    def test_stats_tracking(self, onnx_predictor):
        before = onnx_predictor.stats["inferences"]
        onnx_predictor.predict_pil(Image.new("L", (200, 200), 90))
        assert onnx_predictor.stats["inferences"] == before + 1


class TestOnnxExplainability:
    def test_heatmap_shape_and_range(self, onnx_predictor):
        result, idx, conf, probs = OnnxGradCAM(onnx_predictor).generate(Image.new("L", (200, 200), 140))
        assert result.heatmap.shape == (224, 224)
        assert result.heatmap.min() >= 0.0 and result.heatmap.max() <= 1.0
        assert 0 <= idx < 6
        assert len(probs) == 6
        assert 0.0 <= result.region_area_ratio <= 1.0

    def test_cam_equivalence_vs_torch(self, torch_predictor, onnx_predictor):
        """The exported ONNX CAM must equal torch Grad-CAM (corr >= 0.99)."""
        cam_torch = GradCAM(torch_predictor)
        cam_onnx = OnnxGradCAM(onnx_predictor)
        for img_path in _sample_images(2):
            img = Image.open(img_path).convert("L")
            r_torch, *_ = cam_torch.generate(img)
            r_onnx, *_ = cam_onnx.generate(img)
            corr = float(np.corrcoef(r_torch.heatmap.flatten(), r_onnx.heatmap.flatten())[0, 1])
            assert corr > 0.99, f"heatmap divergence on {img_path.name}: corr={corr}"
            assert r_onnx.label == r_torch.label

    def test_no_torch_import_required(self):
        """The onnx module must be importable in a torch-free image: verify the
        module itself pulls no torch symbols at import time."""
        import ml.inference.onnx_predictor as mod
        import ml.inference.policy as policy_mod
        assert not any(
            getattr(mod, n, None).__class__.__module__.startswith("torch") for n in dir(mod)
        )
        assert not hasattr(policy_mod, "torch")
