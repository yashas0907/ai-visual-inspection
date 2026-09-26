"""ML pipeline unit tests: preprocessing, Grad-CAM, prediction formatting,
severity integration through the real inference path (small random model)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1].parent))  # repo root for ml package

from ml.config import DataConfig, TrainConfig  # noqa: E402
from ml.data.dataset import MEAN, STD, build_transforms  # noqa: E402
from ml.explainability.gradcam import GradCAM  # noqa: E402
from ml.inference.predictor import (  # noqa: E402
    ConfidenceTiers,
    DefectPredictor,
    ImageValidationError,
    confidence_tier,
)
from ml.training.model import build_model  # noqa: E402


@pytest.fixture(scope="module")
def tiny_predictor(tmp_path_factory):
    ckpt_dir = tmp_path_factory.mktemp("model")
    model = build_model(TrainConfig(pretrained=False), num_classes=len(DataConfig.classes))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": list(DataConfig.classes),
            "arch": "resnet18",
            "epoch": 0,
        },
        ckpt_dir / "best.pth",
    )
    return DefectPredictor(
        checkpoint_path=ckpt_dir / "best.pth",
        data_cfg=DataConfig(),
        train_cfg=TrainConfig(pretrained=False),
        model_version="v-test",
    )


class TestPreprocessing:
    def test_transform_shapes(self):
        img = Image.new("L", (200, 200), 120)
        t = build_transforms(DataConfig(), train=False)
        x = t(img)
        assert x.shape == (1, 224, 224)
        assert x.dtype == torch.float32

    def test_normalization_statistics(self):
        # an image filled with the dataset mean should normalize to ~0
        mean_val = int(round(MEAN[0] * 255))
        img = Image.new("L", (200, 200), mean_val)
        x = build_transforms(DataConfig(), train=False)(img)
        assert abs(float(x.mean())) < 0.02

    def test_std_reasonable(self):
        assert 0.05 < STD[0] < 0.5


class TestPredictor:
    def test_prediction_format(self, tiny_predictor):
        img = Image.new("L", (200, 200), 128)
        p = tiny_predictor.predict_pil(img)
        assert p.label in DataConfig.classes
        assert 0 <= p.label_index < 6
        assert 0.0 <= p.confidence <= 1.0
        assert abs(sum(p.probabilities.values()) - 1.0) < 0.01
        assert p.model_version == "v-test"
        assert p.latency_ms >= 0

    def test_invalid_file_rejected(self, tmp_path, tiny_predictor):
        bad = tmp_path / "bad.gif"
        bad.write_bytes(b"x" * 10)
        with pytest.raises(ImageValidationError):
            tiny_predictor.predict_path(bad)

    def test_missing_file_rejected(self, tmp_path, tiny_predictor):
        with pytest.raises(ImageValidationError):
            tiny_predictor.predict_path(tmp_path / "nope.jpg")

    def test_stats_tracking(self, tiny_predictor):
        before = tiny_predictor.stats["inferences"]
        tiny_predictor.predict_pil(Image.new("L", (200, 200), 90))
        assert tiny_predictor.stats["inferences"] == before + 1


class TestGradCAM:
    def test_heatmap_properties(self, tiny_predictor):
        cam = GradCAM(tiny_predictor)
        img = Image.new("L", (200, 200), 140)
        result, idx, conf, probs = cam.generate(img)
        assert result.heatmap.shape == (224, 224)
        assert 0.0 <= result.heatmap.min() and result.heatmap.max() <= 1.0
        assert 0 <= idx < 6
        assert 0 <= conf <= 1
        assert len(probs) == 6
        assert 0.0 <= result.region_area_ratio <= 1.0


class TestTiering:
    def test_tiers_documented_defaults(self):
        t = ConfidenceTiers()
        assert t.high == 0.80 and t.medium == 0.60
        assert confidence_tier(0.85, t) == "high"
        assert confidence_tier(0.65, t) == "medium"
        assert confidence_tier(0.35, t) == "low"
