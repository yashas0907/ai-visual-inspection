"""Export the trained ResNet18 to ONNX for torch-free serving.

Why two outputs: Grad-CAM on torchvision ResNet18 hooks the output of the
last residual stage (layer4). The hooked gradient d(score)/d(A) flows only
through avgpool -> fc (both linear/constant spatially), so the heatmap
reduces EXACTLY to CAM:  heatmap = ReLU( sum_k W_fc[c, k] * A^k )  /  max.
By exporting (logits, layer4 activations) together with the fc weights we
can compute identical explainability in the ONNX backend with a single
forward pass — verified numerically against the torch Grad-CAM (see
verify_onnx_parity below and docs/explainability.md).

Usage (inside the training env, after training + evaluation):
    python ml/export_onnx.py --experiment v1_resnet18
Outputs models/onnx/resnet18_neu.onnx (+fc_weights.npz, +export_meta.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ml.config import DataConfig, TrainConfig
from ml.training.model import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = REPO_ROOT / "models" / "onnx"


class ResNetWithFeatures(nn.Module):
    """ResNet18 wrapper exposing (logits, layer4 feature map) for ONNX."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.model
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        feats = m.layer4(x)  # (1, 512, 7, 7)
        pooled = m.avgpool(feats)
        pooled = torch.flatten(pooled, 1)
        logits = m.fc(pooled)
        return logits, feats


def export(experiment: str = "v1_resnet18") -> Path:
    exp_dir = REPO_ROOT / "models" / "experiments" / experiment
    ckpt_path = exp_dir / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} missing — train first")

    data_cfg = DataConfig()
    train_cfg = TrainConfig()
    model = build_model(train_cfg, num_classes=len(data_cfg.classes))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wrapper = ResNetWithFeatures(model).eval()
    dummy = torch.randn(1, 1, data_cfg.image_size, data_cfg.image_size)

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = ONNX_DIR / "resnet18_neu.onnx"
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits", "layer4_feats"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    # fc weights: needed by the CAM-equivalent explainability at serve time
    fc_w = model.fc.weight.detach().cpu().numpy()  # (num_classes, 512)
    np.savez(ONNX_DIR / "fc_weights.npz", fc_weights=fc_w, classes=np.array(list(data_cfg.classes)))

    meta = {
        "experiment": experiment,
        "source_checkpoint": str(ckpt_path),
        "best_epoch": ckpt.get("epoch"),
        "classes": list(data_cfg.classes),
        "image_size": data_cfg.image_size,
        "channels": 1,
        "normalization": {"mean": [0.5088], "std": [0.2095]},
        "outputs": ["logits", "layer4_feats"],
        "onnx_bytes": onnx_path.stat().st_size,
        "note": "heatmap = relu(fc_weights[class] @ layer4_feats) == torch Grad-CAM "
        "for this architecture (avgpool+fc path is linear); see docs/explainability.md",
    }
    (ONNX_DIR / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"exported {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")
    return onnx_path


def verify_onnx_parity(experiment: str = "v1_resnet18", n_images: int = 25, seed: int = 42) -> dict:
    """Numerical parity: ONNX vs torch on real test images.

    Checks argmax agreement, softmax deltas and CAM heatmap correlation
    against the live Grad-CAM implementation. Run in the training env.
    """
    import onnxruntime as ort
    from PIL import Image

    from ml.data.dataset import NEUDetDataset
    from ml.evaluation.evaluate import collect_predictions
    from ml.explainability.gradcam import GradCAM
    from ml.inference.predictor import DefectPredictor

    root = Path(__file__).resolve().parents[1]
    data_cfg, train_cfg = DataConfig(), TrainConfig()
    exp_dir = root / "models" / "experiments" / experiment

    predictor = DefectPredictor(exp_dir / "best.pth", data_cfg, train_cfg, "parity")
    cam = GradCAM(predictor)
    session = ort.InferenceSession(str(ONNX_DIR / "resnet18_neu.onnx"), providers=["CPUExecutionProvider"])
    fc_w = np.load(ONNX_DIR / "fc_weights.npz")["fc_weights"]

    import random

    rng = random.Random(seed)
    test_dir = root / "data" / "processed" / "test"
    images = sorted(test_dir.rglob("*.jpg"))
    sample = rng.sample(images, min(n_images, len(images)))

    n_match, n_total = 0, 0
    max_prob_delta = 0.0
    cam_corrs: list[float] = []
    from ml.data.dataset import MEAN, STD

    for img_path in sample:
        label_dir = img_path.parent.name
        img = Image.open(img_path).convert("L").resize((data_cfg.image_size, data_cfg.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - MEAN[0]) / STD[0]
        x = arr[None, None, :, :]  # (1,1,H,W)
        logits, feats = session.run(None, {"image": x})
        probs = torch.softmax(torch.from_numpy(logits), dim=1)[0].numpy()

        torch_pred = predictor.predict_pil(Image.open(img_path).convert("L"))
        # full-precision torch probs for the delta check
        with torch.no_grad():
            tlogits = predictor.forward(predictor.preprocess(Image.open(img_path).convert("L")))
        tprobs = torch.softmax(tlogits, dim=1)[0].numpy()

        n_match += int(probs.argmax() == tprobs.argmax())
        n_total += 1
        max_prob_delta = max(max_prob_delta, float(np.abs(probs - tprobs).max()))

        if len(cam_corrs) < 10:
            expl, idx, _, _ = cam.generate(Image.open(img_path).convert("L"))
            h_torch = expl.heatmap.flatten()
            # ONNX CAM: relu(fc_w[idx] @ feats) -> resize to input
            cam_map = np.maximum(
                np.einsum("khw,k->hw", feats[0], fc_w[idx]), 0.0
            )  # (7,7): sum_k w_k * A^k_ij
            cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
            cam_img = Image.fromarray((cam_map * 255).astype(np.uint8)).resize(
                (data_cfg.image_size, data_cfg.image_size), Image.BILINEAR
            )
            h_onnx = np.asarray(cam_img, dtype=np.float32).flatten() / 255.0
            corr = float(np.corrcoef(h_torch, h_onnx)[0, 1])
            cam_corrs.append(corr)
        _ = label_dir  # directory name is the ground truth label (unused here)

    report = {
        "argmax_match": f"{n_match}/{n_total}",
        "max_softmax_abs_delta": round(max_prob_delta, 5),
        "cam_correlation_min": round(min(cam_corrs), 4) if cam_corrs else None,
        "cam_correlation_mean": round(float(np.mean(cam_corrs)), 4) if cam_corrs else None,
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export + verify ONNX serving model")
    parser.add_argument("--experiment", default="v1_resnet18")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    export(args.experiment)
    if not args.skip_verify:
        verify_onnx_parity(args.experiment)
