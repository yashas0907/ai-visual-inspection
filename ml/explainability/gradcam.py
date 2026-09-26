"""Grad-CAM explainability for the defect classifier (torch backend).

Implementation notes:
- Grad-CAM (Selvaraju et al., 2017) uses gradients of the predicted class
  score w.r.t. the last residual block group's output feature map
  (resnet18.layer4 output). Each feature-map channel is weighted by the
  spatial mean of its gradient, then ReLU'd and normalized. The activation
  gradient is captured with a tensor hook (forward hook on the module keeps
  a detached reference and registers a gradient callback on the live graph
  tensor), which sidesteps torchvision's full-backward-hook view/inplace
  limitation with ResNet's residual additions.
- Architecture fact (verified numerically in ml/export_onnx.py): because
  layer4's output feeds avgpool -> fc only, the hooked gradient
  d(score)/d(A) is spatially constant (= W_fc[class] / 49), so this Grad-CAM
  reduces EXACTLY to CAM: heatmap = relu(sum_k W_fc[c,k] * A^k). This
  equivalence is what lets the torch-free ONNX backend reproduce identical
  heatmaps (correlation 1.0 on verification samples).
- The output is an evidence heatmap, NOT a causal proof. It highlights the
  spatial regions that most increased the predicted class score, subject to
  the architecture's receptive field, so nearby background can light up too.
  This limitation is documented in docs/explainability.md.
- ``localized region`` derives a coarse bounding box from the heatmap via
  percentile thresholding + largest connected component. It is a localization
  *hint* for inspectors, not an object-detection output (the dataset provides
  no bounding boxes for this split).
"""
from __future__ import annotations

from PIL import Image

import numpy as np
import torch

from ml.explainability.localization import (  # re-exported for compatibility
    ExplainabilityResult,
    largest_component_bbox,
)
from ml.inference.predictor import DefectPredictor

__all__ = ["ExplainabilityResult", "GradCAM", "largest_component_bbox"]


class GradCAM:
    def __init__(self, predictor: DefectPredictor):
        self.predictor = predictor
        self.model = predictor.model
        self.device = predictor.device

    def generate(self, image: Image.Image) -> tuple[ExplainabilityResult, int, float, dict[str, float]]:
        """Run one forward+backward pass and return (explainability, class_idx, confidence, probs)."""
        from ml.explainability.localization import summarize_heatmap

        x = self.predictor.preprocess(image).to(self.device)

        captured: dict[str, torch.Tensor] = {}

        # Standard tensor-hook Grad-CAM: keep the module output (a live graph
        # tensor) and register a gradient callback on it. Avoids torchvision's
        # full-backward-hook, which is incompatible with ResNet's in-place
        # residual adds.
        def fwd_hook(_m, _inp, out):
            captured["act"] = out.detach()
            if out.requires_grad:
                out.register_hook(lambda grad: captured.__setitem__("grad", grad.detach()))

        target_layer = self.model.layer4
        handle = target_layer.register_forward_hook(fwd_hook)
        try:
            logits = self.model(x)
            idx = int(logits.argmax(dim=1).item())
            score = logits[0, idx]
            self.model.zero_grad(set_to_none=True)
            score.backward()
        finally:
            handle.remove()

        act = captured["act"]  # (1, C, h, w) detached copy
        grad = captured.get("grad")
        if grad is None:
            raise RuntimeError("gradient capture failed for Grad-CAM")
        weights = grad.mean(dim=(2, 3), keepdim=True)  # channel importance
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))[0, 0].detach()  # (h, w)
        cam = cam.cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Upsample to input resolution.
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (x.shape[-1], x.shape[-2]), Image.BILINEAR
        )
        heatmap = np.asarray(cam_img, dtype=np.float32) / 255.0

        label, idx2, conf, probs = self.predictor.postprocess(logits)
        assert idx == idx2  # same forward pass
        return summarize_heatmap(heatmap, label), idx, conf, probs
