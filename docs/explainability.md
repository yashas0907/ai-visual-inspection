# Explainability (Grad-CAM)

## What it is

For each inspection the platform runs **Grad-CAM** (Selvaraju et al., ICCV
2017) during the same forward pass used for the prediction:

1. Forward the preprocessed image through the ResNet18.
2. Take the score of the predicted class and backpropagate to the output of
   the last residual stage (`layer4`, a 7×7×512 feature map for 224² input).
3. Weight each feature channel by the spatial mean of its gradient,
   sum, ReLU, normalize to [0, 1], and upsample to image resolution.

The resulting heatmap highlights regions whose activation most increased
the predicted class score. A coarse bounding box is derived by thresholding
the heatmap at the 90th percentile and taking the largest 4-connected
component; its area fraction ("evidence coverage") feeds the severity engine.

## Why Grad-CAM for this model

The dataset provides image-level labels only (no boxes/masks), so a
detection or segmentation model cannot be trained honestly from it. Grad-CAM
gives per-pixel evidence from a classifier — appropriate, cheap (one extra
backward pass), and well understood by CV engineers.

## Honest limitations (please read in interviews)

- **Grad-CAM is evidence, not proof.** It shows which regions the model
  *used*, not a certified defect location. It can highlight background
  texture that correlates with the class.
- **Receptive field blur.** ResNet18's layer-4 receptive field is large
  (~400+ px effective); heat spreads beyond the exact defect boundary.
- **Localization is approximate.** The bbox/coverage are heuristic summaries
  of the heatmap. They are labeled "approximate" in the UI. The dataset has
  no ground-truth boxes for this split, so no localization metric (IoU) is
  reported — claiming one would be fabrication.
- **Heatmaps are class-conditional.** They explain the *predicted* class
  score; a wrong prediction produces a confidently wrong-looking heatmap.
  This is why human feedback exists.
- **Not a causal explanation.** Ablating the highlighted region and
  re-inferring (the rigorous check) is future work; the current heatmap is
  gradient evidence only.

## Where to see it

- Per inspection: overlay image + bbox + coverage on the Result page.
- Code: `ml/explainability/gradcam.py` (tensor-hook implementation that
  avoids torchvision's full-backward-hook view/inplace limitation with
  ResNet residual adds — see module docstring).
- Tests: `backend/tests/test_ml_pipeline.py::TestGradCAM` (shape, range,
  prob-sum sanity checks).
