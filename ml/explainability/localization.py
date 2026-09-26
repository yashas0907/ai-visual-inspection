"""Torch-free explainability post-processing: heatmap -> region summary.

Shared by the torch Grad-CAM backend and the ONNX (CAM-equivalent) backend:
thresholding, largest-connected-component bbox and evidence coverage are
pure numpy. ExplainabilityResult is defined here so the torch-free serving
image never imports torch (ml/explainability/gradcam.py re-exports it).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExplainabilityResult:
    label: str
    heatmap: np.ndarray  # float32 in [0,1], shape (H, W) = model input size
    heatmap_percentile_region: np.ndarray  # boolean mask of the activated region
    bbox: tuple[int, int, int, int] | None  # (x, y, w, h) in heatmap pixel coords
    region_area_ratio: float  # bbox area / total area


def summarize_heatmap(heatmap: np.ndarray, label: str, percentile: float = 90.0) -> ExplainabilityResult:
    """Derive the mask/bbox/coverage summary from a normalized heatmap."""
    thresh = np.percentile(heatmap, percentile)
    mask = heatmap >= thresh
    bbox, area_ratio = largest_component_bbox(mask)
    return ExplainabilityResult(
        label=label,
        heatmap=heatmap,
        heatmap_percentile_region=mask,
        bbox=bbox,
        region_area_ratio=round(area_ratio, 4),
    )


def largest_component_bbox(mask: np.ndarray) -> tuple[tuple[int, int, int, int] | None, float]:
    """Largest 4-connected component bbox via BFS (cv2 dependency avoided)."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        min_x = max_x = x0
        min_y = max_y = y0
        area = 0
        while stack:
            y, x = stack.pop()
            area += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if area > best_area:
            best_area = area
            best = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
    if best is None:
        return None, 0.0
    return best, (best[2] * best[3]) / mask.size
