"""Compute dataset channel statistics used by normalization.

Two-pass mean/std over all pixels of every image in a dataloader
(without augmentation). The values produced here are copied into
``ml/data/dataset.py`` (MEAN/STD) and recorded in the experiment metadata.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


def compute_stats_two_pass(loader: DataLoader) -> dict[str, list[float]]:
    total = 0
    s = torch.zeros(1, dtype=torch.float64)
    s2 = torch.zeros(1, dtype=torch.float64)
    for imgs, _ in loader:
        x = imgs.flatten().to(torch.float64)
        total += x.numel()
        s += x.sum()
        s2 += (x * x).sum()
    mean = (s / total).item()
    var = (s2 / total - mean * mean).item()
    return {"mean": [round(mean, 4)], "std": [round(float(np.sqrt(max(var, 0.0))), 4)]}
