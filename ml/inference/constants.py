"""Torch-free inference constants (single source of truth).

Shared by the torch backend (ml/data/dataset.py, ml/inference/predictor.py)
and the torch-free ONNX backend, so both preprocess identically.
Importing this module requires NO ML framework — only ml.config (stdlib+yaml)
and numpy.
"""
from __future__ import annotations

from ml.config import DataConfig

# Dataset channel stats in [0,1] space, computed 2026-09-02 over the 1080
# training images (see docs/dataset.md for the reproduction command).
MEAN = [0.5088]
STD = [0.2095]

CLASSES = list(DataConfig.classes)
NUM_CLASSES = len(CLASSES)
IMAGE_SIZE = 224  # training/inference resolution (200x200 originals upscaled)
