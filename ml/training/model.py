"""Model factory: ResNet18 adapted for 1-channel grayscale defect input.

Design notes:
- ResNet18 is deliberately chosen over larger backbones: the dataset is
  small (1080 train images), grayscale and low-resolution, so capacity is
  not the bottleneck and a small model reduces overfitting risk and CPU
  inference latency.
- The first convolution is replaced to accept 1 input channel.
- When ``pretrained=True`` the ImageNet weights are loaded; the stem is
  then re-initialized for 1 channel (ImageNet pretrained 3-channel stem
  weights cannot be reused directly for grayscale). A cheap and common
  alternative (averaging the 3-channel kernel) was tested informally and
  not adopted: with only one channel of steel texture the averaged filter
  behaves like a luminance edge detector, which is acceptable, but random
  init + full training of the stem kept validation slightly more stable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from ml.config import TrainConfig


def build_model(cfg: TrainConfig, num_classes: int) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
    model = resnet18(weights=weights)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)
