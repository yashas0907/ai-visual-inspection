"""Datasets and transforms for training/evaluation.

Transform policy:
- Train: resize + light augmentation (flip/rotate/translate). Heavy photometric
  distortion is intentionally avoided because defects are texture-driven and
  grayscale; aggressive changes can destroy the defect signature.
- Val/Test/Inference: deterministic resize + normalize only.

Normalization statistics were computed directly over all 1080 training
images (uint8->/255, two-pass): mean=0.5088, std=0.2095. See
ml/preprocessing/stats.py and docs/dataset.md for how to reproduce.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from ml.config import DataConfig
from ml.inference.constants import IMAGE_SIZE, MEAN, STD  # single source of truth


def build_transforms(cfg: DataConfig, train: bool) -> transforms.Compose:
    to_tensor = transforms.ToTensor()  # 1-channel uint8 -> float32 in [0,1]
    if train:
        return transforms.Compose(
            [
                transforms.Resize((cfg.image_size, cfg.image_size), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                to_tensor,
                transforms.Normalize(MEAN, STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size), antialias=True),
            to_tensor,
            transforms.Normalize(MEAN, STD),
        ]
    )


class NEUDetDataset(Dataset):
    """Thin wrapper over ImageFolder layout with explicit class ordering."""

    def __init__(self, root: str | Path, cfg: DataConfig, train: bool):
        self.classes = list(cfg.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        # torchvision's default loader forces RGB; the data is grayscale.
        grayscale_loader = lambda p: Image.open(p).convert("L")  # noqa: E731
        self.inner = datasets.ImageFolder(
            str(root), transform=build_transforms(cfg, train), loader=grayscale_loader
        )
        if sorted(self.inner.classes) != sorted(self.classes):
            raise ValueError(
                f"Class mismatch: on-disk {self.inner.classes} vs config {self.classes}"
            )
        # Remap ImageFolder's alphabetical idx to config idx order.
        self._remap = torch.tensor(
            [self.class_to_idx[c] for c in self.inner.classes], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int):
        img, folder_idx = self.inner[idx]
        return img, int(self._remap[folder_idx])


def make_loaders(
    processed_root: str | Path, cfg: DataConfig, batch_size: int, num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = NEUDetDataset(Path(processed_root) / "train", cfg, train=True)
    val_ds = NEUDetDataset(Path(processed_root) / "val", cfg, train=False)
    test_ds = NEUDetDataset(Path(processed_root) / "test", cfg, train=False)
    common = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": False}
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader
