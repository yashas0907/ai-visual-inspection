"""Shared configuration for training, evaluation and inference.

Configuration is loaded from YAML files under ``ml/configs`` and can be
overridden through environment variables (prefix ``VI_``). This keeps the
training pipeline reproducible without hard-coding hyperparameters in code.
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = Path(__file__).resolve().parents[1]

# NOTE: torch is imported lazily inside the functions below so that the
# inference/serving path (including the ONNX backend) never requires torch
# to be installed. This keeps the cloud deployment image torch-free (~200MB
# instead of ~2GB).


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "neu-surface-defects"
    source_url: str = (
        "https://huggingface.co/datasets/Vania43/neu_det_caption/resolve/main/data/train-00000-of-00001.parquet"
    )
    classes: tuple[str, ...] = (
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    )
    image_size: int = 224
    val_split: float = 0.15
    test_split: float = 0.10
    seed: int = 42
    grayscale: bool = True


@dataclass(frozen=True)
class TrainConfig:
    model_arch: str = "resnet18"
    pretrained: bool = True
    batch_size: int = 32
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # cosine | step | none
    warmup_epochs: int = 2
    early_stopping_patience: int = 8
    early_stopping_metric: str = "val_f1_macro"
    num_workers: int = 0
    amp: bool = False


@dataclass(frozen=True)
class EvalConfig:
    batch_size: int = 64
    confusion_out_prefix: str = "confusion_matrix"


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML config file and validate known fields."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data_raw = dict(raw.get("data", {}) or {})
    train_raw = dict(raw.get("train", {}) or {})
    eval_raw = dict(raw.get("eval", {}) or {})
    data = DataConfig(**data_raw)
    train = TrainConfig(**train_raw)
    ev = EvalConfig(**eval_raw)
    return AppConfig(data=data, train=train, eval=ev)


def set_global_seed(seed: int) -> None:
    """Make runs as reproducible as the platform allows (requires torch)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - platform dependent
        pass


def describe_environment() -> dict[str, Any]:
    import torch

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "platform": sys.platform,
    }


def config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    d = asdict(cfg)
    return json.loads(json.dumps(d, default=str))
