"""Training loop with validation, checkpointing, early stopping and scheduling.

The trainer is framework-light: plain PyTorch, no external experiment
tracker. Each run produces an experiment directory under ``models/experiments``
containing:
- best.pth            best checkpoint by the early-stopping metric
- last.pth            final checkpoint (resume/reproducibility)
- history.json        per-epoch train/val metrics
- metadata.json       config, environment, final metrics, timings
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.config import AppConfig, set_global_seed, describe_environment, config_to_dict
from ml.training.model import build_model, model_size_mb

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2] / "models" / "experiments"


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    val_f1_macro: float
    lr: float


class EarlyStopping:
    """Stop when the monitored metric has not improved for ``patience`` epochs."""

    def __init__(self, patience: int, mode: str = "max"):
        assert mode in ("max", "min")
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Returns True when this step is a new best."""
        improved = self.best is None or (
            value > self.best if self.mode == "max" else value < self.best
        )
        if improved:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def _warmup_factor(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return 1.0
    return (epoch + 1) / warmup_epochs


def build_scheduler(cfg: TrainConfig, optimizer) -> Any:
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
    return None


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """Single epoch. optimizer=None -> evaluation mode (no gradients)."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_targets: list[int] = []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * targets.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_targets


def macro_f1(preds: list[int], targets: list[int], num_classes: int) -> float:
    """Unweighted macro F1 computed from scratch (no sklearn dependency here)."""
    f1s = []
    for c in range(num_classes):
        tp = sum(1 for p, t in zip(preds, targets) if p == c and t == c)
        fp = sum(1 for p, t in zip(preds, targets) if p == c and t != c)
        fn = sum(1 for p, t in zip(preds, targets) if p != c and t == c)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return sum(f1s) / len(f1s)


def train(cfg: AppConfig, experiment_name: str, data_root: Path) -> dict[str, Any]:
    from ml.data.dataset import make_loaders

    set_global_seed(cfg.data.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = EXPERIMENTS_ROOT / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg.train, num_classes=len(cfg.data.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = build_scheduler(cfg.train, optimizer)

    train_loader, val_loader, _ = make_loaders(
        data_root, cfg.data, cfg.train.batch_size, cfg.train.num_workers
    )

    stopper = EarlyStopping(cfg.train.early_stopping_patience)
    history: list[dict[str, Any]] = []
    best_metric = -1.0
    t0 = time.time()

    for epoch in range(cfg.train.epochs):
        # Linear warmup applied by scaling LR before stepping the main scheduler.
        if epoch < cfg.train.warmup_epochs:
            warm = _warmup_factor(epoch, cfg.train.warmup_epochs)
            for g in optimizer.param_groups:
                g["lr"] = cfg.train.lr * warm
        elif scheduler is not None and epoch == cfg.train.warmup_epochs:
            for g in optimizer.param_groups:
                g["lr"] = cfg.train.lr  # restore base LR before scheduler decay

        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        with torch.no_grad():
            va_loss, va_acc, preds, targets = run_epoch(model, val_loader, criterion, None, device)
        va_f1 = macro_f1(preds, targets, len(cfg.data.classes))
        lr_now = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(tr_loss, 4),
                "train_acc": round(tr_acc, 4),
                "val_loss": round(va_loss, 4),
                "val_acc": round(va_acc, 4),
                "val_f1_macro": round(va_f1, 4),
                "lr": lr_now,
            }
        )
        print(
            f"epoch {epoch+1:02d} train_loss={tr_loss:.3f} val_loss={va_loss:.3f} "
            f"val_acc={va_acc:.3f} val_f1={va_f1:.3f} lr={lr_now:.2e}",
            flush=True,
        )

        if cfg.train.early_stopping_metric == "val_f1_macro":
            improved = stopper.step(va_f1)
        elif cfg.train.early_stopping_metric == "val_acc":
            improved = stopper.step(va_acc)
        else:
            raise ValueError(f"unknown early stopping metric {cfg.train.early_stopping_metric}")

        if improved:
            best_metric = va_f1 if cfg.train.early_stopping_metric == "val_f1_macro" else va_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": list(cfg.data.classes),
                    "arch": cfg.train.model_arch,
                    "epoch": epoch + 1,
                    "val_metrics": history[-1],
                },
                exp_dir / "best.pth",
            )

        if scheduler is not None and epoch >= cfg.train.warmup_epochs:
            scheduler.step()

        if stopper.should_stop:
            print(f"early stopping at epoch {epoch+1} (best={best_metric:.4f})", flush=True)
            break

    elapsed = time.time() - t0
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": list(cfg.data.classes),
            "arch": cfg.train.model_arch,
            "epoch": len(history),
        },
        exp_dir / "last.pth",
    )
    metadata = {
        "experiment": experiment_name,
        "config": config_to_dict(cfg),
        "environment": describe_environment(),
        "train_seconds": round(elapsed, 1),
        "epochs_run": len(history),
        "best_epoch": int(max(history, key=lambda h: h["val_f1_macro"])["epoch"]),
        "best_val_f1_macro": round(max(h["val_f1_macro"] for h in history), 4),
        "model_size_mb_fp32": round(model_size_mb(model), 2),
    }
    (exp_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (exp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    import argparse

    from ml.config import load_config
    from ml.data.prepare import prepare_dataset

    parser = argparse.ArgumentParser(description="Train the defect classifier")
    parser.add_argument("--config", default=Path(__file__).resolve().parents[1] / "configs" / "train_v1.yaml")
    parser.add_argument("--name", default="v1_resnet18", help="experiment name")
    parser.add_argument("--data-root", default=None, help="override processed data root")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(args.data_root) if args.data_root else prepare_dataset(cfg.data)
    meta = train(cfg, args.name, root)
    print(json.dumps(meta, indent=2))
