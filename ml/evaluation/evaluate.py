"""Model evaluation on the held-out test split.

Produces genuine metrics only:
- accuracy, macro/micro precision/recall/F1 (per-class + macro)
- confusion matrix
- ROC-AUC (macro OVR, meaningful for a balanced 6-class problem)
- per-image inference latency + model size

Results are written to the experiment directory as ``test_metrics.json``
and a confusion-matrix PNG. Nothing here fabricates numbers; all metrics
come from the model checkpoints produced by the training script.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.config import AppConfig
from ml.data.dataset import NEUDetDataset
from ml.training.model import build_model, model_size_mb

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2] / "models" / "experiments"


def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (targets, predicted class idx, softmax probabilities)."""
    model.eval()
    targets: list[int] = []
    preds: list[int] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for images, t in loader:
            logits = model(images.to(device))
            p = torch.softmax(logits, dim=1)
            targets.extend(t.tolist())
            preds.extend(p.argmax(dim=1).cpu().tolist())
            probs.extend(p.cpu().numpy())
    return np.array(targets), np.array(preds), np.array(probs)


def per_class_metrics(targets: np.ndarray, preds: np.ndarray, num_classes: int) -> list[dict[str, float]]:
    out = []
    for c in range(num_classes):
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out.append(
            {
                "class_index": c,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )
    return out


def confusion_matrix(targets: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm


def macro_roc_auc(targets: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    """Macro one-vs-rest ROC-AUC computed without sklearn (trapz over sorted scores)."""
    aucs = []
    for c in range(num_classes):
        y = (targets == c).astype(int)
        scores = probs[:, c]
        order = np.argsort(-scores, kind="stable")
        y_sorted = y[order]
        scores_sorted = scores[order]
        # rank-based AUC handles ties correctly
        ranks = np.empty(len(scores), dtype=float)
        i = 0
        rank = 1.0
        while i < len(scores_sorted):
            j = i
            while j + 1 < len(scores_sorted) and scores_sorted[j + 1] == scores_sorted[i]:
                j += 1
            avg_rank = (rank + (rank + (j - i))) / 2.0
            ranks[i : j + 1] = avg_rank
            rank += j - i + 1
            i = j + 1
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        # ranks were assigned over descending scores (rank 1 = highest),
        # so invert the Mann-Whitney U relative to the ascending convention.
        u_desc = ranks[y_sorted == 1].sum() - n_pos * (n_pos + 1) / 2
        auc = 1.0 - u_desc / (n_pos * n_neg)
        aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float("nan")


def measure_latency(
    model: nn.Module, loader: DataLoader, device: torch.device, n_batches: int = 5
) -> dict[str, float]:
    """Mean single-image latency in ms (after one warmup batch), CPU."""
    model.eval()
    n_warm = 0
    total = 0.0
    count = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            if n_warm == 0:
                model(images)  # warmup
                n_warm = 1
                continue
            start = time.perf_counter()
            model(images)
            total += time.perf_counter() - start
            count += images.size(0)
            if count >= n_batches * loader.batch_size:
                break
    return {
        "mean_latency_ms_per_image_cpu": round(total / max(count, 1) * 1000, 2),
        "images_measured": count,
    }


def evaluate(cfg: AppConfig, experiment_dir: Path, data_root: Path) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(experiment_dir / "best.pth", map_location=device, weights_only=True)
    model = build_model(cfg.train, num_classes=len(cfg.data.classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_ds = NEUDetDataset(Path(data_root) / "test", cfg.data, train=False)
    loader = DataLoader(test_ds, batch_size=cfg.eval.batch_size, num_workers=0, shuffle=False)

    targets, preds, probs = collect_predictions(model, loader, device)
    n = len(cfg.data.classes)
    per_class = per_class_metrics(targets, preds, n)
    cm = confusion_matrix(targets, preds, n)
    metrics: dict[str, Any] = {
        "experiment": experiment_dir.name,
        "checkpoint": "best.pth",
        "best_epoch": ckpt.get("epoch"),
        "test_size": int(len(targets)),
        "accuracy": round(float((preds == targets).mean()), 4),
        "macro_precision": round(float(np.mean([m["precision"] for m in per_class])), 4),
        "macro_recall": round(float(np.mean([m["recall"] for m in per_class])), 4),
        "macro_f1": round(float(np.mean([m["f1"] for m in per_class])), 4),
        "macro_roc_auc_ovr": round(macro_roc_auc(targets, probs, n), 4),
        "per_class": [
            {"class": cfg.data.classes[m["class_index"]], **{k: v for k, v in m.items() if k != "class_index"}}
            for m in per_class
        ],
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_classes": list(cfg.data.classes),
        "model_size_mb_fp32": round(model_size_mb(model), 2),
    }
    metrics.update(measure_latency(model, loader, device))

    (experiment_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Confusion matrix PNG (matplotlib, optional import guard for servers)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n), cfg.data.classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n), cfg.data.classes, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Test confusion matrix (n={len(targets)})")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(experiment_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is best effort
        metrics["plot_error"] = str(exc)
    return metrics


if __name__ == "__main__":
    import argparse

    from ml.config import load_config
    from ml.data.prepare import prepare_dataset

    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split")
    parser.add_argument("--config", default=Path(__file__).resolve().parents[1] / "configs" / "train_v1.yaml")
    parser.add_argument("--experiment", default="v1_resnet18")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(args.data_root) if args.data_root else prepare_dataset(cfg.data)
    out = evaluate(cfg, EXPERIMENTS_ROOT / args.experiment, root)
    print(json.dumps(out, indent=2))
