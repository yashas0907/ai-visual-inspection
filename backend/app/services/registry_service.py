"""Model registry service: registers active version + genuine metrics."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import ModelVersion
from app.services import model_service


def register_active_version(db: Session) -> ModelVersion | None:
    """Insert or refresh the row for the active model version.

    Metrics are copied verbatim from the experiment artifacts produced by
    the ML pipeline — nothing is invented here. If evaluation hasn't run,
    the metrics dict is empty and the UI shows a clear 'not evaluated' note.
    """
    settings = get_settings()
    data = model_service.load_experiment_metrics()
    test_metrics = data.get("test_metrics", {})
    meta = data.get("training_metadata", {})
    history = data.get("training_history", [])

    from ml.config import DataConfig

    classes = list(meta.get("config", {}).get("data", {}).get("classes", [])) or list(
        DataConfig.classes
    )

    existing = db.execute(
        select(ModelVersion).where(ModelVersion.version == settings.model_version)
    ).scalar_one_or_none()

    best_val = None
    if history:
        best_val = max(h.get("val_f1_macro", 0) for h in history)

    payload = dict(
        experiment=settings.model_experiment,
        arch=meta.get("config", {}).get("train", {}).get("model_arch", "resnet18"),
        classes=classes,
        metrics={
            "test": test_metrics,
            "best_val_f1_macro": best_val,
        },
        metadata_json={
            "environment": meta.get("environment", {}),
            "train_seconds": meta.get("train_seconds"),
            "epochs_run": meta.get("epochs_run"),
            "best_epoch": meta.get("best_epoch"),
            "model_size_mb_fp32": meta.get("model_size_mb_fp32"),
        },
        is_active=True,
    )
    if existing:
        for k, v in payload.items():
            setattr(existing, k, v)
        version = existing
    else:
        version = ModelVersion(version=settings.model_version, **payload)
        db.add(version)
    db.commit()
    db.refresh(version)
    return version


def get_active_version(db: Session) -> ModelVersion | None:
    settings = get_settings()
    return db.execute(
        select(ModelVersion).where(ModelVersion.version == settings.model_version)
    ).scalar_one_or_none()
