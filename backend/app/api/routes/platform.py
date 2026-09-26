"""Analytics + model info + health endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.schemas.inspection import AnalyticsOut, HealthOut, ModelInfoOut
from app.services import inspection_service, model_service, registry_service
from app.services import severity

router = APIRouter(prefix="", tags=["platform"])


@router.get("/analytics", response_model=AnalyticsOut, summary="Platform analytics")
def analytics(db: Session = Depends(get_db)) -> AnalyticsOut:
    data = inspection_service.compute_analytics(db)
    return AnalyticsOut(**data)


@router.get("/model/info", response_model=ModelInfoOut, summary="Active model card")
def model_info(db: Session = Depends(get_db)) -> ModelInfoOut:
    settings = get_settings()
    data = model_service.load_experiment_metrics()
    test_metrics = data.get("test_metrics", {})
    meta = data.get("training_metadata", {})
    history = data.get("training_history", [])

    version_row = registry_service.get_active_version(db)
    metrics_payload = version_row.metrics if version_row else {"test": test_metrics}

    best_val = None
    if history:
        best_val = max(h.get("val_f1_macro", 0) for h in history)

    from ml.config import DataConfig

    classes = list(
        meta.get("config", {}).get("data", {}).get("classes", [])
    ) or list(DataConfig.classes)

    return ModelInfoOut(
        model_version=settings.model_version,
        experiment=settings.model_experiment,
        arch=meta.get("config", {}).get("train", {}).get("model_arch", "resnet18"),
        classes=classes,
        image_size=int(
            meta.get("config", {}).get("data", {}).get("image_size", 224)
        ),
        training_summary={
            "epochs_run": meta.get("epochs_run"),
            "best_epoch": meta.get("best_epoch"),
            "best_val_f1_macro": best_val,
            "train_seconds": meta.get("train_seconds"),
            "environment": meta.get("environment", {}),
        },
        test_metrics=metrics_payload.get("test", test_metrics),
        model_size_mb=float(meta.get("model_size_mb_fp32", 0.0) or 0.0),
        confidence_policy={
            "high_min": settings.confidence_high,
            "medium_min": settings.confidence_medium,
            "needs_review_below": settings.confidence_medium,
        },
        severity_policy={
            "defect_risk_weights": severity.DEFECT_RISK_WEIGHTS,
            "confidence_tier_points": severity.CONFIDENCE_TIER_POINTS,
            "coverage_bands": [list(b) for b in severity.COVERAGE_BANDS],
            "thresholds": list(severity.SEVERITY_THRESHOLDS),
        },
    )


@router.get("/health", response_model=HealthOut, summary="Liveness/readiness")
def health(db: Session = Depends(get_db)) -> HealthOut:
    settings = get_settings()
    db_ok = True
    try:
        db.execute(select(1))
    except Exception:
        db_ok = False
    ready = model_service.model_ready()
    status = "ok" if (db_ok and ready) else "degraded"
    return HealthOut(
        status=status,
        database=db_ok,
        model_ready=ready,
        model_version=settings.model_version if ready else None,
        environment=settings.environment,
    )
