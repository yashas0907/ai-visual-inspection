"""Inspection orchestration: the full production inference pipeline.

    Upload -> validation -> sanitize -> persist image -> predict (model)
           -> explainability (Grad-CAM) -> confidence tier -> severity rules
           -> persist inspection+prediction -> response

Also provides history queries, feedback submission (human-in-the-loop) and
analytics aggregation over stored rows only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import (
    FeedbackAlreadyExistsError,
    InspectionNotFoundError,
    InvalidFeedbackError,
)
from app.core.logging import get_logger
from app.db.models import Feedback, Inspection, ModelVersion, Prediction
from app.schemas.inspection import (
    ALLOWED_DEFECT_CLASSES,
    FeedbackCreate,
    InspectionOut,
)
from app.services import image_service, model_service
from app.services.severity import Severity, assess_severity, SeverityInput

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

def run_inspection(db: Session, data: bytes, original_filename: str) -> InspectionOut:
    settings = get_settings()
    predictor = model_service.get_predictor()
    gradcam = model_service.get_gradcam()
    tiers = model_service.get_confidence_tiers()

    # 1) validation + decode (raises 4xx errors)
    image, sha256 = image_service.validate_and_sanitize(data, original_filename)

    # 2) persist canonical image
    stored = image_service.store_image(image, sha256)
    rel_path = image_service.storage_relative(stored)

    # 3) model prediction
    pred = predictor.predict_pil(image)

    # 4) explainability
    try:
        expl, _, _, _ = gradcam.generate(image)
        heatmap = expl.heatmap
        bbox = expl.bbox
        coverage = float(expl.region_area_ratio)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("gradcam failed: %s", exc)
        heatmap, bbox, coverage = None, None, 0.0

    # 5) confidence tier (policy; torch-free module)
    from ml.inference.policy import confidence_tier

    tier = confidence_tier(pred.confidence, tiers)

    # 6) severity rules (business logic, pure function)
    assessment = assess_severity(
        SeverityInput(
            defect_label=pred.label,
            confidence_tier=tier,
            evidence_coverage=coverage,
        )
    )

    # 7) explainability overlay artifact
    overlay_rel = None
    if heatmap is not None:
        # results live next to uploads under the same storage root so that
        # relative paths and the static route stay consistent
        results_dir = Path(settings.upload_dir).parent / "results"
        out_name = Path(rel_path).stem + "_overlay.jpg"
        overlay_abs = image_service.build_overlay(
            image, heatmap, bbox, results_dir / out_name
        )
        overlay_rel = image_service.storage_relative(overlay_abs)

    # 8) persist
    inspection = Inspection(
        image_path=rel_path,
        overlay_path=overlay_rel,
        original_filename=Path(original_filename).name[:255] or "upload",
        image_width=image.width,
        image_height=image.height,
        image_sha256=sha256,
    )
    prediction = Prediction(
        model_version=pred.model_version,
        predicted_label=pred.label,
        predicted_index=pred.label_index,
        confidence=pred.confidence,
        confidence_tier=tier,
        probabilities=pred.probabilities,
        needs_review=(tier == "low"),
        evidence_coverage=round(coverage, 4),
        evidence_bbox=(
            {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]}
            if bbox
            else None
        ),
        severity=assessment.severity.value,
        severity_score=assessment.score,
        severity_components=assessment.components,
        recommended_action=assessment.recommended_action,
        latency_ms=pred.latency_ms,
    )
    inspection.prediction = prediction
    db.add(inspection)
    db.commit()
    db.refresh(inspection)
    logger.info(
        "inspection %s: %s %.3f (%s) severity=%s latency=%.1fms",
        inspection.id, pred.label, pred.confidence, tier,
        assessment.severity.value, pred.latency_ms,
    )
    return _to_out(db, inspection)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _to_out(db: Session, inspection: Inspection) -> InspectionOut:
    settings = get_settings()
    image_url = f"/api/storage/{inspection.image_path}" if inspection.image_path else None
    overlay_url = f"/api/storage/{inspection.overlay_path}" if inspection.overlay_path else None
    return InspectionOut(
        id=inspection.id,
        created_at=inspection.created_at,
        original_filename=inspection.original_filename,
        image_width=inspection.image_width,
        image_height=inspection.image_height,
        image_url=image_url,
        overlay_url=overlay_url,
        prediction=inspection.prediction,
        feedback=inspection.feedback,
    )


def list_inspections(
    db: Session,
    page: int = 1,
    page_size: int = 20,
    label: str | None = None,
    severity: str | None = None,
    tier: str | None = None,
    has_feedback: bool | None = None,
    search: str | None = None,
) -> tuple[int, list[InspectionOut]]:
    q = select(Inspection).order_by(Inspection.created_at.desc())
    if label or severity or tier or has_feedback is not None:
        q = q.join(Prediction, Prediction.inspection_id == Inspection.id)
        if label:
            q = q.where(Prediction.predicted_label == label)
        if severity:
            q = q.where(Prediction.severity == severity)
        if tier:
            q = q.where(Prediction.confidence_tier == tier)
        if has_feedback is True:
            q = q.where(Inspection.feedback != None)  # noqa: E711
        elif has_feedback is False:
            q = q.where(Inspection.feedback == None)  # noqa: E711
    if search:  # filename substring
        q = q.where(Inspection.original_filename.ilike(f"%{search}%"))

    total = db.scalar(select(func.count()).select_from(q.subquery()))
    items = (
        db.execute(q.offset((page - 1) * page_size).limit(page_size))
        .scalars()
        .all()
    )
    return total, [_to_out(db, i) for i in items]


def get_inspection(db: Session, inspection_id: int) -> InspectionOut:
    insp = db.get(Inspection, inspection_id)
    if insp is None:
        raise InspectionNotFoundError(f"inspection {inspection_id} not found")
    return _to_out(db, insp)


# ---------------------------------------------------------------------------
# Human-in-the-loop feedback
# ---------------------------------------------------------------------------

def submit_feedback(db: Session, inspection_id: int, payload: FeedbackCreate) -> InspectionOut:
    insp = db.get(Inspection, inspection_id)
    if insp is None:
        raise InspectionNotFoundError(f"inspection {inspection_id} not found")
    if insp.feedback is not None:
        raise FeedbackAlreadyExistsError(
            f"feedback already exists for inspection {inspection_id}"
        )
    if not payload.is_correct:
        if not payload.corrected_label:
            raise InvalidFeedbackError("corrected_label is required when is_correct is false")
        if payload.corrected_label not in ALLOWED_DEFECT_CLASSES:
            raise InvalidFeedbackError(
                f"corrected_label '{payload.corrected_label}' not in supported classes "
                f"{ALLOWED_DEFECT_CLASSES}"
            )
    if payload.is_correct and payload.corrected_label:
        raise InvalidFeedbackError("corrected_label must be empty when is_correct is true")

    feedback = Feedback(
        inspection_id=insp.id,
        is_correct=payload.is_correct,
        corrected_label=payload.corrected_label,
        notes=payload.notes,
        inspector_name=payload.inspector_name,
        model_version_at_feedback=insp.prediction.model_version,
        review_status="pending",
    )
    db.add(feedback)
    db.commit()
    db.refresh(insp)
    logger.info("feedback recorded for inspection %s (correct=%s)", insp.id, payload.is_correct)
    return _to_out(db, insp)


# ---------------------------------------------------------------------------
# Analytics (all numbers computed from stored rows only)
# ---------------------------------------------------------------------------

def compute_analytics(db: Session) -> dict:
    total = db.scalar(select(func.count()).select_from(Inspection)) or 0
    with_pred = db.scalar(
        select(func.count()).select_from(Prediction)
    ) or 0
    with_fb = db.scalar(select(func.count()).select_from(Feedback)) or 0

    severity_dist = dict(
        db.execute(select(Prediction.severity, func.count()).group_by(Prediction.severity)).all()
    )
    label_dist = dict(
        db.execute(
            select(Prediction.predicted_label, func.count()).group_by(Prediction.predicted_label)
        ).all()
    )
    tier_dist = dict(
        db.execute(
            select(Prediction.confidence_tier, func.count()).group_by(Prediction.confidence_tier)
        ).all()
    )

    # confidence histogram (10 bins over [0,1])
    confs = db.execute(select(Prediction.confidence)).scalars().all()
    hist = []
    if confs:
        arr = np.array(confs, dtype=float)
        counts, _ = np.histogram(arr, bins=10, range=(0.0, 1.0))
        for i, c in enumerate(counts):
            hist.append(
                {
                    "bin_start": round(i / 10, 2),
                    "bin_end": round((i + 1) / 10, 2),
                    "count": int(c),
                }
            )

    # feedback: correction rate + per-true-label counts
    correction_rate = None
    corrections_by_true_label: dict[str, int] = {}
    if with_fb > 0:
        n_correct = db.scalar(
            select(func.count()).select_from(Feedback).where(Feedback.is_correct)
        ) or 0
        correction_rate = round(1 - n_correct / with_fb, 4)
        rows = db.execute(
            select(Feedback.corrected_label, func.count())
            .where(Feedback.corrected_label != None)  # noqa: E711
            .group_by(Feedback.corrected_label)
        ).all()
        corrections_by_true_label = dict(rows)

    mean_latency = db.scalar(select(func.avg(Prediction.latency_ms)))

    # per-day volume (UTC dates)
    day_rows = db.execute(
        select(
            func.date(Inspection.created_at).label("d"),
            func.count(),
        )
        .group_by("d")
        .order_by("d")
    ).all()
    per_day = [{"date": str(d), "count": int(c)} for d, c in day_rows]

    # recent inspections (10)
    recent_q = select(Inspection).order_by(Inspection.created_at.desc()).limit(10)
    recent = [_to_out(db, i) for i in db.execute(recent_q).scalars().all()]

    defect_rate = None
    if with_pred > 0:
        # In this dataset every image contains a defect by design, so the
        # platform-level "defect rate" is the share of inspections the model
        # flagged at/above the review threshold (needs_review=False).
        n_review = db.scalar(
            select(func.count()).select_from(Prediction).where(Prediction.needs_review)
        ) or 0
        defect_rate = round(1 - n_review / with_pred, 4)

    return {
        "total_inspections": total,
        "total_with_feedback": with_fb,
        "defect_rate": defect_rate,
        "severity_distribution": severity_dist,
        "label_distribution": label_dist,
        "confidence_tier_distribution": tier_dist,
        "confidence_histogram": hist,
        "correction_rate": correction_rate,
        "corrections_by_true_label": corrections_by_true_label,
        "mean_latency_ms": round(mean_latency, 2) if mean_latency is not None else None,
        "per_day_counts": per_day,
        "recent_inspections": recent,
    }
