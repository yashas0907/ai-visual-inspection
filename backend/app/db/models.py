"""ORM entities: Inspection, Prediction, Feedback, ModelVersion, FeedbackReview.

Design notes:
- Image BLOBs are deliberately NOT stored in the DB. Inspections reference
  relative storage paths (uploads + generated explainability overlays), which
  are served by the static-files route. Rationale in docs/architecture.md.
- Every prediction row carries model_version so analytics can attribute
  performance to a specific artifact.
- Feedback rows are immutable (one per inspection) and reference the model
  version that produced the original prediction. A FeedbackReview workflow
  gates feedback before it can be used as training data.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Inspection(Base):
    __tablename__ = "inspections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    # Storage references (relative paths under storage dir), never blobs.
    image_path: Mapped[str] = mapped_column(String(255), nullable=False)
    overlay_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    image_width: Mapped[int] = mapped_column(Integer, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False)
    image_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    prediction: Mapped["Prediction | None"] = relationship(
        back_populates="inspection", uselist=False, cascade="all, delete-orphan"
    )
    feedback: Mapped["Feedback | None"] = relationship(
        back_populates="inspection", uselist=False, cascade="all, delete-orphan"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    inspection_id: Mapped[int] = mapped_column(
        ForeignKey("inspections.id", ondelete="CASCADE"), unique=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    predicted_label: Mapped[str] = mapped_column(String(64), nullable=False)
    predicted_index: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_tier: Mapped[str] = mapped_column(String(8), nullable=False)  # high/medium/low
    probabilities: Mapped[dict] = mapped_column(JSON, nullable=False)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)

    # Explainability artifacts (storage refs + numbers)
    evidence_coverage: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_bbox: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Severity (business rule output, not model output)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    severity_score: Mapped[int] = mapped_column(Integer, nullable=False)
    severity_components: Mapped[dict] = mapped_column(JSON, nullable=False)
    recommended_action: Mapped[str] = mapped_column(Text, nullable=False)

    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)

    inspection: Mapped[Inspection] = relationship(back_populates="prediction")


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (UniqueConstraint("inspection_id", name="uq_feedback_inspection"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    inspection_id: Mapped[int] = mapped_column(
        ForeignKey("inspections.id", ondelete="CASCADE"), unique=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    corrected_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Free-form label for "other" cases or "no defect" disputes
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    inspector_name: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Audit: the model version whose prediction is being judged
    model_version_at_feedback: Mapped[str] = mapped_column(String(32), nullable=False)

    # Human-in-the-loop gating: feedback starts unreviewed and must be
    # approved by a maintainer before it can enter the training set.
    review_status: Mapped[str] = mapped_column(
        String(16), default="pending", nullable=False, index=True
    )  # pending | approved | rejected
    reviewed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    inspection: Mapped[Inspection] = relationship(back_populates="feedback")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    experiment: Mapped[str] = mapped_column(String(64), nullable=False)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    arch: Mapped[str] = mapped_column(String(64), nullable=False)
    classes: Mapped[list] = mapped_column(JSON, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)  # genuine test metrics
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
