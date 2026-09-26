"""API request/response schemas (Pydantic v2)."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ConfidenceTier = Literal["high", "medium", "low"]
SeverityLevel = Literal["minor", "major", "critical"]
ReviewStatus = Literal["pending", "approved", "rejected"]

ALLOWED_DEFECT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    predicted_label: str
    predicted_index: int
    confidence: float
    confidence_tier: ConfidenceTier
    probabilities: dict[str, float]
    needs_review: bool
    evidence_coverage: float
    evidence_bbox: dict | None
    severity: SeverityLevel
    severity_score: int
    severity_components: dict[str, int]
    recommended_action: str
    latency_ms: float
    model_version: str


class FeedbackCreate(BaseModel):
    is_correct: bool
    corrected_label: str | None = Field(
        default=None,
        description="Required when is_correct is false. One of the supported defect classes.",
    )
    notes: str | None = Field(default=None, max_length=2000)
    inspector_name: str | None = Field(default=None, max_length=64)


class FeedbackOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    inspection_id: int
    created_at: datetime
    is_correct: bool
    corrected_label: str | None
    notes: str | None
    inspector_name: str | None
    model_version_at_feedback: str
    review_status: ReviewStatus


class InspectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    created_at: datetime
    original_filename: str
    image_width: int
    image_height: int
    image_url: str | None = None
    overlay_url: str | None = None
    prediction: PredictionOut | None = None
    feedback: FeedbackOut | None = None


class InspectionPage(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[InspectionOut]


class AnalyticsOut(BaseModel):
    total_inspections: int
    total_with_feedback: int
    defect_rate: float | None  # None when no predictions exist yet
    severity_distribution: dict[str, int]
    label_distribution: dict[str, int]
    confidence_tier_distribution: dict[str, int]
    confidence_histogram: list[dict[str, float | int]]
    correction_rate: float | None  # None when no feedback exists yet
    corrections_by_true_label: dict[str, int]
    mean_latency_ms: float | None
    per_day_counts: list[dict[str, str | int]]
    recent_inspections: list[InspectionOut]


class ModelInfoOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    experiment: str
    arch: str
    classes: list[str]
    image_size: int
    training_summary: dict
    test_metrics: dict
    model_size_mb: float
    confidence_policy: dict[str, float]
    severity_policy: dict


class HealthOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    database: bool
    model_ready: bool
    model_version: str | None
    environment: str
