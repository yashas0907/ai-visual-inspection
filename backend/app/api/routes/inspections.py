"""Inspection endpoints: create, list, detail, feedback."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.schemas.inspection import (
    FeedbackCreate,
    FeedbackOut,
    InspectionOut,
    InspectionPage,
)
from app.services import inspection_service, model_service

router = APIRouter(prefix="/inspections", tags=["inspections"])

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/pjpeg",
    "image/png", "image/bmp", "image/webp", "image/x-ms-bmp",
}


@router.post(
    "",
    response_model=InspectionOut,
    status_code=201,
    summary="Upload an image and run a full inspection",
)
async def create_inspection(
    file: UploadFile = File(..., description="Surface image (jpg/jpeg/png/bmp/webp)"),
    db: Session = Depends(get_db),
) -> InspectionOut:
    settings = get_settings()

    # Model must be loaded (startup may skip if checkpoint missing).
    if not model_service.model_ready():
        try:
            model_service.load_model()
        except FileNotFoundError:
            from app.core.exceptions import ModelNotReadyError
            raise ModelNotReadyError(
                "model checkpoint is not available; run scripts/train.sh first"
            )

    # Cheap header-based pre-check (byte-level checks happen in the service).
    ctype = (file.content_type or "").lower()
    if ctype and ctype not in ALLOWED_CONTENT_TYPES and not ctype.startswith("image/"):
        from app.core.exceptions import InvalidImageError
        raise InvalidImageError(f"content-type '{ctype}' is not an image type")

    data = await file.read(settings.max_upload_bytes + 1)  # cap read size
    result = inspection_service.run_inspection(db, data, file.filename or "upload")
    return result


@router.get("", response_model=InspectionPage, summary="List inspections")
def list_inspections(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    label: str | None = Query(None, description="Filter by predicted defect label"),
    severity: str | None = Query(None, description="Filter by severity (minor|major|critical)"),
    tier: str | None = Query(None, description="Filter by confidence tier (high|medium|low)"),
    has_feedback: bool | None = Query(None, description="Filter by feedback presence"),
    search: str | None = Query(None, max_length=100, description="Filename substring"),
    db: Session = Depends(get_db),
) -> InspectionPage:
    total, items = inspection_service.list_inspections(
        db, page, page_size, label, severity, tier, has_feedback, search
    )
    return InspectionPage(total=total, page=page, page_size=page_size, items=items)


@router.get("/{inspection_id}", response_model=InspectionOut, summary="Get one inspection")
def get_inspection(inspection_id: int, db: Session = Depends(get_db)) -> InspectionOut:
    return inspection_service.get_inspection(db, inspection_id)


@router.post(
    "/{inspection_id}/feedback",
    response_model=FeedbackOut,
    status_code=201,
    summary="Submit inspector feedback for one inspection",
)
def submit_feedback(
    inspection_id: int,
    payload: FeedbackCreate,
    db: Session = Depends(get_db),
) -> FeedbackOut:
    inspection_service.submit_feedback(db, inspection_id, payload)
    insp = inspection_service.get_inspection(db, inspection_id)
    return insp.feedback
