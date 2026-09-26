"""Domain exceptions mapped to HTTP responses by handlers in main.py."""
from __future__ import annotations


class AppError(Exception):
    status_code = 500
    detail = "internal error"

    def __init__(self, detail: str | None = None):
        super().__init__(detail or self.detail)
        self.detail = detail or self.detail


class InvalidImageError(AppError):
    status_code = 422
    detail = "invalid image"


class ImageTooLargeError(AppError):
    status_code = 413
    detail = "image exceeds size limit"


class InspectionNotFoundError(AppError):
    status_code = 404
    detail = "inspection not found"


class FeedbackAlreadyExistsError(AppError):
    status_code = 409
    detail = "feedback already submitted for this inspection"


class InvalidFeedbackError(AppError):
    status_code = 422
    detail = "invalid feedback payload"


class ModelNotReadyError(AppError):
    status_code = 503
    detail = "model is not ready"
