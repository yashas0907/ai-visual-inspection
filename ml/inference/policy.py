"""Serving policy + image validation — torch-free by design.

Confidence tiers and upload validation are operational rules, not model
logic; keeping them in a dependency-free module lets the ONNX serving
backend (no torch installed) reuse the exact same policy code as the torch
backend. ml/inference/predictor.py re-exports these for compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not a readable, supported image."""


@dataclass(frozen=True)
class Prediction:
    label: str
    label_index: int
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    model_version: str


@dataclass(frozen=True)
class ConfidenceTiers:
    high: float = 0.80
    medium: float = 0.60  # >= this and < high -> medium; below -> low


def confidence_tier(confidence: float, tiers: ConfidenceTiers) -> str:
    """Map a raw model confidence to a human-review policy tier.

    Model confidence is a softmax output, NOT a calibrated probability;
    tiers encode an explicit operational policy instead of pretending
    certainty. Values are configurable through the tier object.
    """
    if confidence >= tiers.high:
        return "high"
    if confidence >= tiers.medium:
        return "medium"
    return "low"


def validate_image_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise ImageValidationError("file not found")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ImageValidationError(
            f"unsupported format '{path.suffix}'; allowed: {sorted(IMAGE_EXTENSIONS)}"
        )
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception as exc:
        raise ImageValidationError("corrupted or unreadable image") from exc


def load_pil_image(path: Path) -> Image.Image:
    validate_image_file(path)
    try:
        with Image.open(path) as im:
            return im.convert("L")  # dataset domain is grayscale
    except Exception as exc:
        raise ImageValidationError("failed to decode image") from exc
