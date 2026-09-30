"""Secure static file serving for stored images/overlays.

Storage layout: <storage>/uploads/... and <storage>/results/...
DB rows store POSIX-style relative paths like ``uploads/abc.jpg``.
This route resolves only within the storage root (no traversal).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.core.exceptions import AppError


class NotFoundError(AppError):
    status_code = 404
    detail = "not found"

router = APIRouter(prefix="/storage", tags=["storage"])

MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


@router.get("/{relative_path:path}", summary="Serve a stored image")
def serve_image(relative_path: str) -> FileResponse:
    storage_root = Path(get_settings().upload_dir).parent.resolve()
    # Normalize and reject traversal attempts explicitly. is_relative_to is
    # a strict path-containment check (no sibling-prefix bypass, e.g.
    # /storage-x when the root is /storage). NOTE: resolve() raises OSError
    # on Linux for path components >255 bytes (ENAMETOOLONG) — Windows
    # tolerates them — so long-path errors are treated as 404, not 500.
    try:
        candidate = (storage_root / relative_path).resolve()
        candidate.relative_to(storage_root)
    except (OSError, ValueError):
        raise NotFoundError("path traversal rejected") from None
    # NOTE: Path.is_file() PROPAGATES non-ENOSENT OSErrors — on Linux a path
    # component >255 bytes raises ENAMETOOLONG (Windows tolerates it), which
    # previously surfaced as a 500. Treat any stat failure as "not found".
    try:
        is_file = candidate.is_file()
    except OSError:
        is_file = False
    if not is_file:
        raise NotFoundError("image not found")
    ext = candidate.suffix.lower()
    if ext not in MIME:
        raise NotFoundError("unsupported file")
    return FileResponse(candidate, media_type=MIME[ext])
