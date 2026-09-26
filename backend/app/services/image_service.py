"""Image intake: validation, safe storage, overlays.

Security considerations implemented here:
- extension allow-list (jpg/jpeg/png/bmp/webp)
- magic-byte sniffing (real format must match extension family)
- decode + re-encode to a canonical JPEG to defang polyglot files
- size caps (bytes before decode; pixel dimensions after decode)
- filenames are never trusted: stored names are server-generated (uuid4 +
   content hash), so path traversal via upload names is impossible
- images live outside the DB as files under storage/, referenced by path
"""
from __future__ import annotations

import hashlib
import io
import uuid
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.core.config import get_settings
from app.core.exceptions import ImageTooLargeError, InvalidImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Magic bytes accepted for sniffing
MAGIC_SIGNATURES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"BM": "bmp",
    b"RIFF": "webp",  # RIFF....WEBP
}


def _sniff_format(data: bytes) -> str | None:
    for magic, fmt in MAGIC_SIGNATURES.items():
        if data.startswith(magic):
            if fmt == "webp" and data[8:12] != b"WEBP":
                return None
            return fmt
    return None


def validate_and_sanitize(data: bytes, original_filename: str) -> tuple[Image.Image, str]:
    """Validate upload bytes; return (decoded grayscale image, sha256 hex).

    Raises InvalidImageError / ImageTooLargeError with precise details.
    """
    settings = get_settings()
    if not data:
        raise InvalidImageError("empty upload")
    if len(data) > settings.max_upload_bytes:
        raise ImageTooLargeError(
            f"upload is {len(data)/1e6:.1f}MB; limit is {settings.max_upload_mb}MB"
        )
    ext = Path(original_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidImageError(
            f"extension '{ext or '(none)'}' not allowed; allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )
    fmt = _sniff_format(data[:16])
    if fmt is None:
        raise InvalidImageError("file content is not a recognized image format")
    # extension/format family agreement (defense in depth)
    if ext in {".jpg", ".jpeg"} and fmt != "jpeg":
        raise InvalidImageError("file content does not match its extension")
    if ext == ".png" and fmt != "png":
        raise InvalidImageError("file content does not match its extension")
    if ext == ".bmp" and fmt != "bmp":
        raise InvalidImageError("file content does not match its extension")
    if ext == ".webp" and fmt != "webp":
        raise InvalidImageError("file content does not match its extension")

    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()  # detect truncation/corruption
        with Image.open(io.BytesIO(data)) as im:
            image = im.convert("L")  # canonical grayscale (dataset domain)
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("corrupted or unreadable image") from exc

    w, h = image.size
    if w < 32 or h < 32:
        raise InvalidImageError(f"image too small ({w}x{h}); minimum 32x32")
    if w > settings.max_image_dim or h > settings.max_image_dim:
        raise ImageTooLargeError(
            f"image dimensions {w}x{h} exceed the {settings.max_image_dim}px limit"
        )

    sha256 = hashlib.sha256(data).hexdigest()
    return image, sha256


def store_image(image: Image.Image, sha256: str) -> Path:
    """Re-encode to canonical JPEG under storage/uploads; return absolute path."""
    upload_dir = Path(get_settings().upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sha256[:16]}-{uuid.uuid4().hex[:8]}.jpg"
    dest = upload_dir / filename
    image.save(dest, "JPEG", quality=92)
    return dest


def storage_relative(absolute: Path) -> str:
    """Convert an absolute storage path to its relative reference for the DB.

    Robust to relative VI_UPLOAD_DIR settings: both sides are resolved to
    absolute paths (relative roots resolve against the process CWD) before
    computing the difference.
    """
    storage_root = Path(get_settings().upload_dir).parent.resolve()
    return absolute.resolve().relative_to(storage_root).as_posix()


def build_overlay(
    image: Image.Image,
    heatmap_2d,  # numpy (H,W) float in [0,1]
    bbox: tuple[int, int, int, int] | None,
    out_path: Path,
) -> Path:
    """Render heatmap + bbox on the original image for inspector review."""
    import numpy as np

    arr = np.asarray(image, dtype=np.float32) / 255.0
    # Fixed turbo-like colormap approximation via channel blending (no mpl dep)
    hm = np.asarray(heatmap_2d, dtype=np.float32)
    if hm.shape != arr.shape:
        hm_img = Image.fromarray((hm * 255).astype(np.uint8)).resize(
            (arr.shape[1], arr.shape[0]), Image.BILINEAR
        )
        hm = np.asarray(hm_img, dtype=np.float32) / 255.0
    # red-yellow ramp: low evidence -> transparent, high -> red overlay
    overlay = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
    overlay[..., 0] = hm  # red channel by intensity
    overlay[..., 1] = hm * 0.45  # fade toward orange/yellow
    alpha = (hm > 0.35) * 0.45
    blended = arr[..., None] * (1 - alpha[..., None]) + overlay * alpha[..., None]
    out = Image.fromarray((np.clip(blended[..., 0], 0, 1) * 255).astype(np.uint8)) if blended.shape[-1] == 1 else None
    if out is None:
        # blended is (H,W,3) grayscale-replicated? keep it simple: convert
        out = Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, "JPEG", quality=92)
    return out_path
