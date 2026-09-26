"""Unit tests: image validation + security rules."""
from __future__ import annotations

import io

import pytest
from PIL import Image

from tests.conftest import make_jpeg_bytes, make_png_bytes

from app.core.exceptions import ImageTooLargeError, InvalidImageError
from app.services import image_service


class TestValidateAndSanitize:
    def test_accepts_jpeg(self):
        _, sha = image_service.validate_and_sanitize(make_jpeg_bytes(), "part.jpg")
        assert len(sha) == 64

    def test_accepts_png(self):
        _, sha = image_service.validate_and_sanitize(make_png_bytes(), "part.png")
        assert len(sha) == 64

    def test_rejects_wrong_extension_family(self):
        # PNG content named .jpg -> content/extension mismatch
        with pytest.raises(InvalidImageError):
            image_service.validate_and_sanitize(make_png_bytes(), "part.jpg")

    def test_rejects_unknown_extension(self):
        with pytest.raises(InvalidImageError):
            image_service.validate_and_sanitize(b"x" * 100, "part.gif")

    def test_rejects_empty(self):
        with pytest.raises(InvalidImageError):
            image_service.validate_and_sanitize(b"", "part.jpg")

    def test_rejects_non_image_bytes(self):
        with pytest.raises(InvalidImageError):
            image_service.validate_and_sanitize(b"not an image at all" * 10, "part.jpg")

    def test_rejects_truncated_image(self):
        data = make_jpeg_bytes()
        with pytest.raises(InvalidImageError):
            image_service.validate_and_sanitize(data[: len(data) // 2], "part.jpg")

    def test_rejects_oversize(self, monkeypatch):
        from app.core.config import get_settings

        s = get_settings()
        monkeypatch.setattr(s, "max_upload_mb", 0)  # forces 0-byte limit
        with pytest.raises(ImageTooLargeError):
            image_service.validate_and_sanitize(make_jpeg_bytes(), "part.jpg")

    def test_rejects_too_small_image(self):
        img = Image.new("L", (16, 16), 100)
        buf = io.BytesIO()
        img.save(buf, "JPEG")
        with pytest.raises(InvalidImageError):
            image_service.validate_and_sanitize(buf.getvalue(), "tiny.jpg")

    def test_rejects_dimension_overflow(self, monkeypatch):
        from app.core.config import get_settings

        monkeypatch.setattr(get_settings(), "max_image_dim", 64)
        with pytest.raises(ImageTooLargeError):
            image_service.validate_and_sanitize(make_jpeg_bytes(size=(200, 200)), "big.jpg")


class TestSniffing:
    def test_magic_detection(self):
        assert image_service._sniff_format(make_jpeg_bytes()[:16]) == "jpeg"
        assert image_service._sniff_format(make_png_bytes()[:16]) == "png"

    def test_rejects_text_magic(self):
        assert image_service._sniff_format(b"<?php echo 1; ?>") is None
