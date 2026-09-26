"""Shared fixtures: temp DB, app with overridden model (tiny random checkpoint),
and synthetic test images. The real checkpoint is NOT required for API tests;
tests create a 1-epoch 'mock' checkpoint in a tmp model dir.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(REPO))  # ml package
sys.path.insert(0, str(BACKEND))  # app package

from ml.config import DataConfig, TrainConfig  # noqa: E402
from ml.training.model import build_model  # noqa: E402

from app.core.config import Settings  # noqa: E402


@pytest.fixture()
def fake_settings(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    uploads = storage / "uploads"
    results = storage / "results"
    uploads.mkdir(parents=True)
    results.mkdir(parents=True)

    # tiny model checkpoint (untrained resnet18, 1 channel)
    ckpt_dir = tmp_path / "models" / "v1_test"
    ckpt_dir.mkdir(parents=True)
    model = build_model(TrainConfig(pretrained=False), num_classes=len(DataConfig.classes))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": list(DataConfig.classes),
            "arch": "resnet18",
            "epoch": 1,
        },
        ckpt_dir / "best.pth",
    )

    monkeypatch.setenv("VI_DATABASE_URL", f"sqlite:///{tmp_path/'test.db'}")
    monkeypatch.setenv("VI_UPLOAD_DIR", str(uploads))
    monkeypatch.setenv("VI_MODEL_CHECKPOINT", str(ckpt_dir / "best.pth"))
    monkeypatch.setenv("VI_MODEL_VERSION", "v9.9-test")
    monkeypatch.setenv("VI_MODEL_EXPERIMENT", "v1_test")
    # force fresh singletons for settings, engine and model
    import app.core.config as cfgmod

    cfgmod.get_settings.cache_clear()
    import app.db.session as dbsess

    dbsess._engine = None
    dbsess._SessionLocal = None
    import app.services.model_service as ms

    ms._predictor = None
    ms._gradcam = None
    # create schema on the fresh engine for DB-level tests
    from app.db.session import Base

    Base.metadata.create_all(bind=dbsess.get_engine())
    yield uploads
    cfgmod.get_settings.cache_clear()
    dbsess._engine = None
    dbsess._SessionLocal = None
    ms._predictor = None
    ms._gradcam = None


@pytest.fixture()
def client(fake_settings):
    from fastapi.testclient import TestClient

    from app.db.session import get_engine, get_session_local
    from app.main import create_app

    app = create_app()
    # reset DB for isolation
    from app.db.session import Base

    Base.metadata.drop_all(bind=get_engine())
    Base.metadata.create_all(bind=get_engine())

    with TestClient(app) as c:
        yield c


def make_jpeg_bytes(size=(200, 200), mode="L", color=128) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, "JPEG")
    return buf.getvalue()


def make_png_bytes(size=(200, 200), color=(120, 130, 140)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()
