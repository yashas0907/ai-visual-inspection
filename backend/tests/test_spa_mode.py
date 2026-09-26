"""Tests for the single-origin SPA deployment mode (VI_FRONTEND_DIST).

Regression guard: Path("") normalizes to Path(".") — the CWD — so the
frontend-dist check must never activate from an unset/empty setting (this
once broke every API test via a bad /assets mount).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


MINIMAL_INDEX = """<!doctype html>
<html><head><title>SPA</title></head>
<body><div id="root"></div><script src="/assets/app.js"></script></body></html>"""


@pytest.fixture()
def spa_dist(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(MINIMAL_INDEX, encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log('x')", encoding="utf-8")
    return dist


@pytest.fixture()
def spa_client(fake_settings, spa_dist, monkeypatch):
    import app.core.config as cfgmod

    cfgmod.get_settings.cache_clear()
    monkeypatch.setenv("VI_FRONTEND_DIST", str(spa_dist))

    import app.services.model_service as ms

    ms._predictor = None
    ms._gradcam = None
    import app.db.session as dbsess

    dbsess._engine = None
    dbsess._SessionLocal = None
    from app.db.session import Base

    Base.metadata.create_all(bind=dbsess.get_engine())

    from app.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c

    cfgmod.get_settings.cache_clear()
    dbsess._engine = None
    dbsess._SessionLocal = None
    ms._predictor = None
    ms._gradcam = None


@pytest.fixture()
def api_only_client(fake_settings, monkeypatch):
    """No VI_FRONTEND_DIST set at all (the default state)."""
    import app.core.config as cfgmod

    cfgmod.get_settings.cache_clear()
    monkeypatch.delenv("VI_FRONTEND_DIST", raising=False)

    import app.services.model_service as ms

    ms._predictor = None
    ms._gradcam = None
    import app.db.session as dbsess

    dbsess._engine = None
    dbsess._SessionLocal = None
    from app.db.session import Base

    Base.metadata.create_all(bind=dbsess.get_engine())

    from app.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c

    cfgmod.get_settings.cache_clear()
    dbsess._engine = None
    dbsess._SessionLocal = None
    ms._predictor = None
    ms._gradcam = None


class TestSpaMode:
    def test_index_served(self, spa_client: TestClient):
        r = spa_client.get("/")
        assert r.status_code == 200
        assert 'id="root"' in r.text

    def test_deep_link_falls_back_to_index(self, spa_client: TestClient):
        r = spa_client.get("/history")
        assert r.status_code == 200
        assert 'id="root"' in r.text  # SPA handles routing client-side

    def test_assets_served(self, spa_client: TestClient):
        r = spa_client.get("/assets/app.js")
        assert r.status_code == 200
        assert "javascript" in r.headers["content-type"]

    def test_api_still_works_in_spa_mode(self, spa_client: TestClient):
        r = spa_client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_static_file_no_traversal(self, spa_client: TestClient):
        # a crafted path must not escape the dist directory
        r = spa_client.get("/..%2F..%2Falembic.ini")
        assert r.status_code == 200
        assert "sqlalchemy" not in r.text  # falls back to index.html, not the file


class TestApiOnlyMode:
    def test_unset_frontend_dist_does_not_break_app(self, api_only_client: TestClient):
        """Regression: empty setting must not activate SPA mode."""
        r = api_only_client.get("/api/health")
        assert r.status_code == 200

    def test_root_404s_without_spa(self, api_only_client: TestClient):
        r = api_only_client.get("/")
        assert r.status_code == 404  # no SPA configured -> nothing at /
