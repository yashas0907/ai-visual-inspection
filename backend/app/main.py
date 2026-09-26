"""FastAPI application factory.

Startup behavior (lifespan):
- create tables (dev convenience; production uses alembic migrations)
- attempt model load; the app still boots without the checkpoint so that
  /api/health and /api/model/info work while model artifacts are absent
  (degraded state), which keeps Docker builds testable.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import inspections, platform, storage
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging, get_logger
from app.db.session import Base, get_engine, get_session_local
from app.services import model_service, registry_service

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=get_engine())
    # Best-effort model load (degraded mode if checkpoint missing).
    try:
        model_service.load_model()
    except FileNotFoundError as exc:
        logger.warning("startup without model: %s", exc)
    # Register active version metadata (needs DB, after create_all).
    try:
        with get_session_local()() as db:
            registry_service.register_active_version(db)
    except Exception:
        logger.exception("failed to register model version metadata")
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()

    app = FastAPI(
        title=settings.app_name,
        version=settings.model_version,
        description=(
            "Industrial visual inspection platform for hot-rolled steel strip "
            "surface defects (NEU dataset). Classification + Grad-CAM "
            "explainability + rule-based severity + human-in-the-loop feedback."
        ),
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(inspections.router, prefix=settings.api_prefix)
    app.include_router(platform.router, prefix=settings.api_prefix)
    app.include_router(storage.router, prefix=settings.api_prefix)

    # ----- single-origin SPA mode (deployments) ----------------------------
    # When VI_FRONTEND_DIST points at a built frontend, serve it from the
    # same origin as the API: no CORS, no second container. Used by the
    # single-container deployment (HF Spaces / any host).
    # NOTE: Path("") normalizes to Path(".") (the CWD!), so the setting must
    # be explicitly non-empty — is_dir() alone is not a safe truthiness test.
    frontend_dist = Path(settings.frontend_dist) if settings.frontend_dist else None
    if frontend_dist and (frontend_dist / "index.html").is_file():
        from fastapi.staticfiles import StaticFiles

        assets_dir = frontend_dist / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="spa-assets")

        @app.get("/{spa_path:path}", include_in_schema=False)
        async def spa(spa_path: str):
            # API routes are matched first (registered above), so anything
            # reaching here that isn't a known file falls back to index.html.
            candidate = (frontend_dist / spa_path).resolve()
            if spa_path and candidate.is_file() and str(candidate).startswith(str(frontend_dist.resolve())):
                return FileResponse(candidate)
            return FileResponse(frontend_dist / "index.html")

    # ----- exception handling ----------------------------------------------

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.__class__.__name__, "detail": exc.detail},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "HTTPError", "detail": exc.detail},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": "ValidationError", "detail": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception("unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "InternalServerError", "detail": "an unexpected error occurred"},
        )

    return app


app = create_app()
