"""Backend settings (12-factor style; everything overridable via env)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="VI_", extra="ignore", protected_namespaces=("settings_",)
    )

    # App
    app_name: str = "AI Visual Inspection & Defect Detection Platform"
    environment: str = "development"
    api_prefix: str = "/api"

    # Database
    database_url: str = f"sqlite:///{REPO_ROOT / 'backend' / 'inspection.db'}"

    # Storage
    upload_dir: str = str(REPO_ROOT / "backend" / "storage" / "uploads")
    result_image_dir: str = str(REPO_ROOT / "backend" / "storage" / "results")
    max_upload_mb: int = 10
    max_image_dim: int = 4096

    # Model artifacts
    model_checkpoint: str = str(
        REPO_ROOT / "models" / "experiments" / "v1_resnet18" / "best.pth"
    )
    model_version: str = "v1.0.0"
    model_experiment: str = "v1_resnet18"

    # Inference backend: "torch" (training env) or "onnx" (torch-free
    # serving; ~4x smaller memory footprint — used in the cloud deployment).
    inference_backend: str = "torch"

    # Confidence policy (documented in docs/severity.md)
    confidence_high: float = 0.80
    confidence_medium: float = 0.60
    # Predictions below this are still stored but flagged needs_review=True
    confidence_min: float = 0.35

    # CORS (frontend dev server)
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Single-container deployments (e.g. HF Spaces): if set to a built
    # frontend dist directory, the API also serves the SPA from the same
    # origin. Empty = API-only (compose/dev mode).
    frontend_dist: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
