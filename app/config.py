"""Runtime configuration for the wrinkle-detection API.

Settings are read from environment variables (or a ``.env`` file) via
``pydantic-settings``. Defaults are tuned for a single-process container
that fronts an inexpensive CPU pipeline.

All knobs that have legitimate per-deployment variance live here; algorithm
constants tuned on real selfies (Frangi sigmas, hysteresis floors, zone
margins, severity thresholds) live next to the code that uses them in
``app.core`` and are NOT exposed as settings.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root = parent of `app/`. Used to anchor the default model path so
# the service works regardless of where uvicorn is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings.

    Attributes:
        api_name: Human-readable API name surfaced at ``/`` and in OpenAPI.
        api_version: Semver string for the deployed build.
        model_path: On-disk path to the MediaPipe FaceLandmarker ``.task``
            bundle. Auto-downloaded on startup if missing.
        model_url: Source URL for the model bundle when auto-downloading.
        download_timeout_s: HTTP timeout for fetching the user-supplied
            image (and the model bundle on first start).
        max_image_bytes: Hard cap on downloaded image size. Enforced both
            from the ``Content-Length`` header (cheap reject) and while
            streaming (the header can lie).
        log_level: stdlib ``logging`` level name. ``INFO`` is appropriate
            for production; ``DEBUG`` adds Frangi/threshold telemetry.
        block_private_ips: SSRF guard. When True (default), reject URLs
            whose resolved IP is private, loopback, or link-local. Only
            disable in trusted environments where you intentionally want
            to fetch from internal hosts.
    """

    model_config = SettingsConfigDict(
        env_prefix="WRINKLE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_*` is a pydantic-protected namespace by default; we use it
        # for the FaceLandmarker bundle path/url, so silence the warning.
        protected_namespaces=(),
    )

    api_name: str = "Wrinkle Detection API"
    api_version: str = "1.0.0"

    model_path: Path = _PROJECT_ROOT / "models" / "face_landmarker.task"
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )

    download_timeout_s: float = 10.0
    max_image_bytes: int = 15 * 1024 * 1024  # 15 MB

    log_level: str = "INFO"
    block_private_ips: bool = True


# Module-level singleton. Importing `settings` anywhere returns the same
# object; tests that need to override values should construct a fresh
# `Settings(...)` and inject it via FastAPI's dependency-override hooks.
settings = Settings()
