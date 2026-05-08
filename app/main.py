"""FastAPI app, lifespan, and HTTP route handlers.

The lifespan ensures:

1. The FaceLandmarker ``.task`` bundle exists (auto-download on first
   start), then
2. A single :class:`~app.services.wrinkle_detector.WrinkleDetector` is
   constructed and stored on ``app.state``.

On shutdown the detector is closed cleanly so MediaPipe's native
resources are released — leaking instances across uvicorn reloads will
eventually exhaust handles.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import schemas
from .config import settings
from .services.image_loader import (
    ImageDecodeError,
    ImageLoaderError,
    ImageTooLargeError,
    InvalidContentTypeError,
    URLBlockedError,
    URLUnreachableError,
    fetch_image,
)
from .services.wrinkle_detector import (
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    WrinkleDetector,
    WrinkleDetectorError,
)


# --- Logging --------------------------------------------------------------
# Configure once, in main.py. Uvicorn has its own access logger; this
# logger covers application-level events (request outcomes, model load,
# pipeline failures).

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("wrinkle_api")


# --- Model bootstrap (lifespan) ------------------------------------------


def _download_model_sync(url: str, dest: str, timeout_s: float) -> None:
    """Synchronously fetch the FaceLandmarker bundle to ``dest``.

    Sync rather than async because lifespan startup is single-threaded
    and there's no concurrency to gain from awaiting.
    """
    logger.info("downloading FaceLandmarker bundle from %s", url)
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
    logger.info("FaceLandmarker bundle saved to %s", dest)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: ensure model + build detector. Shutdown: close detector."""
    model_path = settings.model_path
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # Run the blocking download in a worker thread so we don't tie
        # up the event loop. Using a generous timeout — 10s default is
        # for image downloads, the model bundle is ~3.8 MB and on a
        # cold network can take longer.
        await asyncio.to_thread(
            _download_model_sync,
            settings.model_url,
            str(model_path),
            max(60.0, settings.download_timeout_s * 6),
        )
    else:
        logger.info("FaceLandmarker bundle already present at %s", model_path)

    # Build the detector singleton (MediaPipe init is sync + slow — punt
    # to a worker thread so the event loop is responsive during boot).
    detector = await asyncio.to_thread(WrinkleDetector, model_path)
    app.state.detector = detector
    try:
        yield
    finally:
        await asyncio.to_thread(detector.close)


# --- App & exception handlers --------------------------------------------


app = FastAPI(
    title=settings.api_name,
    version=settings.api_version,
    description=(
        "Detects static wrinkles and fine lines from a single selfie URL "
        "and returns per-zone metrics plus a cosmetic severity grade. "
        "The grade is informational, not clinical."
    ),
    lifespan=lifespan,
)


def _error_response(exc: Exception, status: int) -> JSONResponse:
    """Render an exception as a uniform error envelope.

    The exception's class name doubles as a stable ``error_code`` tag.
    """
    payload = schemas.ErrorResponse(
        detail=str(exc) or exc.__class__.__name__,
        error_code=exc.__class__.__name__,
    ).model_dump()
    return JSONResponse(status_code=status, content=payload)


@app.exception_handler(ImageLoaderError)
async def _handle_image_loader(request: Request, exc: ImageLoaderError) -> JSONResponse:
    # All image-loader errors are caller-fixable → 400.
    logger.info("image-loader error: %s: %s", exc.__class__.__name__, exc)
    return _error_response(exc, status=400)


@app.exception_handler(WrinkleDetectorError)
async def _handle_detector(request: Request, exc: WrinkleDetectorError) -> JSONResponse:
    logger.info("detector error: %s: %s", exc.__class__.__name__, exc)
    return _error_response(exc, status=400)


@app.exception_handler(Exception)
async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    # Anything that escapes the typed handlers above is a server bug.
    # NEVER leak the traceback to the client; log it server-side and
    # return a generic message.
    logger.exception("unexpected pipeline failure")
    return _error_response(
        Exception("internal pipeline failure"),  # generic message
        status=500,
    )


# --- Routes ---------------------------------------------------------------


@app.get(
    "/",
    response_model=schemas.RootResponse,
    summary="API banner",
)
async def root() -> schemas.RootResponse:
    return schemas.RootResponse(
        name=settings.api_name,
        version=settings.api_version,
        docs="/docs",
    )


@app.get(
    "/health",
    response_model=schemas.HealthResponse,
    summary="Liveness probe",
)
async def health() -> schemas.HealthResponse:
    # Intentionally cheap — no model invocation, no file I/O. If this
    # endpoint is slow, your process is hung; restart it.
    return schemas.HealthResponse(status="ok")


@app.post(
    "/analyze",
    response_model=schemas.AnalyzeResponse,
    responses={
        400: {"model": schemas.ErrorResponse, "description": "Caller-side error"},
        500: {"model": schemas.ErrorResponse, "description": "Internal pipeline failure"},
    },
    summary="Run the wrinkle-detection pipeline on a remote selfie",
)
async def analyze(req: schemas.AnalyzeRequest, request: Request) -> schemas.AnalyzeResponse:
    """Download ``image_url`` and run the wrinkle pipeline.

    See README for the full response shape, the SSRF guard, and the
    cosmetic-grade disclaimer (informational, not clinical).
    """
    detector: WrinkleDetector = request.app.state.detector
    host = req.image_url.host or "<unknown>"
    t0 = time.perf_counter()

    # 1. HTTP fetch (async)
    img_rgb = await fetch_image(
        req.image_url,
        timeout_s=settings.download_timeout_s,
        max_size_bytes=settings.max_image_bytes,
        block_private_ips=settings.block_private_ips,
    )

    # 2. Synchronous CPU-bound pipeline → worker thread.
    # asyncio.to_thread releases the event loop so concurrent requests
    # don't block on each other (they will still serialize at the
    # detector lock, but Frangi/skeletonize run in parallel).
    result = await asyncio.to_thread(detector.analyze, img_rgb)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    result["processing_time_ms"] = elapsed_ms

    logger.info(
        "analyze ok host=%s width=%d height=%d ms=%d wrinkles=%d severity=%s",
        host,
        result["image_size"]["width"],
        result["image_size"]["height"],
        elapsed_ms,
        len(result["per_wrinkle"]),
        result["severity"]["consumer_label"],
    )
    # Pydantic validates the output shape on its way out, so a
    # backwards-incompatible code change becomes a 500 here rather than
    # a silently malformed payload to the client.
    return schemas.AnalyzeResponse.model_validate(result)
