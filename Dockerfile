# syntax=docker/dockerfile:1.6
#
# Multi-stage build:
#  1. `builder` — installs Python deps into a clean prefix and pre-downloads
#     the FaceLandmarker bundle. Pip wheels and build tooling stay here.
#  2. `runtime` — copies only the installed site-packages and the model
#     bundle into a slim base. Runs as a non-root user.
#
# Image size targets ~1.0-1.4 GB depending on platform; mediapipe and
# opencv pull in most of that.

# -----------------------------------------------------------------------------
# Stage 1: builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build-time deps for wheels that may not have manylinux for our arch
# (mediapipe and scikit-image both publish wheels, but keep curl + ca for
# the model fetch and a minimal toolchain in case a wheel is missing).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# Install into a relocatable prefix so we can copy it into the runtime
# stage without dragging pip / cache along.
RUN pip install --prefix=/install -r requirements.txt

# Pre-download the FaceLandmarker bundle so the first request after deploy
# isn't blocked on a cold network call to Google's CDN.
RUN mkdir -p /install/models \
 && curl -fsSL \
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
        -o /install/models/face_landmarker.task \
 && ls -la /install/models/face_landmarker.task


# -----------------------------------------------------------------------------
# Stage 2: runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# opencv-python-headless avoids most system libs, but mediapipe and a few
# scikit-image components still need libglib2.0 and libgomp at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Non-root user. UID 10001 is conventional for "service accounts" and stays
# clear of any host UIDs.
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home /app --shell /usr/sbin/nologin app

# Copy installed Python packages from the builder stage.
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy the application source (kept small by .dockerignore).
COPY --chown=app:app app/ ./app/

# Move the pre-downloaded model into the location app/config.py expects
# (PROJECT_ROOT/models/face_landmarker.task).
COPY --from=builder /install/models/face_landmarker.task ./models/face_landmarker.task
RUN chown -R app:app /app/models

USER app

EXPOSE 8000

# Single-process uvicorn. Run multiple replicas for horizontal scale; the
# in-process serialization on the FaceLandmarker lock means adding workers
# inside one container only helps if you parallelize at the orchestrator
# level (which would require a detector pool — not in scope for the MVP).
#
# We bind to ${PORT:-8000} so the same image works on:
#   - local docker run (PORT unset → falls back to 8000),
#   - DigitalOcean App Platform / Cloud Run / Render / Fly.io / Railway
#     (PORT injected by the platform and the gateway routes traffic there).
# `sh -c` is required because Docker's exec form ([...]) does not perform
# variable expansion.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
