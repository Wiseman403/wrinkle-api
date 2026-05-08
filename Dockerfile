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
    PIP_NO_CACHE_DIR=1 \
    # matplotlib gets imported transitively by mediapipe / scikit-image
    # and tries to write its config cache under $HOME/.config/matplotlib.
    # Our non-root `app` user can't write to /app (created by WORKDIR
    # while still owned by root), so it would fall back to a fresh
    # tmpdir on every startup and emit a warning. Pointing MPLCONFIGDIR
    # at /tmp explicitly silences the warning and makes the cache
    # reusable across imports within one container lifetime.
    MPLCONFIGDIR=/tmp/matplotlib_cache

# Runtime system libs needed by the Python wheels.
#
# Why each one is here:
#   libglib2.0-0  pulled in by opencv and mediapipe
#   libgomp1      OpenMP runtime, used by scikit-image / numpy / opencv
#   libgl1        mediapipe declares a hard dep on `opencv-contrib-python`
#                 (the *non*-headless variant), which gets pulled in
#                 alongside our explicit `opencv-python-headless`. The
#                 contrib build's `cv2.so` links libGL even when no GUI
#                 is used.
#   libsm6, libxext6 commonly required by opencv's GUI build alongside libgl1.
#   libegl1       mediapipe >= 0.10.20 splits its native library into a
#                 separately-loaded c-bindings .so that links libEGL even
#                 when running CPU-only with no GPU.
#   libgles2      same — the c-bindings .so also links libGLESv2.
#                 Without these two, FaceLandmarker.create_from_options()
#                 raises "OSError: libGLESv2.so.2: cannot open shared
#                 object file" at lifespan startup.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
        libsm6 \
        libxext6 \
        libegl1 \
        libgles2 \
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
