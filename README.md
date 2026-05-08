# Wrinkle Detection API

A FastAPI service that takes the URL of a single selfie, runs a
MediaPipe + multi-scale Frangi pipeline over it, and returns per-zone
wrinkle metrics plus a cosmetic severity grade as JSON.

> ## ⚠️ Informational, not clinical.
>
> The severity grade returned by this service is **NOT** Lemperle WSS,
> **NOT** Merz, and **NOT** Glogau. Those clinical scales are defined
> by wrinkle depth and trained-rater visual comparison — neither is
> measurable from a single RGB photo.
>
> What this service measures is **density** (mm of wrinkle per cm² of
> skin) and **longest-wrinkle length** per zone, then buckets them
> into Mild / Moderate / Pronounced. Treat the output as a relative
> cosmetic indicator, not a medical assessment. Do not use it for
> diagnosis, treatment planning, or insurance/eligibility decisions.

---

## Quick start

```bash
# One-time setup
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the server (auto-downloads the FaceLandmarker bundle on first start)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# In another terminal
curl http://localhost:8000/health
# → {"status":"ok"}
```

OpenAPI docs are at `http://localhost:8000/docs`.

---

## Endpoints

| Method | Path       | Purpose |
|--------|------------|---------|
| `GET`  | `/`        | API banner: name, version, link to `/docs`. |
| `GET`  | `/health`  | Liveness probe. No model invocation; cheap. |
| `POST` | `/analyze` | Run the pipeline on the image at `image_url`. |

### `POST /analyze`

**Request body:**

```json
{ "image_url": "https://example.com/selfie.jpg" }
```

**Behavior:**

1. Validate the URL with `pydantic.HttpUrl` (must be `http`/`https`).
2. SSRF guard: refuse `localhost`, `*.local`, and any URL that resolves
   to a private / loopback / link-local / multicast / reserved address
   (IPv4 and IPv6).
3. Download with a configurable timeout and a 15 MB size cap. The cap
   is enforced both from the `Content-Length` header (cheap reject) and
   while streaming, because the header can lie.
4. Apply EXIF rotation, decode to RGB, run the pipeline.
5. Return JSON.

**Errors:** every failure mode has its own `error_code` so callers can
branch without parsing prose.

| HTTP | `error_code`                       | Cause |
|------|------------------------------------|-------|
| 422  | (FastAPI default)                  | Body fails Pydantic validation (malformed URL, unsupported scheme, missing field). |
| 400  | `URLBlockedError`                  | SSRF guard tripped (localhost / private IP / etc). |
| 400  | `URLUnreachableError`              | DNS failed, connection refused, non-2xx response, or a redirect (we don't follow). |
| 400  | `InvalidContentTypeError`          | Response `Content-Type` is not `image/*`. |
| 400  | `ImageTooLargeError`               | Image exceeds the size cap (header or stream). |
| 400  | `ImageDecodeError`                 | Bytes downloaded but not decodable as an image. |
| 400  | `NoFaceDetectedError`              | Pipeline saw zero faces. |
| 400  | `MultipleFacesDetectedError`       | Pipeline saw more than one face — distinct from `NoFaceDetectedError` so callers can prompt the user differently. |
| 500  | `Exception` (generic)              | Internal pipeline failure. The traceback is logged server-side; a generic message goes to the client. Tracebacks are never returned in the response. |

**Error envelope:**

```json
{ "detail": "refusing to fetch from non-public IP '192.168.1.1' ...",
  "error_code": "URLBlockedError" }
```

### Example

```bash
curl -X POST http://localhost:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"image_url": "https://example.com/selfie.jpg"}'
```

Trimmed response (truncated arrays — `per_wrinkle` is sorted longest-first
and typically holds 30-100 entries; the 14 `zones` keys are all present):

```json
{
  "image_size": { "width": 1463, "height": 1068 },
  "calibration": {
    "ipd_px": 243.21,
    "ipd_mm_assumed": 63.0,
    "mm_per_px": 0.25904
  },
  "frangi_sigmas_px": [0.8, 1.025, 1.314, 1.683, 2.156, 2.762],
  "hysteresis_threshold": { "low": 0.0208, "high": 0.0555 },
  "min_component_mm": 4.0,
  "zones": {
    "forehead": {
      "wrinkle_count": 2, "total_length_mm": 15.02,
      "zone_area_cm2": 22.98, "density_mm_per_cm2": 0.65,
      "longest_wrinkle_mm": 7.51, "component_lengths_mm": [7.51, 7.51]
    },
    "crows_feet_left": {
      "wrinkle_count": 10, "total_length_mm": 63.20,
      "zone_area_cm2": 4.43, "density_mm_per_cm2": 14.27,
      "longest_wrinkle_mm": 12.17,
      "component_lengths_mm": [12.17, 9.07, 7.77, "..."]
    },
    "...": "12 more zones"
  },
  "whole_face": {
    "wrinkle_count": 65,
    "total_length_mm": 407.45,
    "zone_area_cm2": 147.29,
    "density_mm_per_cm2": 2.77,
    "longest_wrinkle_mm": 19.43,
    "component_lengths_mm": [19.43, 16.06, 14.76, "..."]
  },
  "leftover_skin": {
    "wrinkle_count": 8, "total_length_mm": 38.08,
    "zone_area_cm2": 30.95, "density_mm_per_cm2": 1.23,
    "longest_wrinkle_mm": 6.48
  },
  "per_wrinkle": [
    { "id": 0, "zone": "under_eye_left", "length_mm": 19.43,
      "centroid_xy": [628, 564], "bbox_xywh": [591, 559, 75, 8] },
    { "id": 1, "zone": "cheek_left",     "length_mm": 16.06,
      "centroid_xy": [688, 658], "bbox_xywh": [682, 630, 23, 57] },
    { "id": 2, "zone": "jowl_right",     "length_mm": 14.76,
      "centroid_xy": [901, 834], "bbox_xywh": [887, 806, 30, 57] },
    "... ~62 more"
  ],
  "severity": {
    "method": "cosmetic_density_and_length_v1",
    "method_note": "Informational only. NOT Lemperle/Merz/Glogau. ...",
    "thresholds": {
      "density_mm_per_cm2": [
        {"max": 0.5,  "grade": 0}, {"max": 3.0, "grade": 1},
        {"max": 10.0, "grade": 2}, {"max": "inf", "grade": 3}
      ],
      "longest_mm": [
        {"max": 2.0,  "grade": 0}, {"max": 8.0, "grade": 1},
        {"max": 20.0, "grade": 2}, {"max": "inf", "grade": 3}
      ]
    },
    "per_zone": { "...": "all 14 zones, each {grade, label, density_grade, length_grade, driven_by}" },
    "whole_face": {
      "grade": 2, "label": "Moderate",
      "density_grade": 1, "length_grade": 2, "driven_by": "length"
    },
    "consumer_label": "Medium"
  },
  "processing_time_ms": 1180
}
```

---

## Configuration

All settings are read from environment variables (or a `.env` file in
the working directory) with the prefix `WRINKLE_`. Algorithm constants
(Frangi sigmas, hysteresis floors, severity thresholds, exclusion
margins) are **not** exposed as settings — they were tuned by hand on
real selfies and live next to the code in `app.core`.

| Variable                       | Default                                          | What it controls |
|--------------------------------|--------------------------------------------------|------------------|
| `WRINKLE_API_NAME`             | `"Wrinkle Detection API"`                        | Surfaces at `/` and in OpenAPI. |
| `WRINKLE_API_VERSION`          | `"1.0.0"`                                        | Surfaces at `/` and in OpenAPI. |
| `WRINKLE_MODEL_PATH`           | `models/face_landmarker.task`                    | Where the FaceLandmarker bundle lives. Auto-downloaded if missing. |
| `WRINKLE_MODEL_URL`            | Google CDN URL (see `app/config.py`)             | Source for the auto-download. |
| `WRINKLE_DOWNLOAD_TIMEOUT_S`   | `10.0`                                           | Total HTTP timeout for fetching the user's image. The startup model download uses 6× this with a 60 s floor. |
| `WRINKLE_MAX_IMAGE_BYTES`      | `15728640` (15 MB)                               | Hard cap on downloaded image size; enforced via `Content-Length` and while streaming. |
| `WRINKLE_LOG_LEVEL`            | `INFO`                                           | stdlib logging level name. `DEBUG` adds Frangi/threshold telemetry. |
| `WRINKLE_BLOCK_PRIVATE_IPS`    | `true`                                           | SSRF guard. Set `false` only inside trusted environments where you intentionally fetch from internal hosts. |

`.env` example:

```env
WRINKLE_LOG_LEVEL=DEBUG
WRINKLE_MAX_IMAGE_BYTES=20971520
WRINKLE_DOWNLOAD_TIMEOUT_S=15
```

---

## Local development

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Running the test suite (no network, no MediaPipe init):

```bash
pip install pytest
python -m pytest tests/ -v
```

The fixtures in `tests/test_smoke.py` stub the FaceLandmarker so the
suite runs in ~15 s without downloading the 3.8 MB model bundle.

---

## Docker

```bash
docker build -t wrinkle-api .
docker run --rm -p 8000:8000 wrinkle-api
```

The image:

- is multi-stage (slim `python:3.11-slim-bookworm`),
- pre-downloads `face_landmarker.task` at build time so the first
  request after deploy isn't blocked on a cold network call,
- runs as non-root user `app` (UID 10001),
- installs only `libglib2.0-0` and `libgomp1` at runtime
  (`opencv-python-headless` avoids most system libs),
- exposes port 8000 and runs `uvicorn` directly.

To override settings:

```bash
docker run --rm -p 8000:8000 \
  -e WRINKLE_LOG_LEVEL=DEBUG \
  -e WRINKLE_MAX_IMAGE_BYTES=20971520 \
  -e WRINKLE_BLOCK_PRIVATE_IPS=true \
  wrinkle-api
```

For horizontal scale, run multiple replicas behind a load balancer.
The MVP serializes `FaceLandmarker.detect(...)` behind a
`threading.Lock` inside one process, so adding `--workers` to uvicorn
is the right knob for in-process parallelism only if you replace the
lock with a detector pool (out of scope for this MVP — see
`app/services/wrinkle_detector.py` for a one-paragraph note).

The container's `CMD` binds to `${PORT:-8000}` so the same image runs
unchanged on platforms that inject `$PORT` (DigitalOcean App Platform,
Cloud Run, Render, Fly.io, Railway).

---

## Deploy on DigitalOcean App Platform

A ready-to-use spec lives in [`.do/app.yaml`](.do/app.yaml). It is
configured for **Professional XS** (1 GB / 1 dedicated vCPU,
~$12/month at time of writing), single instance, auto-deploy on push
to `main`.

**Before first deploy:**

1. Edit `.do/app.yaml` and set the `github.repo` field to your repo
   slug (`<username>/<repo>`). The placeholder `YOUR_GITHUB_USERNAME`
   is intentional — the spec will refuse to deploy until you change
   it.
2. Push the repo to GitHub.

**Deploy via the web UI:**

1. [cloud.digitalocean.com/apps](https://cloud.digitalocean.com/apps)
   → **Create App** → **GitHub** → authorize → select your repo and
   branch.
2. App Platform auto-detects `.do/app.yaml` and pre-fills the form.
   Confirm: Resource size = **Professional XS**, HTTP port = **8000**,
   Health check path = **/health**.
3. **Create Resources**.

**Or via the CLI:**

```bash
doctl auth init
doctl apps create --spec .do/app.yaml
```

**First build takes ~8-12 minutes** (mediapipe wheels are large and
the build pre-downloads the FaceLandmarker bundle). Subsequent
deploys take ~3-5 minutes thanks to layer caching.

**Smoke-test the deployed app:**

```bash
APP=https://your-app-xxxxx.ondigitalocean.app

curl $APP/health
# → {"status":"ok"}

curl -X POST $APP/analyze \
  -H 'Content-Type: application/json' \
  -d '{"image_url":"https://example.com/selfie.jpg"}'
```

**Things to know:**

- **Don't disable `WRINKLE_BLOCK_PRIVATE_IPS`** in production. App
  Platform's runtime exposes services on private/link-local addresses
  (the metadata service is at `169.254.169.254`). The default `True`
  blocks them; the spec re-asserts it explicitly.
- **The `initial_delay_seconds: 60`** on the health check matters.
  MediaPipe init can take a few seconds even on a dedicated vCPU; a
  shorter delay can boot-loop the container before it's ready.
- **Don't scale `instance_count > 1` and try to share state.** The
  detector lock is process-local; multiple replicas just give you N
  independent detectors, which is fine for stateless `/analyze`
  traffic.

---

## Project layout

```
wrinkle_api/
├── app/
│   ├── __init__.py
│   ├── main.py                 FastAPI app, lifespan, exception handlers, routes
│   ├── config.py               pydantic-settings (everything WRINKLE_* above)
│   ├── schemas.py              Pydantic v2 request/response models (drives /docs)
│   ├── services/
│   │   ├── image_loader.py     URL → numpy RGB array; SSRF guard, EXIF, size/type validation
│   │   └── wrinkle_detector.py orchestrator + FaceLandmarker singleton + threading.Lock
│   └── core/
│       ├── landmarks.py        FACE_OVAL, eye/brow/lip indices, anchors, ZONE_PRIORITY,
│       │                       ZONE_ITERATION_ORDER, IPD constants
│       ├── masks.py            face mask, skin mask, 14 zone masks + overlap resolution
│       ├── preprocess.py       L* → CLAHE → bilateral → mild unsharp
│       ├── ridges.py           Frangi (multi-scale) → hysteresis → skeleton → branch-split
│       ├── measure.py          per-zone metrics, per-wrinkle entries, leftover-skin mask
│       └── severity.py         cosmetic grading (informational, not clinical)
├── models/
│   └── face_landmarker.task    auto-downloaded on first startup if missing
├── tests/
│   └── test_smoke.py           22 tests; no network; ~15 s
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .dockerignore
└── README.md
```

---

## Severity grading

| Driver           | Grade 0 (None) | Grade 1 (Mild) | Grade 2 (Moderate) | Grade 3 (Pronounced) |
|------------------|----------------|----------------|--------------------|----------------------|
| Density (mm/cm²) | ≤ 0.5          | ≤ 3.0          | ≤ 10.0             | > 10.0               |
| Longest (mm)     | ≤ 2.0          | ≤ 8.0          | ≤ 20.0             | > 20.0               |

**Combination rule:** per-zone grade = `max(density_grade, length_grade)`.
This catches both failure modes — many short wrinkles (caught by density)
and a few prominent long ones (caught by length). The `driven_by` field
tells you which of the two won (or `"both"` when they tied at the same
non-zero grade).

**Whole-face grade** is computed from whole-face stats, **not** by
averaging per-zone grades — empty zones would otherwise drag the score
artificially low.

**Whole-face → consumer label:** 0 → Minimal · 1 → Low · 2 → Medium · 3 → High.

The full threshold table is echoed back in every response under
`severity.thresholds`, so clients don't have to keep their own copy in
sync. A label change will bump `severity.method` (currently
`cosmetic_density_and_length_v1`) so callers can branch on it.

> Reminder: this grade is **informational, not clinical.** It is *not*
> Lemperle WSS, *not* Merz, *not* Glogau. See the disclaimer at the top
> of this README.

---

## Accuracy & known limits

These limits are inherent to single-RGB-photo wrinkle analysis and
were called out in the source notebook. They are not bugs.

- **Front-flat lighting** (ring light, on-axis flash) under-reveals
  wrinkles → scores low.
- **Side lighting** over-reveals wrinkles → scores high.
- **Smiling / squinting** selfies → crow's feet read as static
  wrinkles even when they're really expression lines. The pipeline
  cannot distinguish dynamic from static wrinkles from a single
  frame.
- **Beauty-filtered photos** → wrinkles are pre-erased before they
  reach the pipeline; nothing to detect.
- **Hair on the forehead** may leak into the forehead zone — there is
  no hair segmenter in the MVP.
- **IPD calibration** assumes a 63 mm adult inter-pupillary distance.
  Real adults span roughly 54-68 mm, so per-image absolute mm/cm²
  values carry up to ±10% calibration error. This is fine for a
  relative cosmetic score; do not treat the mm values as
  measurements.

For best results: even, slightly off-axis lighting; near-frontal pose;
neutral expression; no beauty filter; no hair on the forehead; face
fills most of the frame.

---

## Differences from the source notebook

What a consumer of the JSON would notice, vs. running the original
`wrinkle_detection_v7_1.ipynb`:

- **`image_path` removed.** The notebook's Colab-local path is
  meaningless over HTTP.
- **`processing_time_ms` added.** Wall-clock pipeline time, excluding
  HTTP I/O.
- **No overlay PNG.** This is a JSON API; the visualization step from
  the notebook (`*_wrinkle_overlay.png`) is intentionally not
  produced. If you need that, render it client-side from the
  `per_wrinkle` bounding boxes and the zone names.
- **Multi-face image now returns 400** with `error_code:
  "MultipleFacesDetectedError"`, distinct from `NoFaceDetectedError`.
  The notebook's bare `assert` was promoted to a typed exception.
- **Numeric output matches the notebook within ~1%** on the
  reference selfie (`my_selfie.png`):

  | metric                   | notebook | API   | delta |
  |--------------------------|---------:|------:|------:|
  | whole-face wrinkle count |       64 |    65 |    +1 |
  | total length (mm)        |   402.53 | 407.45| +1.2% |
  | density (mm/cm²)         |     2.73 |  2.77 | +1.5% |
  | longest wrinkle (mm)     |    19.43 | 19.43 | exact |
  | severity grade           |        2 |     2 | exact |
  | consumer label           |   Medium | Medium| exact |

  The residual delta is from upstream library version drift between
  Colab's pinned versions and the pinned-major versions in
  `requirements.txt` (numpy, scikit-image, opencv all release minor
  numerical/algorithmic tweaks regularly). The algorithm constants
  and call shapes are byte-for-byte preserved from the notebook; the
  delta is not from any code change in the port. If you need
  bit-for-bit reproducibility against a specific notebook run, pin
  every dependency to the exact version in that environment.

- All matplotlib visualizations, stdout tables, and shell magics from
  the notebook are removed. The pipeline is now a single function
  call (`WrinkleDetector.analyze`) that returns a dict.

---

## License & attribution

- MediaPipe FaceLandmarker model bundle: see Google's [model card]
  (https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).
- Frangi vesselness filter: implementation from [scikit-image]
  (https://scikit-image.org/).

The source notebook (`wrinkle_detection_v7_1.ipynb`) was the working
reference for every algorithm constant in this codebase. When in
doubt about why a magic number is what it is, consult the inline
comments — they were preserved during the port.
