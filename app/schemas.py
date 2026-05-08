"""Pydantic v2 request/response models.

These models drive both runtime validation and the OpenAPI schema served
at ``/docs``. Field shapes mirror the JSON the notebook builds in its
``results`` dict, with two intentional changes:

1. ``image_path`` is dropped — the notebook's Colab-local path is
   meaningless over HTTP.
2. ``processing_time_ms`` is added for observability.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


# --- Request --------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Body for ``POST /analyze``.

    The URL is validated by ``pydantic.HttpUrl`` (scheme must be http/https,
    well-formed netloc/path). SSRF and reachability checks happen later in
    ``services.image_loader`` — they cannot be expressed at the schema layer.
    """

    image_url: HttpUrl = Field(
        ...,
        description="HTTP/HTTPS URL of the selfie to analyze.",
        examples=["https://example.com/selfie.jpg"],
    )


# --- Sub-models reused inside the response -------------------------------


class ImageSize(BaseModel):
    width: int
    height: int


class Calibration(BaseModel):
    """Pixel→mm calibration derived from inter-pupillary distance."""

    ipd_px: float = Field(..., description="Inter-pupillary distance in pixels.")
    ipd_mm_assumed: float = Field(
        ..., description="Assumed adult IPD in millimeters (63 mm)."
    )
    mm_per_px: float = Field(..., description="Scale factor (mm per pixel).")


class HysteresisThreshold(BaseModel):
    low: float
    high: float


class ZoneMetrics(BaseModel):
    """Aggregate metrics for a single zone (or whole-face / leftover-skin)."""

    wrinkle_count: int
    total_length_mm: float
    zone_area_cm2: float
    density_mm_per_cm2: float
    longest_wrinkle_mm: float
    component_lengths_mm: List[float] = Field(
        default_factory=list,
        description="Lengths of accepted wrinkle components, sorted desc.",
    )


class WrinkleEntry(BaseModel):
    """A single wrinkle component that survived filtering.

    Coordinates are in image pixel space (origin top-left, x→right, y→down).
    """

    id: int
    zone: str
    length_mm: float
    centroid_xy: List[int] = Field(..., min_length=2, max_length=2)
    bbox_xywh: List[int] = Field(..., min_length=4, max_length=4)


class SeverityScore(BaseModel):
    """Cosmetic grade for one zone (or whole face).

    The ``grade`` field is the larger of ``density_grade`` and
    ``length_grade``; ``driven_by`` reports which one drove the call.
    """

    grade: int = Field(..., ge=0, le=3)
    label: str
    density_grade: int = Field(..., ge=0, le=3)
    length_grade: int = Field(..., ge=0, le=3)
    driven_by: str


class Severity(BaseModel):
    """Top-level severity payload.

    The threshold table is echoed back in the response so callers don't
    have to keep their own copy in sync. ``method_note`` carries the
    'informational, not clinical' disclaimer; do not strip it from
    downstream UIs.
    """

    method: str
    method_note: str
    thresholds: Dict[str, List[Dict[str, object]]]
    per_zone: Dict[str, SeverityScore]
    whole_face: SeverityScore
    consumer_label: str


# --- Response -------------------------------------------------------------


class AnalyzeResponse(BaseModel):
    """Body for a successful ``POST /analyze``."""

    # Permissive on extra fields so we can ship additive changes without a
    # breaking schema bump for clients that pin to this model.
    model_config = ConfigDict(extra="ignore")

    image_size: ImageSize
    calibration: Calibration
    frangi_sigmas_px: List[float]
    hysteresis_threshold: HysteresisThreshold
    min_component_mm: float
    zones: Dict[str, ZoneMetrics] = Field(
        ...,
        description=(
            "All 14 zones keyed by name. See README for the anatomical "
            "definitions and overlap-resolution priority."
        ),
    )
    whole_face: ZoneMetrics
    leftover_skin: ZoneMetrics = Field(
        ...,
        description=(
            "Skeleton pixels in skin but outside every zone. A large value "
            "means wrinkles are landing in regions the zone map doesn't cover."
        ),
    )
    per_wrinkle: List[WrinkleEntry] = Field(
        ...,
        description="Per-component entries, sorted by length (longest first).",
    )
    severity: Severity
    processing_time_ms: int = Field(
        ..., description="Wall-clock pipeline time, excluding HTTP I/O."
    )


# --- Misc endpoints -------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"


class RootResponse(BaseModel):
    name: str
    version: str
    docs: str


class ErrorResponse(BaseModel):
    """Standard error envelope used for 400/500 paths.

    ``detail`` is a short human message safe to surface to end users.
    ``error_code`` is a stable machine-readable tag (the exception class
    name) so clients can branch without parsing prose.
    """

    detail: str
    error_code: str
