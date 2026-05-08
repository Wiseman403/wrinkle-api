"""Cosmetic severity grading.

This is **informational, not clinical**. It is *not* Lemperle WSS, Merz,
or Glogau — those scales are defined by wrinkle depth and trained-rater
visual comparison, neither of which are measurable from a single RGB
photo. The grading here uses what we actually measure: density (mm of
wrinkle per cm² of skin) and longest-wrinkle length per zone.

Combination rule: per-zone grade = ``max(density_grade, length_grade)``.
This catches both failure modes — many short wrinkles (caught by
density) and a few prominent long ones (caught by length).

Whole-face severity is computed from the whole-face stats, **not** by
averaging per-zone grades. Empty zones (e.g. no glabella wrinkles)
would otherwise drag the score artificially low.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


# Threshold tables: (max_value, grade). The list is scanned in order;
# the first row whose ``max`` is >= the value sets the grade.
DENSITY_THRESHOLDS: List[Tuple[float, int]] = [
    (0.5, 0),
    (3.0, 1),
    (10.0, 2),
    (float("inf"), 3),
]
LONGEST_THRESHOLDS: List[Tuple[float, int]] = [
    (2.0, 0),
    (8.0, 1),
    (20.0, 2),
    (float("inf"), 3),
]
GRADE_LABELS: Dict[int, str] = {0: "None", 1: "Mild", 2: "Moderate", 3: "Pronounced"}
CONSUMER_LABELS: Dict[int, str] = {0: "Minimal", 1: "Low", 2: "Medium", 3: "High"}

#: Method tag stamped into the response payload. Bump if you change
#: thresholds — clients can branch on this to handle older fixtures.
METHOD_NAME: str = "cosmetic_density_and_length_v1"
METHOD_NOTE: str = (
    "Informational only. NOT Lemperle/Merz/Glogau. "
    "Defined in README -> Severity grading."
)


def _grade_from_thresholds(value: float, thresholds: List[Tuple[float, int]]) -> int:
    """Bucket ``value`` against an ordered (max, grade) table."""
    for cap, grade in thresholds:
        if value <= cap:
            return grade
    return thresholds[-1][1]


def compute_severity(zone_data: Dict[str, object]) -> Dict[str, object]:
    """Grade a single zone (or whole-face) metrics dict.

    Args:
        zone_data: Output of :func:`app.core.measure.measure_zone`. Must
            have ``density_mm_per_cm2`` and ``longest_wrinkle_mm`` keys.

    Returns:
        Dict with: ``grade`` (overall, 0-3), ``label``,
        ``density_grade``, ``length_grade``, ``driven_by`` (one of
        ``length`` / ``density`` / ``both`` / ``n/a``).
    """
    d = float(zone_data["density_mm_per_cm2"])  # type: ignore[arg-type]
    l = float(zone_data["longest_wrinkle_mm"])  # type: ignore[arg-type]
    dg = _grade_from_thresholds(d, DENSITY_THRESHOLDS)
    lg = _grade_from_thresholds(l, LONGEST_THRESHOLDS)
    grade = max(dg, lg)
    if lg > dg:
        driver = "length"
    elif dg > lg:
        driver = "density"
    else:
        driver = "both" if grade > 0 else "n/a"
    return {
        "grade": int(grade),
        "label": GRADE_LABELS[grade],
        "density_grade": int(dg),
        "length_grade": int(lg),
        "driven_by": driver,
    }


def build_severity_payload(
    per_zone_metrics: Dict[str, Dict[str, object]],
    whole_face_metrics: Dict[str, object],
) -> Dict[str, object]:
    """Assemble the full ``severity`` block for the response.

    Args:
        per_zone_metrics: ``{zone_name: measure_zone(...)}`` for every zone.
        whole_face_metrics: ``measure_zone(skel, skin_mask, ...)``.

    Returns:
        Dict matching :class:`app.schemas.Severity`. Threshold tables
        are echoed back as serializable lists so callers don't have to
        keep their own copy in sync. ``inf`` is encoded as the string
        ``"inf"`` to keep the JSON valid.
    """
    per_zone_severity = {z: compute_severity(d) for z, d in per_zone_metrics.items()}
    whole_face_severity = compute_severity(whole_face_metrics)
    consumer_label = CONSUMER_LABELS[int(whole_face_severity["grade"])]  # type: ignore[arg-type]

    def _serialize(thresholds: List[Tuple[float, int]]) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for cap, grade in thresholds:
            out.append({"max": "inf" if cap == float("inf") else cap, "grade": grade})
        return out

    return {
        "method": METHOD_NAME,
        "method_note": METHOD_NOTE,
        "thresholds": {
            "density_mm_per_cm2": _serialize(DENSITY_THRESHOLDS),
            "longest_mm": _serialize(LONGEST_THRESHOLDS),
        },
        "per_zone": per_zone_severity,
        "whole_face": whole_face_severity,
        "consumer_label": consumer_label,
    }
