"""MediaPipe FaceMesh landmark indices used by the pipeline.

The MediaPipe Tasks ``FaceLandmarker`` returns 478 normalized landmarks
(468 mesh + 10 iris). All indices below are taken from the canonical
FaceMesh topology and have stable meanings across MediaPipe versions.

These are not magic numbers — each is a well-defined anatomical anchor.
Do NOT renumber them; downstream zone construction assumes specific
labels (e.g. "alar base", not "lateral nostril edge near the cheek").
"""

from __future__ import annotations

from typing import List, Tuple

# --- IPD anchors (used for px↔mm calibration) ----------------------------

#: Right iris center (image right = subject's left). MediaPipe iris landmarks
#: are indices 468..477; 468 is the right-iris center.
RIGHT_IRIS_CENTER: int = 468

#: Left iris center (image left = subject's right). 473 is the left-iris center.
LEFT_IRIS_CENTER: int = 473

#: Adult mean inter-pupillary distance, used as the calibration scalar.
#: Real adult IPD ranges roughly 54-68 mm; 63 mm is the common adult mean.
#: Per-image accuracy is therefore ~±10%, which is acceptable for a relative
#: cosmetic score and is what the original notebook assumed.
IPD_MM_ASSUMED: float = 63.0


# --- Eye contours --------------------------------------------------------
# Image perspective (left = viewer's left). Used as exclusion zones (we
# don't want Frangi firing on eyelashes or lid creases).

RIGHT_EYE: List[int] = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE: List[int] = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


# --- Eyebrow contours ----------------------------------------------------
# We dilate these heavily — MediaPipe traces the visible brow but stray
# brow hairs sit a few px above, and Frangi loves them.

RIGHT_EYEBROW: List[int] = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 156, 113]
LEFT_EYEBROW: List[int] = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285, 383, 342]


# --- Lips outer outline --------------------------------------------------

LIPS_OUTER: List[int] = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185,
]


# --- Nostril area --------------------------------------------------------
# Sparse — just the bridge/alar/columella points we want to mask out so
# nostril shadows don't read as ridges.

NOSTRIL_AREA: List[int] = [98, 97, 2, 326, 327, 64, 294]


# --- Face oval (ordered) -------------------------------------------------
# Defines the outer face polygon. Ordering matters: ``cv2.fillPoly``
# expects a continuous boundary trace, not a point cloud.

FACE_OVAL: List[int] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


# --- Single-point anchors used to position zone masks --------------------
# Each pair is (image-perspective-A, image-perspective-B). Construction
# code is responsible for ordering them by x-coordinate at runtime via
# ``order_by_x`` in masks.py — landmark index does NOT necessarily equal
# image-left/right (depends on the subject and camera).

#: Outer canthi (lateral corners of each eye). Used to seed crow's-feet boxes.
EYE_OUTER_A: int = 33
EYE_OUTER_B: int = 263

#: Alar base (lateral nostril). This is the actual upper terminus of the
#: nasolabial fold — earlier versions of the notebook used 129/358 (which
#: sit on the lip border) and produced strips that missed the fold entirely.
ALA_A: int = 64
ALA_B: int = 294

#: Mouth corners.
MOUTH_A: int = 61
MOUTH_B: int = 291

#: Inner eyebrow tips. Used as the upper anchor of the glabella box.
INNER_BROW_A: int = 55
INNER_BROW_B: int = 285

#: Lower-lid midpoints. Used as the upper anchor of the under-eye boxes.
LOWER_LID_A: int = 145
LOWER_LID_B: int = 374

#: Lateral cheekbone (zygomatic prominence). Used as the lateral edge of
#: the cheek and jowl boxes.
CHEEK_LATERAL_A: int = 234
CHEEK_LATERAL_B: int = 454

#: Chin tip (mental protuberance / pogonion).
CHIN_BOTTOM: int = 152

#: Bottom of the lower lip. Used as the upper anchor of the chin box.
LIP_BOTTOM: int = 17


# --- Zone names ----------------------------------------------------------
# Two named tuples so we never confuse "iteration order" with "overlap
# priority" — they are NOT the same. Iteration is alphabetical-ish for
# stable JSON output; priority is hand-tuned so a specific zone (e.g.
# glabella) wins over a coarse one (forehead) in the overlap pass.

#: Order in which zone masks are *built and emitted* in the response. This
#: is the order the JSON `zones` dict will be keyed in.
ZONE_ITERATION_ORDER: Tuple[str, ...] = (
    "forehead",
    "glabella",
    "crows_feet_left",
    "crows_feet_right",
    "under_eye_left",
    "under_eye_right",
    "nasolabial_left",
    "nasolabial_right",
    "perioral",
    "cheek_left",
    "cheek_right",
    "chin",
    "jowl_left",
    "jowl_right",
)

#: Order in which overlap is resolved: a pixel claimed by an earlier zone
#: in this list cannot be claimed by a later one. This is the SPECIFIC →
#: COARSE priority. ``glabella`` beats ``forehead`` because the small box
#: between the brows is more anatomically specific than the whole-forehead
#: catch-all. ``cheek_*`` is last because it overlaps with most lower-face
#: zones and should only collect what nothing else wanted.
ZONE_PRIORITY: Tuple[str, ...] = (
    "glabella",
    "forehead",
    "crows_feet_left",
    "crows_feet_right",
    "under_eye_left",
    "under_eye_right",
    "nasolabial_left",
    "nasolabial_right",
    "perioral",
    "chin",
    "jowl_left",
    "jowl_right",
    "cheek_left",
    "cheek_right",
)

# Sanity: both tuples must contain the same set of names.
assert set(ZONE_ITERATION_ORDER) == set(ZONE_PRIORITY), (
    "ZONE_ITERATION_ORDER and ZONE_PRIORITY must contain the same zone names"
)
assert len(ZONE_PRIORITY) == 14, "Expected exactly 14 zones"
