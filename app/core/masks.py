"""Skin and per-zone mask construction.

The pipeline builds three layers of masks, in this order:

1. **Face mask** — fills the ``FACE_OVAL`` polygon. Defines the outer edge
   of "the face" in the image.
2. **Skin mask** — face mask minus dilated exclusions for eyes, brows,
   lips and nostrils, then eroded slightly to pull off the hair boundary.
   This is the canvas for ridge detection.
3. **Zone masks** — 14 anatomical zones carved out of the skin mask,
   then resolved against each other so each pixel belongs to at most
   one zone.

All exclusion margins, erosion radii, and zone box dimensions are
expressed as fractions of inter-pupillary distance (in pixels), so they
scale automatically with selfie size.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from . import landmarks as lm


# --- Type aliases --------------------------------------------------------

#: A 2-D mask with values in {0, 255} (uint8). All masks in this module use
#: this representation so OpenCV bitwise ops apply directly.
Mask = NDArray[np.uint8]

#: An (N, 2) float array of landmark XY pixel coordinates.
Points = NDArray[np.floating]


# --- Polygon helpers -----------------------------------------------------


def fill_hull(
    indices: List[int],
    pts: Points,
    shape: Tuple[int, int],
    dilate: int = 0,
) -> Mask:
    """Fill the convex hull of selected landmarks.

    Convex-hulling is more forgiving than ``fill_poly`` when the index
    list is unordered or when MediaPipe noise pushes a landmark off the
    contour — the hull always closes cleanly.

    Args:
        indices: Landmark indices to include.
        pts: ``(N, 2)`` array of all landmark coordinates.
        shape: ``(h, w)`` of the image.
        dilate: Radius in pixels of an elliptical dilation applied after
            filling. Use this to add a safety margin around the hull
            (e.g. for eyelashes around the eye contour).

    Returns:
        A ``(h, w)`` uint8 mask with the hull filled at 255.
    """
    h, w = shape
    mask: Mask = np.zeros((h, w), dtype=np.uint8)
    poly = pts[indices].astype(np.int32)
    hull = cv2.convexHull(poly)
    cv2.fillConvexPoly(mask, hull, 255)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        mask = cv2.dilate(mask, k)
    return mask


def fill_poly(
    indices: List[int],
    pts: Points,
    shape: Tuple[int, int],
    dilate: int = 0,
) -> Mask:
    """Fill the (ordered) polygon defined by selected landmarks.

    Use this for boundaries where the index ordering matters — e.g.
    ``FACE_OVAL`` is a continuous trace and convex-hulling it would
    round off the chin.

    Args:
        indices: Ordered landmark indices forming a closed polygon.
        pts: ``(N, 2)`` array of all landmark coordinates.
        shape: ``(h, w)`` of the image.
        dilate: Optional elliptical dilation radius applied after filling.

    Returns:
        A ``(h, w)`` uint8 mask with the polygon filled at 255.
    """
    h, w = shape
    mask: Mask = np.zeros((h, w), dtype=np.uint8)
    poly = pts[indices].astype(np.int32)
    cv2.fillPoly(mask, [poly], 255)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        mask = cv2.dilate(mask, k)
    return mask


# --- Skin mask -----------------------------------------------------------


def build_skin_mask(
    pts: Points,
    shape: Tuple[int, int],
    ipd_px: float,
) -> Mask:
    """Build the skin mask: face oval, minus exclusions, eroded inward.

    Why exclusion is mandatory: Frangi fires on every eyebrow hair and
    eyelash. Without aggressive exclusion the forehead "wrinkle count"
    balloons into the hundreds of false positives. This is what
    separates a working detector from a useless one.

    Why erosion: the ``FACE_OVAL`` polygon sits right at the skin/hair
    boundary, where landmark noise lets hair shadows leak in and
    produce false ridges. A ~1 mm erosion (``0.018 * ipd_px``, with a
    floor of 2 px so it still does something on tiny selfies) pulls the
    boundary safely inside the skin.

    Args:
        pts: ``(N, 2)`` array of all 478 landmark coordinates.
        shape: ``(h, w)`` of the image.
        ipd_px: Inter-pupillary distance in pixels. All margins scale
            with this so they're consistent across selfie resolutions.

    Returns:
        A ``(h, w)`` uint8 mask of skin pixels (255 = skin, 0 = elsewhere).
    """
    face_mask = fill_poly(lm.FACE_OVAL, pts, shape, dilate=0)

    # Exclusion margins scaled to IPD so they're consistent across selfie
    # sizes. Tuned conservatively so we DON'T eat the actual wrinkles
    # (crow's-feet roots are right at the canthus; forehead lines are
    # right above the brow).
    d_eye = int(0.035 * ipd_px)   # was 0.06 — kill eyelashes only, keep crow's-feet roots
    d_brow = int(0.045 * ipd_px)  # was 0.07 — kill brow hairs only, keep above-brow forehead lines
    d_lip = int(0.025 * ipd_px)
    d_nose = int(0.025 * ipd_px)

    h, w = shape
    exclusion: Mask = np.zeros((h, w), dtype=np.uint8)
    exclusion |= fill_hull(lm.RIGHT_EYE,     pts, shape, dilate=d_eye)
    exclusion |= fill_hull(lm.LEFT_EYE,      pts, shape, dilate=d_eye)
    exclusion |= fill_hull(lm.RIGHT_EYEBROW, pts, shape, dilate=d_brow)
    exclusion |= fill_hull(lm.LEFT_EYEBROW,  pts, shape, dilate=d_brow)
    exclusion |= fill_hull(lm.LIPS_OUTER,    pts, shape, dilate=d_lip)
    exclusion |= fill_hull(lm.NOSTRIL_AREA,  pts, shape, dilate=d_nose)

    skin_mask = cv2.bitwise_and(face_mask, cv2.bitwise_not(exclusion))

    # Erode skin mask slightly — see the docstring above for rationale.
    erode_px = max(2, int(0.018 * ipd_px))
    skin_mask = cv2.erode(
        skin_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)),
    )
    return skin_mask


# --- Zone construction ---------------------------------------------------


def _order_by_x(idx_a: int, idx_b: int, pts: Points) -> Tuple[int, int]:
    """Return ``(left, right)`` sorted by x-coordinate.

    Landmark index does not necessarily correspond to image-left vs
    image-right — depends on the subject's pose. Always sort at runtime.
    """
    return (idx_a, idx_b) if pts[idx_a][0] < pts[idx_b][0] else (idx_b, idx_a)


def _crows_feet_box(canthus_pt: NDArray, side: str, ipd_px: float, shape: Tuple[int, int]) -> Mask:
    """Lateral box anchored at the outer canthus.

    Wider lateral reach (``cf_w = 0.38 * ipd``, was 0.30) and asymmetric
    vertical extent — more reach below the canthus than above — to catch
    the radial fan that fans down toward the cheek.
    """
    h, w = shape
    cx, cy = int(canthus_pt[0]), int(canthus_pt[1])
    cf_w = int(0.38 * ipd_px)        # was 0.30 — wider lateral reach
    cf_up = int(0.18 * ipd_px)
    cf_dn = int(0.22 * ipd_px)       # asymmetric: more reach below canthus
    box: Mask = np.zeros((h, w), dtype=np.uint8)
    if side == "left":
        cv2.rectangle(box, (max(0, cx - cf_w), cy - cf_up), (cx, cy + cf_dn), 255, -1)
    else:
        cv2.rectangle(box, (cx, cy - cf_up), (min(w, cx + cf_w), cy + cf_dn), 255, -1)
    return box


def _under_eye_box(lid_pt: NDArray, ipd_px: float, shape: Tuple[int, int]) -> Mask:
    """Box directly below the lower lid.

    Starts ``0.03*ipd`` below the lid (pad over lashes), reaches
    ``0.20*ipd`` further down. Captures lower-lid lines.
    """
    h, w = shape
    cx, cy = int(lid_pt[0]), int(lid_pt[1])
    ue_w = int(0.32 * ipd_px)
    ue_top = int(0.03 * ipd_px)      # start a hair below the lid (pad over lashes)
    ue_h = int(0.20 * ipd_px)
    box: Mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(
        box,
        (cx - ue_w // 2, cy + ue_top),
        (cx + ue_w // 2, cy + ue_top + ue_h),
        255,
        -1,
    )
    return box


def _nasolabial_strip(
    nose_pt: NDArray,
    mouth_pt: NDArray,
    half_width: int,
    shape: Tuple[int, int],
) -> Mask:
    """Oblique strip from alar base to mouth corner.

    This is the actual fold path. Earlier indices (129/358) sat on the
    lip border and produced strips that missed the fold.

    Constructs a parallelogram by walking ``half_width`` along the
    perpendicular to the alar→mouth direction at each end.
    """
    h, w = shape
    direction = mouth_pt - nose_pt
    length = np.linalg.norm(direction)
    if length < 1:
        return np.zeros((h, w), dtype=np.uint8)
    perp = np.array([-direction[1], direction[0]]) / length * half_width
    poly = np.array(
        [nose_pt + perp, nose_pt - perp, mouth_pt - perp, mouth_pt + perp],
        dtype=np.int32,
    )
    box: Mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(box, [poly], 255)
    return box


def _cheek_box(
    cheek_pt: NDArray,
    eye_pt: NDArray,
    ala_pt: NDArray,
    side: str,
    ipd_px: float,
    shape: Tuple[int, int],
) -> Mask:
    """Lateral cheek box.

    Catch-all for lateral-cheek wrinkles that fall outside crow's feet,
    under-eye, and nasolabial. Spans from ``0.08*ipd`` below the eye to
    ``0.10*ipd`` below the alar base.
    """
    h, w = shape
    cx_lat = int(cheek_pt[0])
    cy_top = int(eye_pt[1] + 0.08 * ipd_px)
    cy_bot = int(ala_pt[1] + 0.10 * ipd_px)
    cx_med = int(ala_pt[0])
    box: Mask = np.zeros((h, w), dtype=np.uint8)
    if side == "left":
        cv2.rectangle(box, (max(0, cx_lat), cy_top), (cx_med, cy_bot), 255, -1)
    else:
        cv2.rectangle(box, (cx_med, cy_top), (min(w, cx_lat), cy_bot), 255, -1)
    return box


def _jowl_box(
    ala_pt: NDArray,
    chin_tip_pt: NDArray,
    chin_w: int,
    side: str,
    ipd_px: float,
    shape: Tuple[int, int],
) -> Mask:
    """The LARGE strip lateral to the chin, below the cheek zone.

    Catches:
      - marionette lines (vertical, from mouth corner downward)
      - lateral mandibular wrinkles
      - lower-cheek wrinkles below the cheek-zone bottom edge

    Top edge meets exactly where cheek ends (``ala.y + 0.10*ipd``) so
    there's no coverage gap. Width is ``0.85*ipd`` (~2x the v1 jowl
    width).
    """
    h, w = shape
    top_y = int(ala_pt[1] + 0.10 * ipd_px)        # meets cheek bottom edge
    bot_y = int(chin_tip_pt[1] + 0.06 * ipd_px)   # slightly past chin tip
    chin_half_local = chin_w // 2
    chin_cx_local = int(chin_tip_pt[0])
    width = int(0.85 * ipd_px)                    # was 0.42 — doubled
    if side == "left":
        med_x = chin_cx_local - chin_half_local
        lat_x = max(0, med_x - width)
        box_l, box_r = lat_x, med_x
    else:
        med_x = chin_cx_local + chin_half_local
        lat_x = min(w, med_x + width)
        box_l, box_r = med_x, lat_x
    box: Mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(box, (box_l, top_y), (box_r, bot_y), 255, -1)
    return box


def build_zone_masks(
    pts: Points,
    shape: Tuple[int, int],
    ipd_px: float,
    skin_mask: Mask,
) -> Dict[str, Mask]:
    """Build all 14 zone masks and resolve overlap by priority.

    Coverage goal: zones should cover the regions where wrinkles
    ACTUALLY appear, not leave 70% of the skin in an "unzoned" gap.

    After construction, overlap is resolved by walking ``ZONE_PRIORITY``
    in order: a pixel claimed by an earlier zone cannot be claimed by a
    later one. The priority order is hand-tuned (specific → coarse) so
    a wrinkle near the inner brow is counted as "glabella" rather than
    "forehead", and lateral cheek wrinkles fall into "cheek_*" only if
    no nearer zone wanted them.

    Args:
        pts: ``(N, 2)`` landmark coordinates.
        shape: ``(h, w)`` of the image.
        ipd_px: Inter-pupillary distance in pixels.
        skin_mask: Pre-built skin mask. Each zone is intersected with
            this so zones never extend past the skin region.

    Returns:
        A dict mapping zone name → uint8 mask. Keys follow
        ``ZONE_ITERATION_ORDER``.
    """
    h, w = shape
    zone_masks: Dict[str, Mask] = {}

    # Resolve image-left vs image-right per anchor pair
    EYE_LEFT, EYE_RIGHT = _order_by_x(lm.EYE_OUTER_A, lm.EYE_OUTER_B, pts)
    ALA_LEFT, ALA_RIGHT = _order_by_x(lm.ALA_A, lm.ALA_B, pts)
    MOUTH_LEFT, MOUTH_RIGHT = _order_by_x(lm.MOUTH_A, lm.MOUTH_B, pts)
    LL_LEFT, LL_RIGHT = _order_by_x(lm.LOWER_LID_A, lm.LOWER_LID_B, pts)
    CK_LEFT, CK_RIGHT = _order_by_x(lm.CHEEK_LATERAL_A, lm.CHEEK_LATERAL_B, pts)

    # --- Forehead: skin above eyebrow line, inside face oval ---
    brow_top_y = int(min(pts[i][1] for i in lm.RIGHT_EYEBROW + lm.LEFT_EYEBROW)) - int(0.03 * ipd_px)
    fh_box: Mask = np.zeros((h, w), dtype=np.uint8)
    fh_box[:brow_top_y, :] = 255
    zone_masks["forehead"] = cv2.bitwise_and(skin_mask, fh_box)

    # --- Glabella: between inner brows, just above nose bridge ---
    glab_center = ((pts[lm.INNER_BROW_A] + pts[lm.INNER_BROW_B]) / 2).astype(int)
    glab_w = int(0.20 * ipd_px)
    glab_up = int(0.22 * ipd_px)
    glab_dn = int(0.04 * ipd_px)
    glab_box: Mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(
        glab_box,
        (glab_center[0] - glab_w // 2, glab_center[1] - glab_up),
        (glab_center[0] + glab_w // 2, glab_center[1] + glab_dn),
        255,
        -1,
    )
    zone_masks["glabella"] = cv2.bitwise_and(skin_mask, glab_box)

    # --- Crow's feet: lateral to outer canthi ---
    zone_masks["crows_feet_left"] = cv2.bitwise_and(
        skin_mask, _crows_feet_box(pts[EYE_LEFT], "left", ipd_px, shape)
    )
    zone_masks["crows_feet_right"] = cv2.bitwise_and(
        skin_mask, _crows_feet_box(pts[EYE_RIGHT], "right", ipd_px, shape)
    )

    # --- Under-eye: directly below lower lid ---
    zone_masks["under_eye_left"] = cv2.bitwise_and(
        skin_mask, _under_eye_box(pts[LL_LEFT], ipd_px, shape)
    )
    zone_masks["under_eye_right"] = cv2.bitwise_and(
        skin_mask, _under_eye_box(pts[LL_RIGHT], ipd_px, shape)
    )

    # --- Nasolabial: oblique strip from alar base to mouth corner ---
    nl_hw = int(0.14 * ipd_px)
    zone_masks["nasolabial_left"] = cv2.bitwise_and(
        skin_mask,
        _nasolabial_strip(pts[ALA_LEFT], pts[MOUTH_LEFT], nl_hw, shape),
    )
    zone_masks["nasolabial_right"] = cv2.bitwise_and(
        skin_mask,
        _nasolabial_strip(pts[ALA_RIGHT], pts[MOUTH_RIGHT], nl_hw, shape),
    )

    # --- Perioral: ring around the lips (radial fines / barcode lines) ---
    perioral_outer = fill_hull(lm.LIPS_OUTER, pts, shape, dilate=int(0.12 * ipd_px))
    perioral_inner = fill_hull(lm.LIPS_OUTER, pts, shape, dilate=0)
    perioral_ring = cv2.bitwise_and(perioral_outer, cv2.bitwise_not(perioral_inner))
    zone_masks["perioral"] = cv2.bitwise_and(skin_mask, perioral_ring)

    # --- Cheek (lateral) ---
    zone_masks["cheek_left"] = cv2.bitwise_and(
        skin_mask,
        _cheek_box(pts[CK_LEFT], pts[EYE_LEFT], pts[ALA_LEFT], "left", ipd_px, shape),
    )
    zone_masks["cheek_right"] = cv2.bitwise_and(
        skin_mask,
        _cheek_box(pts[CK_RIGHT], pts[EYE_RIGHT], pts[ALA_RIGHT], "right", ipd_px, shape),
    )

    # --- Chin / mental crease ---
    chin_top_y = int(pts[lm.LIP_BOTTOM][1] + 0.06 * ipd_px)
    chin_bot_y = int(pts[lm.CHIN_BOTTOM][1] - 0.02 * ipd_px)
    chin_w = int(0.42 * ipd_px)
    chin_cx = int(pts[lm.CHIN_BOTTOM][0])
    chin_box: Mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(
        chin_box,
        (chin_cx - chin_w // 2, chin_top_y),
        (chin_cx + chin_w // 2, chin_bot_y),
        255,
        -1,
    )
    zone_masks["chin"] = cv2.bitwise_and(skin_mask, chin_box)

    # --- Jowl (left + right) ---
    zone_masks["jowl_left"] = cv2.bitwise_and(
        skin_mask,
        _jowl_box(pts[ALA_LEFT], pts[lm.CHIN_BOTTOM], chin_w, "left", ipd_px, shape),
    )
    zone_masks["jowl_right"] = cv2.bitwise_and(
        skin_mask,
        _jowl_box(pts[ALA_RIGHT], pts[lm.CHIN_BOTTOM], chin_w, "right", ipd_px, shape),
    )

    # --- Resolve overlap by priority (specific → coarse). A pixel
    # claimed by an earlier zone cannot be claimed by a later one. This
    # prevents a wrinkle from being double-counted across zones. ---
    already_claimed: Mask = np.zeros((h, w), dtype=np.uint8)
    for zname in lm.ZONE_PRIORITY:
        zone_masks[zname] = cv2.bitwise_and(zone_masks[zname], cv2.bitwise_not(already_claimed))
        already_claimed = cv2.bitwise_or(already_claimed, zone_masks[zname])

    # Re-emit dict in iteration order so the response JSON is stable.
    return {name: zone_masks[name] for name in lm.ZONE_ITERATION_ORDER}
