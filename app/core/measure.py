"""Per-zone metrics, per-wrinkle entries, and zone-id mapping.

After the skeleton has been built and branch-split, this module slices
it by zone, applies the minimum component length filter, and computes
the JSON-shaped output structures.

A real static wrinkle should be at least 4 mm long. Shorter components
are texture, JPEG ringing, pore-chain fragments, or leftover noise from
the threshold pass. ``min_component_mm`` is therefore a property of the
algorithm (echoed in the response, not a per-deployment knob).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.measure import label, regionprops


#: Minimum accepted component length, in mm. Components shorter than this
#: are dropped from per-zone counts and from the per-wrinkle list.
MIN_COMPONENT_MM: float = 4.0


def measure_zone(
    skel: NDArray[np.bool_],
    zone_mask: NDArray[np.uint8],
    mm_per_px: float,
    min_component_mm: float = MIN_COMPONENT_MM,
) -> Dict[str, object]:
    """Compute aggregate metrics for a single zone.

    Length is approximated by counting skeleton pixels × ``mm_per_px``.
    Diagonal pixels under-count by up to √2; this is fine for a relative
    MVP score.

    Args:
        skel: Branch-split skeleton (bool array).
        zone_mask: uint8 mask (0/255) for the zone.
        mm_per_px: Pixel-to-mm calibration.
        min_component_mm: Minimum accepted component length (default 4 mm).

    Returns:
        Dict with keys:
        ``wrinkle_count``, ``total_length_mm``, ``zone_area_cm2``,
        ``density_mm_per_cm2``, ``longest_wrinkle_mm``,
        ``component_lengths_mm``.
    """
    zone_skel = skel & (zone_mask > 0)
    area_px = int((zone_mask > 0).sum())
    area_cm2 = area_px * (mm_per_px ** 2) / 100.0

    if not zone_skel.any():
        return {
            "wrinkle_count": 0,
            "total_length_mm": 0.0,
            "zone_area_cm2": round(area_cm2, 2),
            "density_mm_per_cm2": 0.0,
            "longest_wrinkle_mm": 0.0,
            "component_lengths_mm": [],
        }

    labeled = label(zone_skel, connectivity=2)
    component_lengths: List[float] = []
    for region in regionprops(labeled):
        length_mm = region.area * mm_per_px
        if length_mm >= min_component_mm:
            component_lengths.append(length_mm)
    total = float(sum(component_lengths))
    return {
        "wrinkle_count": len(component_lengths),
        "total_length_mm": round(total, 2),
        "zone_area_cm2": round(area_cm2, 2),
        "density_mm_per_cm2": round(total / area_cm2, 2) if area_cm2 > 0 else 0.0,
        "longest_wrinkle_mm": round(max(component_lengths), 2) if component_lengths else 0.0,
        "component_lengths_mm": [round(l, 2) for l in sorted(component_lengths, reverse=True)],
    }


def build_zone_id_map(
    zone_masks: Dict[str, NDArray[np.uint8]],
    shape: Tuple[int, int],
) -> Tuple[NDArray[np.int32], Dict[int, str]]:
    """Build a per-pixel zone-id image and the id→name reverse map.

    Pixels not in any zone get id 0. Iteration follows the dict's
    insertion order, which (because zone_masks is built via
    :func:`app.core.masks.build_zone_masks`) follows
    ``ZONE_ITERATION_ORDER``. Overlap has already been resolved at mask
    construction time, so the order doesn't change pixel assignments —
    it only stabilizes the integer ids across runs.

    Args:
        zone_masks: Dict of zone-name → uint8 mask.
        shape: ``(h, w)`` of the image.

    Returns:
        ``(zone_id_map, zone_id_to_name)``:

        - ``zone_id_map``: int32 image, 0 = no zone, 1..N for each zone.
        - ``zone_id_to_name``: ``{i: zone_name}`` reverse lookup.
    """
    h, w = shape
    zone_id_map: NDArray[np.int32] = np.zeros((h, w), dtype=np.int32)
    zone_id_to_name: Dict[int, str] = {}
    for i, (zname, zmask) in enumerate(zone_masks.items(), start=1):
        zone_id_map[zmask > 0] = i
        zone_id_to_name[i] = zname
    return zone_id_map, zone_id_to_name


def per_wrinkle_entries(
    skeleton: NDArray[np.bool_],
    zone_id_map: NDArray[np.int32],
    zone_id_to_name: Dict[int, str],
    mm_per_px: float,
    min_component_mm: float = MIN_COMPONENT_MM,
) -> List[Dict[str, object]]:
    """Build the per-wrinkle entries list, sorted longest first.

    Each entry's ``zone`` is the majority-vote of its pixels' zone ids
    (handles the rare case where a component straddles a zone boundary
    — the priority pass during mask construction makes this uncommon).
    Components that fall entirely outside every zone are tagged
    ``"unzoned"``.

    Ids are assigned post-sort, so id 0 is always the longest wrinkle
    in the image.

    Args:
        skeleton: Branch-split skeleton (bool).
        zone_id_map: From :func:`build_zone_id_map`.
        zone_id_to_name: From :func:`build_zone_id_map`.
        mm_per_px: Pixel-to-mm calibration.
        min_component_mm: Minimum accepted component length (default 4 mm).

    Returns:
        List of dicts: ``id``, ``zone``, ``length_mm``,
        ``centroid_xy`` (``[x, y]``), ``bbox_xywh`` (``[x, y, w, h]``).
    """
    entries: List[Dict[str, object]] = []
    labeled_skel = label(skeleton, connectivity=2)
    for region in regionprops(labeled_skel):
        length_mm = region.area * mm_per_px
        if length_mm < min_component_mm:
            continue
        coords = region.coords
        pixel_zones = zone_id_map[coords[:, 0], coords[:, 1]]
        nonzero_zones = pixel_zones[pixel_zones > 0]
        if len(nonzero_zones) > 0:
            zid = int(np.bincount(nonzero_zones).argmax())
            zone_assignment = zone_id_to_name[zid]
        else:
            zone_assignment = "unzoned"
        cy, cx = region.centroid
        miny, minx, maxy, maxx = region.bbox
        entries.append(
            {
                "id": len(entries),  # placeholder; rewritten below post-sort
                "zone": zone_assignment,
                "length_mm": round(length_mm, 2),
                "centroid_xy": [int(round(cx)), int(round(cy))],
                "bbox_xywh": [int(minx), int(miny), int(maxx - minx), int(maxy - miny)],
            }
        )
    # Sort by length, longest first, and re-id so id 0 = longest.
    entries.sort(key=lambda d: -float(d["length_mm"]))  # type: ignore[arg-type]
    for new_id, entry in enumerate(entries):
        entry["id"] = new_id
    return entries


def compute_leftover_skin_mask(
    skin_mask: NDArray[np.uint8],
    zone_masks: Dict[str, NDArray[np.uint8]],
) -> NDArray[np.uint8]:
    """Skin pixels that fall outside every zone.

    A large leftover-skin reading (e.g. dozens of mm of wrinkles) means
    wrinkles are landing in regions the zone map doesn't cover yet
    (e.g. neck, jawline, ear area). Surfacing this in the response
    helps callers know when to enlarge a zone.

    Args:
        skin_mask: uint8 skin mask.
        zone_masks: Dict of zone-name → uint8 mask.

    Returns:
        uint8 mask: skin AND (NOT union-of-zones).
    """
    h, w = skin_mask.shape
    union_zone_mask: NDArray[np.uint8] = np.zeros((h, w), dtype=np.uint8)
    for zmask in zone_masks.values():
        union_zone_mask = cv2.bitwise_or(union_zone_mask, zmask)
    return cv2.bitwise_and(skin_mask, cv2.bitwise_not(union_zone_mask))
