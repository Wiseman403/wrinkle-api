"""Multi-scale Frangi ridges → hysteresis → skeleton → branch-split.

This module owns the "what is a wrinkle?" half of the pipeline. Inputs
are the sharpened L* image and the skin mask; outputs are a skeleton
(1-px curves) and the threshold values actually used (echoed in the
response so callers can see what happened).

All numeric constants are the values that worked in the notebook on
real selfies. Do not silently "modernize" them.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.filters import apply_hysteresis_threshold, frangi
from skimage.morphology import remove_small_objects, skeletonize


def compute_frangi_response(
    L_sharp: NDArray[np.uint8],
    skin_mask: NDArray[np.uint8],
    ipd_px: float,
) -> Tuple[NDArray[np.float32], List[float]]:
    """Run multi-scale Frangi on the sharpened L*, mask to skin.

    Wrinkle widths span roughly 0.3-3 mm. Sigmas are tuned to that
    range, scaled to image px via IPD. Six scales (vs. five in older
    versions) give finer multi-scale fusion at the fine-line end.

    ``gamma=10`` (vs. 15 in older versions) is more sensitive to
    low-contrast structures — this is the right call for fine lines on
    smooth/well-lit skin where Frangi otherwise under-fires.

    ``black_ridges=True`` because wrinkles read as dark valleys under
    most lighting.

    Args:
        L_sharp: Output of :func:`app.core.preprocess.preprocess_l_channel`.
        skin_mask: Skin mask from :func:`app.core.masks.build_skin_mask`.
            Pixels outside the mask are zeroed in the response.
        ipd_px: Inter-pupillary distance in pixels.

    Returns:
        A tuple ``(ridge_in_skin, sigmas)``:

        - ``ridge_in_skin``: float32 ridge-strength image, masked to skin.
        - ``sigmas``: the 6 sigma values used (in pixels), for telemetry.
    """
    sigma_min = max(0.8, 0.4 * (0.3 * ipd_px / 63))
    sigma_max = max(sigma_min + 1.0, 0.5 * (3.0 * ipd_px / 63))
    sigmas = np.geomspace(sigma_min, sigma_max, 6)

    L_float = L_sharp.astype(np.float32)
    ridge_response = frangi(
        L_float,
        sigmas=sigmas,
        alpha=0.5,
        beta=0.5,
        gamma=10,
        black_ridges=True,  # wrinkles read as dark valleys under most lighting
    ).astype(np.float32)

    # Apply skin mask (zero everything outside skin so thresholds aren't
    # contaminated by hair, lips, etc.).
    ridge_in_skin = ridge_response.copy()
    ridge_in_skin[skin_mask == 0] = 0
    return ridge_in_skin, [float(s) for s in sigmas]


def threshold_and_skeletonize(
    ridge_response: NDArray[np.float32],
    skin_mask: NDArray[np.uint8],
    mm_per_px: float,
) -> Tuple[NDArray[np.bool_], float, float]:
    """Threshold the Frangi response and produce a 1-px skeleton.

    Three things matter here:

    1. **Hysteresis threshold** instead of a single threshold. A pixel
       is kept if it (a) exceeds ``high_t``, or (b) exceeds ``low_t``
       AND is connected to a ``high_t`` pixel. This is the same trick
       as Canny edge detection — it follows weak parts of a wrinkle as
       long as they connect to a confidently-strong part, while
       rejecting isolated weak pixels (texture/noise).
    2. **Percentile-based bounds with floors.** Otsu over-thresholds
       when most skin is smooth (it splits between near-zero and a few
       hot iris/specular pixels). Percentiles adapt to the actual
       response distribution per image.
    3. **Min component length is 4 mm** (handled in measure.py).
       Anything shorter is texture or a fragment, not a clinically-
       meaningful wrinkle.

    Branch-point splitting on the skeleton: a real wrinkle whose edge
    happens to touch a noise blob otherwise gets measured as a single
    100+ mm component. A branch point is a skeleton pixel with 3+
    skeleton neighbors. Removing those pixels disconnects the skeleton
    into individual paths, each measured separately.

    Args:
        ridge_response: Float32 Frangi response, already masked to skin.
        skin_mask: The same skin mask (uint8, 0/255).
        mm_per_px: Calibration scalar; controls the ``min_blob_px`` floor.

    Returns:
        A tuple ``(skeleton, low_t, high_t)``:

        - ``skeleton``: bool array, True at skeleton pixels.
        - ``low_t``, ``high_t``: hysteresis thresholds actually used.
    """
    skin_pix = ridge_response[skin_mask > 0]
    nonzero = skin_pix[skin_pix > 1e-6]

    if len(nonzero) > 1000:
        low_t = max(0.020, float(np.percentile(nonzero, 88)))
        high_t = max(0.050, float(np.percentile(nonzero, 96)))
        binary_full = apply_hysteresis_threshold(ridge_response, low_t, high_t)
    else:
        # Sparse response (very smooth face / under-detection): single
        # threshold fallback. Both bounds collapse to p95 so the
        # response payload still has well-defined low/high values.
        high_t = float(np.percentile(skin_pix, 95)) if len(skin_pix) else 0.0
        low_t = high_t
        binary_full = ridge_response > high_t

    binary = binary_full & (skin_mask > 0)

    # Drop tiny noise blobs (≈ < 0.5 mm² area). Floor of 10 px keeps the
    # filter sane on extremely small selfies.
    min_blob_px = max(10, int((0.7 / mm_per_px) ** 2))
    binary_clean = remove_small_objects(binary, min_size=min_blob_px)

    # Skeletonize to 1-px curves
    skeleton_raw = skeletonize(binary_clean)

    # CRITICAL: split skeleton at branch points. Without this, a real
    # wrinkle whose edge happens to touch a noise blob gets measured as
    # a single 100+ mm component. The 8-neighbor count is computed by
    # convolving with a 3×3 ones-kernel-with-zero-center and keeping
    # pixels with 3+ neighbors.
    neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton_raw.astype(np.uint8), -1, neighbor_kernel)
    branch_points = (neighbor_count >= 3) & skeleton_raw
    skeleton = skeleton_raw & ~branch_points

    return skeleton.astype(bool), low_t, high_t
