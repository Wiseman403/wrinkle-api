"""L* preprocessing: CLAHE → bilateral → unsharp.

Frangi expects a single-channel image where ridge structures stand out
relative to their neighborhood. The transformations below were tuned
on real selfies; in particular:

- The unsharp step is what most reliably moves subtle real wrinkles
  above the detection threshold without amplifying noise much
  (bilateral already de-noised).
- Mild unsharp settings (1.3 / -0.3 around a σ=1.5 Gaussian) — strong
  unsharp amplifies pore-scale texture and creates hundreds of false
  positives on otherwise-smooth skin.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def preprocess_l_channel(img_rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Build the sharpened L* image fed to the Frangi filter.

    Pipeline:

    1. ``RGB → CIE LAB``, take L\\* (perceptual lightness — wrinkles read
       cleanly here regardless of skin tone).
    2. CLAHE (clip 2.0, tile 8×8) for local contrast normalization.
    3. Bilateral filter (d=5, σ_color=20, σ_space=5) — edge-preserving
       denoise.
    4. Mild unsharp mask via ``addWeighted`` against a σ=1.5 Gaussian.

    Args:
        img_rgb: ``(h, w, 3)`` uint8 RGB image.

    Returns:
        A ``(h, w)`` uint8 image suitable for ``skimage.filters.frangi``
        with ``black_ridges=True``.
    """
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L = img_lab[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)

    L_smooth = cv2.bilateralFilter(L_clahe, d=5, sigmaColor=20, sigmaSpace=5)

    # Unsharp mask: enhance ridges of typical wrinkle width (~3 px at this
    # scale). Mild settings (1.3 / -0.3) — strong unsharp amplifies
    # pore-scale texture and creates hundreds of false positives on
    # otherwise-smooth skin.
    blur = cv2.GaussianBlur(L_smooth, (0, 0), sigmaX=1.5)
    L_sharp = cv2.addWeighted(L_smooth, 1.3, blur, -0.3, 0)
    return L_sharp
