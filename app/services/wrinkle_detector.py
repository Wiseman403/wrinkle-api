"""Pipeline orchestrator and FaceLandmarker singleton.

This module owns the long-lived MediaPipe ``FaceLandmarker`` instance.
It is created once in the FastAPI ``lifespan`` and reused for every
request — instantiation is the single most expensive thing in the
pipeline (~300 ms cold), so doing it per-request would be a non-starter.

Concurrency model
-----------------

The MVP serializes ``detector.detect(...)`` calls behind a
``threading.Lock``. MediaPipe ``FaceLandmarker`` is not documented as
safe for concurrent calls, and the cost of contention here (~tens of
ms per call) is acceptable for the expected request rate. Everything
downstream (Frangi, scikit-image) is pure numpy/cython — those release
the GIL during compute and don't need locking. If throughput becomes a
problem, swap the lock for a small pool of detectors behind an
``asyncio.Queue``.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from numpy.typing import NDArray

from ..core import landmarks as lm
from ..core import masks as masks_mod
from ..core import measure as measure_mod
from ..core import preprocess as preprocess_mod
from ..core import ridges as ridges_mod
from ..core import severity as severity_mod

logger = logging.getLogger(__name__)


# --- Exception hierarchy -------------------------------------------------


class WrinkleDetectorError(Exception):
    """Base for pipeline errors that map to client-facing 400 responses."""


class NoFaceDetectedError(WrinkleDetectorError):
    """The image contained no detectable face."""


class MultipleFacesDetectedError(WrinkleDetectorError):
    """The image contained more than one face. Pipeline expects exactly one.

    With ``num_faces=1`` the MediaPipe FaceLandmarker shouldn't return
    >1 face in practice, but we defensively check anyway so a future
    config change can't silently produce nonsense readings (mixed
    landmarks from two faces would calibrate to garbage IPD).
    """


# --- The orchestrator ----------------------------------------------------


class WrinkleDetector:
    """Singleton holding the FaceLandmarker and running the full pipeline.

    Construct once at startup; call :meth:`analyze` per request; call
    :meth:`close` at shutdown to release native resources. Reusing
    leaked instances across uvicorn reloads will eventually exhaust
    GPU/native handles.
    """

    def __init__(self, model_path: Path):
        """Create the FaceLandmarker.

        Args:
            model_path: Path to the ``face_landmarker.task`` bundle.
                Must exist on disk; auto-download is the caller's
                responsibility (see ``app.main.lifespan``).
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"FaceLandmarker model not found at {model_path}. "
                "Auto-download must run before constructing WrinkleDetector."
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            # Tracking confidence is ignored in RunningMode.IMAGE (no
            # frame-to-frame state). Kept for parity with the notebook.
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)
        # Lock around detector.detect() only — the rest of the pipeline
        # (Frangi, skeletonize, regionprops) is GIL-friendly and per-call
        # thread-safe on independent inputs.
        self._detect_lock = threading.Lock()
        logger.info("FaceLandmarker initialized from %s", model_path)

    def close(self) -> None:
        """Release native FaceLandmarker resources."""
        try:
            self._detector.close()
            logger.info("FaceLandmarker closed")
        except Exception:
            # Don't propagate cleanup errors — they would mask the
            # original shutdown reason if any.
            logger.exception("error while closing FaceLandmarker")

    # --- Pipeline ------------------------------------------------------

    def _detect_landmarks(self, img_rgb: NDArray[np.uint8]) -> NDArray[np.floating]:
        """Run FaceLandmarker.detect under the lock and return pixel-space pts.

        Raises:
            NoFaceDetectedError, MultipleFacesDetectedError.
        """
        h, w = img_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        with self._detect_lock:
            result = self._detector.detect(mp_image)

        face_landmarks_list = result.face_landmarks or []
        if not face_landmarks_list:
            raise NoFaceDetectedError(
                "no face detected. Try a clearer/closer selfie with the face roughly upright."
            )
        if len(face_landmarks_list) > 1:
            raise MultipleFacesDetectedError(
                f"expected exactly one face, got {len(face_landmarks_list)}. "
                "Pipeline does not support multi-face images."
            )

        face_landmarks = face_landmarks_list[0]
        pts = np.array([(lm_pt.x * w, lm_pt.y * h) for lm_pt in face_landmarks])
        return pts

    def analyze(self, img_rgb: NDArray[np.uint8]) -> Dict[str, object]:
        """Run the full pipeline on an EXIF-rotated RGB image.

        This is the synchronous, CPU-bound entrypoint. Callers from an
        async route handler should dispatch via ``asyncio.to_thread``.

        Args:
            img_rgb: ``(h, w, 3)`` uint8 RGB image.

        Returns:
            A JSON-shaped dict matching :class:`app.schemas.AnalyzeResponse`,
            minus ``processing_time_ms`` (the route handler stamps that).

        Raises:
            NoFaceDetectedError, MultipleFacesDetectedError.
        """
        h, w = img_rgb.shape[:2]
        shape = (h, w)

        # 1. Landmarks (locked)
        pts = self._detect_landmarks(img_rgb)

        # 2. IPD calibration
        ipd_px = float(np.linalg.norm(pts[lm.RIGHT_IRIS_CENTER] - pts[lm.LEFT_IRIS_CENTER]))
        if ipd_px <= 1.0:
            # Degenerate landmarks (face too small / collapsed iris). Treat
            # as a detection failure rather than dividing by zero downstream.
            raise NoFaceDetectedError(
                f"degenerate IPD ({ipd_px:.2f} px) — face is too small or landmarks failed."
            )
        mm_per_px = lm.IPD_MM_ASSUMED / ipd_px

        # 3. Skin mask + zone masks
        skin_mask = masks_mod.build_skin_mask(pts, shape, ipd_px)
        zone_masks = masks_mod.build_zone_masks(pts, shape, ipd_px, skin_mask)

        # 4. Preprocess
        L_sharp = preprocess_mod.preprocess_l_channel(img_rgb)

        # 5. Frangi + threshold + skeleton
        ridge_response, sigmas = ridges_mod.compute_frangi_response(L_sharp, skin_mask, ipd_px)
        skeleton, low_t, high_t = ridges_mod.threshold_and_skeletonize(
            ridge_response, skin_mask, mm_per_px
        )

        # 6. Per-zone metrics
        zones = {
            zname: measure_mod.measure_zone(skeleton, zmask, mm_per_px)
            for zname, zmask in zone_masks.items()
        }
        whole_face = measure_mod.measure_zone(skeleton, skin_mask, mm_per_px)

        # 7. Leftover skin (inside skin mask, outside every zone)
        leftover_mask = measure_mod.compute_leftover_skin_mask(skin_mask, zone_masks)
        leftover_skin = measure_mod.measure_zone(skeleton, leftover_mask, mm_per_px)

        # 8. Per-wrinkle entries (build zone_id_map ONCE, reuse here only)
        zone_id_map, zone_id_to_name = measure_mod.build_zone_id_map(zone_masks, shape)
        per_wrinkle = measure_mod.per_wrinkle_entries(
            skeleton, zone_id_map, zone_id_to_name, mm_per_px
        )

        # 9. Severity (informational, not clinical)
        severity = severity_mod.build_severity_payload(zones, whole_face)

        return {
            "image_size": {"width": int(w), "height": int(h)},
            "calibration": {
                "ipd_px": round(ipd_px, 2),
                "ipd_mm_assumed": lm.IPD_MM_ASSUMED,
                "mm_per_px": round(mm_per_px, 5),
            },
            "frangi_sigmas_px": [round(float(s), 3) for s in sigmas],
            "hysteresis_threshold": {
                "low": round(low_t, 5),
                "high": round(high_t, 5),
            },
            "min_component_mm": measure_mod.MIN_COMPONENT_MM,
            "zones": zones,
            "whole_face": whole_face,
            "leftover_skin": leftover_skin,
            "per_wrinkle": per_wrinkle,
            "severity": severity,
        }
