"""Face quality checks: set aside only faces that cannot be recognised reliably.

The first real run (1,081 photos) set aside 41% of faces, which was far too strict. Thresholds
are now lenient, sharpness is measured at a fixed 112px scale so it does not depend on how big
the face is in the photo, and every rejection carries its reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityThresholds:
    min_face_px: int = 40  # shorter side of the face box, in the (max 2048px) decoded photo
    min_det_score: float = 0.5
    min_sharpness: float = 8.0  # Laplacian variance on the 112px grey crop
    max_abs_yaw_deg: float = 75.0


def laplacian_variance(grey: np.ndarray) -> float:
    """Variance of a 4-neighbour Laplacian; low values mean blur."""
    g = grey.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def sharpness_112(face_rgb: np.ndarray) -> float:
    if face_rgb.size == 0 or min(face_rgb.shape[:2]) < 3:
        return 0.0
    grey = Image.fromarray(face_rgb).convert("L").resize((112, 112), Image.Resampling.BILINEAR)
    return laplacian_variance(np.asarray(grey))


def estimate_yaw_from_landmarks(kps: np.ndarray) -> float:
    """Rough yaw (degrees) from 5-point landmarks: eyes, nose, mouth corners.

    0 when the nose sits midway between the eyes, approaching +/-90 as it reaches one eye.
    Good enough to reject near-profile faces, not a real head-pose estimate.
    """
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_span = right_eye[0] - left_eye[0]
    if abs(eye_span) < 1e-6:
        return 90.0
    rel = (nose[0] - left_eye[0]) / eye_span
    return float(np.clip((rel - 0.5) * 2, -1, 1) * 90)


def assess(
    face_rgb: np.ndarray,
    bbox: tuple[float, float, float, float],
    det_score: float,
    kps: np.ndarray | None,
    th: QualityThresholds = QualityThresholds(),
) -> tuple[bool, str | None, dict]:
    x1, y1, x2, y2 = bbox
    short_side = min(x2 - x1, y2 - y1)
    sharp = sharpness_112(face_rgb)
    yaw = estimate_yaw_from_landmarks(kps) if kps is not None else 0.0
    q = {"face_px": round(short_side, 1), "det_score": round(det_score, 3),
         "sharpness": round(sharp, 1), "yaw_deg": round(yaw, 1)}
    if short_side < th.min_face_px:
        return False, "too_small", q
    if det_score < th.min_det_score:
        return False, "low_detection_score", q
    if sharp < th.min_sharpness:
        return False, "blurry", q
    if abs(yaw) > th.max_abs_yaw_deg:
        return False, "extreme_angle", q
    return True, None, q
