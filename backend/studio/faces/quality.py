"""Cheap face quality checks so bad crops are set aside instead of misclassified."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QualityThresholds:
    min_face_px: int = 64  # shorter side of the face box
    min_det_score: float = 0.6
    min_sharpness: float = 40.0  # variance of Laplacian on the grey face crop
    max_abs_yaw_deg: float = 60.0


def laplacian_variance(grey: np.ndarray) -> float:
    """Variance of a 4-neighbour Laplacian; low values mean blur. No OpenCV needed."""
    g = grey.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def estimate_yaw_from_landmarks(kps: np.ndarray) -> float:
    """Rough yaw (degrees) from 5-point landmarks: eyes, nose, mouth corners.

    Uses where the nose sits between the eyes: 0 when centred, approaching +/-90 when it
    reaches one eye. Good enough to reject near-profile faces, not a real head-pose estimate.
    """
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_span = right_eye[0] - left_eye[0]
    if abs(eye_span) < 1e-6:
        return 90.0
    rel = (nose[0] - left_eye[0]) / eye_span  # 0.5 when frontal
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
    grey = face_rgb.mean(axis=2) if face_rgb.ndim == 3 else face_rgb
    sharp = laplacian_variance(grey) if min(grey.shape) >= 3 else 0.0
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
