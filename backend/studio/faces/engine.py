"""Face detection + embedding via InsightFace (SCRFD detector, ArcFace recogniser).

Weights (buffalo_l, ~330MB) download to ~/.insightface on first use. They are licensed for
non-commercial research use, which fits this personal project.
"""

from __future__ import annotations

import numpy as np


def _providers() -> list[str]:
    import onnxruntime as ort

    avail = ort.get_available_providers()
    order = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in avail] or avail


class FaceEngine:
    def __init__(self, model_pack: str = "buffalo_l", det_size: int = 640, providers=None):
        from insightface.app import FaceAnalysis

        self.providers = providers or _providers()
        self.app = FaceAnalysis(
            name=model_pack,
            allowed_modules=["detection", "recognition"],
            providers=self.providers,
        )
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

    def __call__(self, rgb: np.ndarray):
        """Return InsightFace face objects (bbox, det_score, kps, normed_embedding)."""
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        return self.app.get(bgr)
