"""Face detection + embedding via InsightFace (SCRFD detector, ArcFace recogniser).

Model packs (download to ~/.insightface/models on first use; non-commercial research licence):
    buffalo_l    ResNet-50 ArcFace trained on WebFace600K (default, ~330MB)
    antelopev2   ResNet-100 ArcFace trained on Glint360K (stronger, slower, ~360MB)

Embeddings from different packs are not comparable, so a workspace is tied to one pack.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

MODEL_PACKS = ("buffalo_l", "antelopev2")


def _providers() -> list[str]:
    import onnxruntime as ort

    avail = ort.get_available_providers()
    order = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in avail] or avail


def _flatten_nested_pack(name: str) -> bool:
    """antelopev2's zip unpacks to models/antelopev2/antelopev2/*.onnx; move files up a level."""
    root = Path.home() / ".insightface" / "models" / name
    nested = root / name
    if not nested.is_dir():
        return False
    for f in nested.iterdir():
        shutil.move(str(f), root / f.name)
    nested.rmdir()
    return True


class FaceEngine:
    def __init__(self, model_pack: str = "buffalo_l", det_size: int = 640, providers=None):
        from insightface.app import FaceAnalysis

        if model_pack not in MODEL_PACKS:
            raise ValueError(f"model_pack must be one of {MODEL_PACKS}")
        self.model_pack = model_pack
        self.providers = providers or _providers()

        def build():
            app = FaceAnalysis(name=model_pack, allowed_modules=["detection", "recognition"],
                               providers=self.providers)
            app.prepare(ctx_id=0, det_size=(det_size, det_size))
            return app

        try:
            self.app = build()
        except (AssertionError, KeyError):
            if not _flatten_nested_pack(model_pack):
                raise
            self.app = build()

    def __call__(self, rgb: np.ndarray):
        """Return InsightFace face objects (bbox, det_score, kps, normed_embedding)."""
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        return self.app.get(bgr)
