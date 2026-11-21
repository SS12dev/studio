from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Face:
    photo: str  # path of the source photo
    index: int  # face number within the photo
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    det_score: float
    embedding: np.ndarray | None = None  # 512-d, L2-normalised
    quality: dict = field(default_factory=dict)
    usable: bool = True
    reject_reason: str | None = None

    @property
    def id(self) -> str:
        return f"{self.photo}#{self.index}"
