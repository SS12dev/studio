"""Decode any still image into an RGB uint8 array with EXIF orientation applied.

Phase 0 scope: JPEG, PNG, TIFF, WebP, HEIC/HEIF (iPhone), and camera RAW when rawpy is installed.
EXR and 16-bit/linear handling arrive with the export work in phase 2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

RAW_EXTS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".rw2", ".orf"}
PIL_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".heic", ".heif"}
SUPPORTED_EXTS = RAW_EXTS | PIL_EXTS

try:  # iPhone HEIC support
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pass


def iter_images(root: Path):
    """Yield supported image files under root, skipping hidden files, in stable order."""
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def load_rgb(path: Path, max_side: int | None = 2048) -> np.ndarray:
    """Return HxWx3 uint8 RGB. Downscale so the long side is at most max_side (faces don't need more)."""
    ext = path.suffix.lower()
    if ext in RAW_EXTS:
        import rawpy  # optional dependency

        with rawpy.imread(str(path)) as raw:
            img = Image.fromarray(raw.postprocess(use_camera_wb=True, output_bps=8))
    else:
        img = ImageOps.exif_transpose(Image.open(path))
    img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(img)
