"""Faces you draw by hand in the photo viewer, for faces the detector missed or boxed badly.

The detector is run again on a padded, upscaled patch around your box, which finds small or
side-on faces it skipped at full-photo scale. If it finds one there, that face's landmarks and
embedding are used, so it can be grouped and matched like any other. If not, the box is kept
without an embedding: you can still name it (it counts for training crops), but it is never
used for matching.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sqlmodel import func, select

from studio.db import Face, Photo
from studio.faces.quality import QualityThresholds, assess
from studio.ingest import load_rgb
from studio.library.core import Library
from studio.library.importer import CROP_PX

PATCH_PX = 640


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def add_face(lib: Library, photo_id: int, box: tuple[float, float, float, float]) -> Face:
    """box is (x1, y1, x2, y2) in the photo's stored pixel size (Photo.width/height)."""
    with lib.session() as s:
        photo = s.get(Photo, photo_id)
        if photo is None:
            raise KeyError(photo_id)
    rgb = load_rgb(Path(photo.path), 2048)
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = (max(0.0, min(float(v), lim)) for v, lim in zip(box, (w, h, w, h)))
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        raise ValueError("box is too small")

    # padded patch around the box, upscaled so a small face fills more of the detector input
    bw, bh = x2 - x1, y2 - y1
    px1, py1 = int(max(0, x1 - bw)), int(max(0, y1 - bh))
    px2, py2 = int(min(w, x2 + bw)), int(min(h, y2 + bh))
    patch = rgb[py1:py2, px1:px2]
    scale = PATCH_PX / max(patch.shape[:2])
    big = np.asarray(Image.fromarray(patch).resize(
        (max(1, round(patch.shape[1] * scale)), max(1, round(patch.shape[0] * scale)))))
    engine = lib.face_engine()
    best, best_iou = None, 0.2
    for d in engine(big):
        bx = [v / scale for v in d.bbox]
        bx = [bx[0] + px1, bx[1] + py1, bx[2] + px1, bx[3] + py1]
        iou = _iou(bx, (x1, y1, x2, y2))
        if iou > best_iou:
            best, best_iou, best_box = d, iou, bx

    if best is not None:
        x1, y1, x2, y2 = best_box
        kps = np.asarray(best.kps, float) / scale + [px1, py1]
        det, emb = float(best.det_score), np.asarray(best.normed_embedding, np.float32)
    else:
        kps, det, emb = None, 0.0, np.zeros(512, np.float32)

    crop = rgb[int(y1):int(y2), int(x1):int(x2)]
    if kps is not None:
        ok, why, q = assess(crop, (x1, y1, x2, y2), det, kps, QualityThresholds())
    else:
        ok, why, q = False, "no_embedding", {"sharpness": 0.0, "yaw_deg": 0.0}

    with lib.write_lock, lib.session() as s:
        idx = (s.exec(select(func.max(Face.idx)).where(Face.photo_id == photo_id)).one() or 0) + 1
        crop_name = f"{photo.sha256[:2]}/{photo.sha256}_{idx}.jpg"
        (lib.crops / photo.sha256[:2]).mkdir(exist_ok=True)
        Image.fromarray(crop).resize((CROP_PX, CROP_PX)).save(lib.crops / crop_name, quality=88)
        f = Face(photo_id=photo_id, idx=idx, x1=x1, y1=y1, x2=x2, y2=y2, det_score=det,
                 sharpness=q["sharpness"], yaw=q["yaw_deg"], usable=ok, reject_reason=why,
                 model_pack=getattr(engine, "model_pack", "buffalo_l"), crop=crop_name,
                 embedding=emb.tobytes(), manual=True)
        s.add(f)
        photo = s.get(Photo, photo_id)
        photo.face_count += 1
        s.add(photo)
        s.commit()
        s.refresh(f)
        return f
