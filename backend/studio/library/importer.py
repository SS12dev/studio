"""Import photos into the library: dedupe, thumbnail, detect + embed faces, auto-file.

One background thread per import job; photos are committed one at a time so the UI can show
progress and an interrupted import keeps everything done so far.
"""

from __future__ import annotations

import hashlib
import shutil
import threading
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from sqlmodel import select

from studio.db import Face, ImportJob, Photo, now
from studio.faces.quality import QualityThresholds, assess
from studio.ingest import SUPPORTED_EXTS, iter_images, load_rgb
from studio.library.core import Library

THUMB_PX = 512
CROP_PX = 160
_cancel: set[int] = set()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _taken_at(path: Path) -> datetime | None:
    try:
        exif = Image.open(path).getexif()
        raw = exif.get_ifd(0x8769).get(36867) or exif.get(306)  # DateTimeOriginal, DateTime
        return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S") if raw else None
    except Exception:  # noqa: BLE001 - no or odd EXIF is normal
        return None


def collect(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = p.expanduser()
        if p.is_dir():
            out.extend(iter_images(p))
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            out.append(p)
    return out


def start_import(lib: Library, paths: list[Path], source: str, copy_into_library: bool = False) -> int:
    with lib.session() as s:
        job = ImportJob(source=source)
        s.add(job)
        s.commit()
        job_id = job.id
    t = threading.Thread(target=_run, args=(lib, job_id, paths, copy_into_library), daemon=True)
    t.start()
    return job_id


def cancel(job_id: int) -> None:
    _cancel.add(job_id)


def _update(lib: Library, job_id: int, **kw) -> None:
    with lib.session() as s:
        job = s.get(ImportJob, job_id)
        for k, v in kw.items():
            setattr(job, k, v)
        s.add(job)
        s.commit()


def _run(lib: Library, job_id: int, paths: list[Path], copy_into_library: bool) -> None:
    try:
        files = collect(paths)
        _update(lib, job_id, status="running", total=len(files))
        counts = {"processed": 0, "added": 0, "duplicates": 0, "failed": 0, "faces": 0}
        th = QualityThresholds()
        for path in files:
            if job_id in _cancel:
                _update(lib, job_id, status="cancelled", finished_at=now(), **counts)
                return
            try:
                result = _import_one(lib, job_id, path, copy_into_library, th)
                counts["added" if result is not None else "duplicates"] += 1
                counts["faces"] += result or 0
            except Exception as e:  # noqa: BLE001 - one bad file must not stop the import
                counts["failed"] += 1
                _update(lib, job_id, message=f"{path.name}: {type(e).__name__}: {e}")
            counts["processed"] += 1
            _update(lib, job_id, **counts)

        from studio.people.service import auto_file

        filed = auto_file(lib, import_id=job_id)
        _update(lib, job_id, status="done", finished_at=now(), auto_filed=filed, **counts)
    except Exception as e:  # noqa: BLE001
        _update(lib, job_id, status="error", finished_at=now(),
                message=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
    finally:
        _cancel.discard(job_id)
        if copy_into_library:  # uploads were staged in a temp folder
            for p in paths:
                if p.is_dir() and p.parent == lib.uploads / "_staging":
                    shutil.rmtree(p, ignore_errors=True)


def _import_one(lib: Library, job_id: int, path: Path, copy: bool, th: QualityThresholds):
    """Returns the number of faces found, or None if the photo was already in the library."""
    digest = sha256_file(path)
    with lib.session() as s:
        if s.exec(select(Photo.id).where(Photo.sha256 == digest)).first():
            return None
    if copy:
        dest = lib.uploads / digest[:2] / f"{digest}{path.suffix.lower()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        stored = dest
    else:
        stored = path

    rgb = load_rgb(stored, 2048)
    h, w = rgb.shape[:2]
    thumb_name = f"{digest[:2]}/{digest}.jpg"
    (lib.thumbs / digest[:2]).mkdir(exist_ok=True)
    timg = Image.fromarray(rgb)
    timg.thumbnail((THUMB_PX, THUMB_PX))
    timg.save(lib.thumbs / thumb_name, quality=85)

    engine = lib.face_engine()
    detected = engine(rgb)
    pack = getattr(engine, "model_pack", "buffalo_l")
    with lib.session() as s:
        photo = Photo(sha256=digest, path=str(stored.resolve()), filename=path.name, width=w,
                      height=h, taken_at=_taken_at(stored), import_id=job_id, thumb=thumb_name,
                      face_count=len(detected))
        s.add(photo)
        s.flush()
        for i, f in enumerate(detected):
            x1, y1, x2, y2 = [float(v) for v in f.bbox]
            crop = rgb[max(0, int(y1)):max(0, int(y2)), max(0, int(x1)):max(0, int(x2))]
            ok, why, q = assess(crop, (x1, y1, x2, y2), float(f.det_score), f.kps, th)
            crop_name = f"{digest[:2]}/{digest}_{i}.jpg"
            if crop.size:
                (lib.crops / digest[:2]).mkdir(exist_ok=True)
                Image.fromarray(crop).resize((CROP_PX, CROP_PX)).save(lib.crops / crop_name,
                                                                      quality=88)
            s.add(Face(photo_id=photo.id, idx=i, x1=x1, y1=y1, x2=x2, y2=y2,
                       det_score=float(f.det_score), sharpness=q["sharpness"], yaw=q["yaw_deg"],
                       usable=ok, reject_reason=why, model_pack=pack, crop=crop_name,
                       embedding=np.asarray(f.normed_embedding, np.float32).tobytes()))
        s.commit()
    return len(detected)
