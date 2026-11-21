"""Studio API: library import, photos, people. Serves the built React app at / when present.

Run: uv run uvicorn studio.api.main:app --port 8000   (or `make app`)
"""

from __future__ import annotations

import io
import re
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import col, select

from studio import __version__
from studio.config import settings
from studio.db import Face, ImportJob, Person, Photo
from studio.library import importer, manual
from studio.library.core import Library
from studio.people import service as people

ASSET_RE = re.compile(r"^[0-9a-f]{2}/[0-9a-f]{64}(_\d+)?\.jpg$")
FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"


class FolderImport(BaseModel):
    path: str


class FaceIds(BaseModel):
    face_ids: list[int]


class RemoveReq(FaceIds):
    person_id: int


class MoveReq(FaceIds):
    name: str


class TrainingReq(FaceIds):
    exclude: bool


class NameGroups(BaseModel):
    keys: list[int]
    name: str


class RenameReq(BaseModel):
    name: str


class AnswerReq(BaseModel):
    face_id: int
    name: str | None = None


class NotFaceReq(FaceIds):
    value: bool = True


class BoxReq(BaseModel):
    box: tuple[float, float, float, float]


class GroupReq(BaseModel):
    link_sim: float = 0.25


def _face_json(f: Face, photo: Photo | None = None) -> dict:
    return {"id": f.id, "photo_id": f.photo_id, "crop": f"/media/crops/{f.crop}",
            "person_id": f.person_id, "source": f.source, "guess_person_id": f.guess_person_id,
            "score": f.score, "usable": f.usable, "reject_reason": f.reject_reason,
            "second_pass": f.second_pass, "no_training": f.no_training,
            "not_face": f.not_face, "manual": f.manual,
            "filename": photo.filename if photo else None,
            "thumb": f"/media/thumbs/{photo.thumb}" if photo and photo.thumb else None}


def create_app(lib: Library | None = None) -> FastAPI:
    lib = lib or Library(settings.home)
    app = FastAPI(title="Studio", version=__version__)
    app.state.lib = lib

    # health -------------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "version": __version__, "home": str(lib.home)}

    # imports ------------------------------------------------------------------
    @app.post("/api/imports/folder")
    def import_folder(req: FolderImport) -> dict:
        p = Path(req.path).expanduser()
        if not p.exists():
            raise HTTPException(400, f"{p} does not exist")
        return {"job_id": importer.start_import(lib, [p], source=str(p))}

    @app.post("/api/imports/upload")
    async def import_upload(files: list[UploadFile] = File(...)) -> dict:
        staging = lib.uploads / "_staging" / uuid.uuid4().hex
        staging.mkdir(parents=True)
        for uf in files:
            name = Path(uf.filename or "upload").name
            with open(staging / name, "wb") as out:
                shutil.copyfileobj(uf.file, out)
        job = importer.start_import(lib, [staging], source=f"{len(files)} dropped files",
                                    copy_into_library=True)
        return {"job_id": job}

    @app.get("/api/imports")
    def imports() -> list[ImportJob]:
        with lib.session() as s:
            return s.exec(select(ImportJob).order_by(col(ImportJob.id).desc()).limit(20)).all()

    @app.get("/api/imports/{job_id}")
    def import_status(job_id: int) -> ImportJob:
        with lib.session() as s:
            job = s.get(ImportJob, job_id)
            if not job:
                raise HTTPException(404)
            return job

    @app.post("/api/imports/{job_id}/cancel")
    def import_cancel(job_id: int) -> dict:
        importer.cancel(job_id)
        return {"ok": True}

    # photos -------------------------------------------------------------------
    @app.get("/api/photos")
    def photos(offset: int = 0, limit: int = 200, person_id: int | None = None) -> dict:
        with lib.session() as s:
            q = select(Photo)
            if person_id is not None:
                q = q.where(col(Photo.id).in_(select(Face.photo_id).where(Face.person_id == person_id)))
            total = s.exec(select(func.count()).select_from(q.subquery())).one()
            rows = s.exec(q.order_by(col(Photo.taken_at).desc(), col(Photo.id).desc())
                          .offset(offset).limit(limit)).all()
            return {"total": total, "items": [
                {"id": p.id, "filename": p.filename, "thumb": f"/media/thumbs/{p.thumb}",
                 "width": p.width, "height": p.height, "face_count": p.face_count,
                 "taken_at": p.taken_at} for p in rows]}

    @app.get("/api/photos/{photo_id}")
    def photo_detail(photo_id: int) -> dict:
        with lib.session() as s:
            p = s.get(Photo, photo_id)
            if not p:
                raise HTTPException(404)
            faces = s.exec(select(Face).where(Face.photo_id == p.id)).all()
            names = {x.id: x.name for x in s.exec(select(Person)).all()}
            return {"id": p.id, "filename": p.filename, "path": p.path, "width": p.width,
                    "height": p.height, "taken_at": p.taken_at,
                    "preview": f"/api/photos/{p.id}/preview",
                    "faces": [{**_face_json(f), "box": [f.x1, f.y1, f.x2, f.y2],
                               "person": names.get(f.person_id)} for f in faces]}

    @app.get("/api/photos/{photo_id}/preview")
    def photo_preview(photo_id: int):
        from PIL import Image

        from studio.ingest import load_rgb

        with lib.session() as s:
            p = s.get(Photo, photo_id)
        if not p or not Path(p.path).exists():
            raise HTTPException(404)
        buf = io.BytesIO()
        Image.fromarray(load_rgb(Path(p.path), 2048)).save(buf, "JPEG", quality=88)
        return Response(buf.getvalue(), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})

    @app.get("/media/{kind}/{a}/{b}")
    def media(kind: str, a: str, b: str):
        rel = f"{a}/{b}"
        root = {"thumbs": lib.thumbs, "crops": lib.crops}.get(kind)
        if root is None or not ASSET_RE.match(rel) or not (root / rel).exists():
            raise HTTPException(404)
        return FileResponse(root / rel, headers={"Cache-Control": "max-age=604800"})

    # people -------------------------------------------------------------------
    @app.get("/api/people")
    def people_list() -> list[dict]:
        with lib.session() as s:
            out = []
            for p in s.exec(select(Person)).all():
                faces = s.exec(select(Face).where(Face.person_id == p.id)).all()
                cover = next((f for f in faces if f.source == "confirmed" and f.usable), None) \
                    or (faces[0] if faces else None)
                out.append({"id": p.id, "name": p.name, "faces": len(faces),
                            "confirmed": sum(f.source == "confirmed" for f in faces),
                            "no_training": sum(f.no_training for f in faces),
                            "cover": f"/media/crops/{cover.crop}" if cover else None})
            return sorted(out, key=lambda x: -x["faces"])

    @app.get("/api/people/{person_id}/faces")
    def person_faces(person_id: int) -> list[dict]:
        with lib.session() as s:
            rows = s.exec(select(Face, Photo).join(Photo).where(Face.person_id == person_id)
                          .order_by(col(Face.source).desc(), col(Photo.taken_at).desc())).all()
            return [_face_json(f, p) for f, p in rows]

    @app.post("/api/people/{person_id}/rename")
    def person_rename(person_id: int, req: RenameReq) -> dict:
        return {"person_id": people.rename(lib, person_id, req.name)}

    @app.get("/api/groups")
    def groups() -> list[dict]:
        with lib.session() as s:
            rows = s.exec(select(Face, Photo).join(Photo).where(Face.group_key.is_not(None),
                                                                 Face.not_face.is_(False))).all()
        out: dict[int, list] = {}
        for f, p in rows:
            out.setdefault(f.group_key, []).append(_face_json(f, p))
        return [{"key": k, "faces": v} for k, v in sorted(out.items(), key=lambda kv: -len(kv[1]))]

    @app.post("/api/groups/refresh")
    def groups_refresh(req: GroupReq | None = None) -> dict:
        return {"groups": people.regroup(lib, (req or GroupReq()).link_sim)}

    @app.post("/api/groups/name")
    def groups_name(req: NameGroups) -> dict:
        pid = people.name_groups(lib, req.keys, req.name)
        filed = people.auto_file(lib)  # new prototypes: try filing the rest
        return {"person_id": pid, "auto_filed": filed}

    @app.get("/api/review")
    def review() -> list[dict]:
        with lib.session() as s:
            rows = s.exec(select(Face, Photo).join(Photo).where(Face.source == "review",
                                                               Face.not_face.is_(False))
                          .order_by(col(Face.score).desc())).all()
            return [_face_json(f, p) for f, p in rows]

    @app.get("/api/unsorted")
    def unsorted(limit: int = 500) -> list[dict]:
        with lib.session() as s:
            rows = s.exec(select(Face, Photo).join(Photo).where(
                Face.person_id.is_(None), Face.group_key.is_(None),
                Face.source.is_(None), Face.not_face.is_(False))
                .order_by(col(Face.usable).desc(), col(Face.id)).limit(limit)).all()
            return [_face_json(f, p) for f, p in rows]

    @app.post("/api/people/auto-file")
    def auto_file() -> dict:
        return {"auto_filed": people.auto_file(lib)}

    @app.post("/api/faces/remove")
    def faces_remove(req: RemoveReq) -> dict:
        people.remove(lib, req.person_id, req.face_ids)
        return {"ok": True}

    @app.post("/api/faces/move")
    def faces_move(req: MoveReq) -> dict:
        return {"person_id": people.move(lib, req.face_ids, req.name)}

    @app.post("/api/faces/training")
    def faces_training(req: TrainingReq) -> dict:
        people.set_training(lib, req.face_ids, req.exclude)
        return {"ok": True}

    @app.post("/api/faces/answer")
    def faces_answer(req: AnswerReq) -> dict:
        people.answer(lib, req.face_id, req.name)
        return {"ok": True}

    @app.post("/api/faces/not-face")
    def faces_not_face(req: NotFaceReq) -> dict:
        people.set_not_face(lib, req.face_ids, req.value)
        return {"ok": True}

    @app.get("/api/faces/{face_id}/similar")
    def faces_similar(face_id: int, person_id: int) -> list[dict]:
        hits = people.similar_in(lib, face_id, person_id)
        with lib.session() as s:
            return [{**_face_json(f, s.get(Photo, f.photo_id)), "sim": round(sim, 3)}
                    for f, sim in hits]

    @app.post("/api/photos/{photo_id}/faces")
    def photo_add_face(photo_id: int, req: BoxReq) -> dict:
        try:
            f = manual.add_face(lib, photo_id, req.box)
        except KeyError:
            raise HTTPException(404) from None
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return {"face_id": f.id, "found": f.reject_reason != "no_embedding"}

    @app.get("/api/conflicts")
    def conflicts() -> list[dict]:
        rows = people.conflicts(lib)
        with lib.session() as s:
            names = {x.id: x.name for x in s.exec(select(Person)).all()}
            return [{"photo_id": r["photo_id"], "person_id": r["person_id"],
                     "person": names.get(r["person_id"]),
                     "faces": [_face_json(f, s.get(Photo, f.photo_id)) for f in r["faces"]]}
                    for r in rows]

    @app.post("/api/people/{person_id}/delete")
    def person_delete(person_id: int) -> dict:
        people.delete_person(lib, person_id)
        return {"ok": True}

    @app.get("/api/stats")
    def stats() -> dict:
        with lib.session() as s:
            def n(q):
                return s.exec(select(func.count()).select_from(q.subquery())).one()

            return {"photos": n(select(Photo)), "faces": n(select(Face)),
                    "people": n(select(Person)),
                    "review": n(select(Face).where(Face.source == "review")),
                    "groups": len({k for k in s.exec(select(Face.group_key).where(
                        Face.group_key.is_not(None))).all()}),
                    "unsorted": n(select(Face).where(Face.person_id.is_(None),
                                                     Face.group_key.is_(None),
                                                     Face.source.is_(None),
                                                     Face.not_face.is_(False))),
                    "conflicts": len(people.conflicts(lib))}

    # frontend -----------------------------------------------------------------
    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

    return app


def __getattr__(name):  # lazy `app` so importing this module does not create ~/StudioData
    if name == "app":
        return create_app()
    raise AttributeError(name)
