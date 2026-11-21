"""Face-level fixes: one person per photo, move kicks the duplicate, not-a-face, drawn boxes."""

import types

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import select
from studio.api.main import create_app
from studio.db import Face, Photo, make_engine
from studio.library.core import Library
from studio.people import service

RNG = np.random.default_rng(3)
ME, BRO = RNG.standard_normal(512), RNG.standard_normal(512)


def _v(base, noise=0.4):
    v = base + noise * RNG.standard_normal(512)
    return (v / np.linalg.norm(v)).astype(np.float32)


class Detector:
    """Sees the whole 160px photo (upscaled to 640) and finds one face at 40..100."""

    model_pack = "fake"

    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        s = max(h, w) / 160
        kps = np.array([[50, 60], [90, 60], [70, 75], [55, 90], [85, 90]], float) * s
        return [types.SimpleNamespace(bbox=[40 * s, 40 * s, 100 * s, 100 * s], det_score=0.9,
                                      kps=kps, normed_embedding=_v(BRO))]


@pytest.fixture
def lib(tmp_path):
    lib = Library(tmp_path / "home", face_engine_factory=Detector)
    img = tmp_path / "p.jpg"
    Image.fromarray((RNG.random((160, 160, 3)) * 255).astype(np.uint8)).save(img)
    with lib.session() as s:
        for i in range(6):
            p = Photo(sha256=f"{i:064x}", path=str(img), filename=f"p{i}.jpg", width=160,
                      height=160, face_count=2)
            s.add(p)
            s.flush()
            # photo 0 and 1: me + brother; others: just me. All in one unnamed group,
            # which is what happens when clustering lumps brothers together.
            s.add(Face(photo_id=p.id, idx=0, x1=0, y1=0, x2=50, y2=50, crop=f"c{i}0",
                       embedding=_v(ME).tobytes(), group_key=1))
            if i < 2:
                s.add(Face(photo_id=p.id, idx=1, x1=60, y1=0, x2=110, y2=50, crop=f"c{i}1",
                           embedding=_v(BRO, 1.2).tobytes(), group_key=1))
        s.commit()
    return lib


def _faces(lib, **kw):
    with lib.session() as s:
        q = select(Face)
        for k, v in kw.items():
            q = q.where(getattr(Face, k) == v)
        return s.exec(q).all()


def test_naming_a_group_files_one_face_per_photo(lib):
    me = service.name_groups(lib, [1], "Me")
    mine = _faces(lib, person_id=me)
    assert len(mine) == 6 and len({f.photo_id for f in mine}) == 6
    assert all(f.idx == 0 for f in mine)  # the brother faces were left out
    assert len(_faces(lib, person_id=None)) == 2
    assert service.conflicts(lib) == []


def test_move_kicks_duplicate_and_similar_finds_the_rest(lib):
    # simulate the old behaviour: everything in the group filed as Me
    me = service.move(lib, [f.id for f in _faces(lib)], "Me")
    assert len(service.conflicts(lib)) == 2
    bro_faces = [f for f in _faces(lib) if f.idx == 1]
    # in the viewer you click the brother's face and create a new person
    bro = service.move(lib, [bro_faces[0].id], "Brother")
    # the other brother face in "Me" is suggested as the same kind of mistake
    hits = service.similar_in(lib, bro_faces[0].id, me)
    assert [f.id for f, _ in hits] == [bro_faces[1].id]
    service.move(lib, [bro_faces[1].id], "Brother")
    assert service.conflicts(lib) == []
    # moving my face from photo 0 onto Brother would put Brother twice: the old one is kicked
    my0 = next(f for f in _faces(lib, person_id=me) if f.photo_id == bro_faces[0].photo_id)
    service.move(lib, [my0.id], "Brother")
    assert {f.id for f in _faces(lib, person_id=bro)} == {my0.id, bro_faces[1].id}


def test_not_a_face_hides_it_everywhere(lib, tmp_path):
    c = TestClient(create_app(lib))
    f = _faces(lib)[0]
    c.post("/api/faces/not-face", json={"face_ids": [f.id]})
    assert all(x["id"] != f.id for g in c.get("/api/groups").json() for x in g["faces"])
    assert c.get("/api/stats").json()["unsorted"] == 0
    assert c.post("/api/groups/refresh").json()["groups"] == 1
    assert all(x["id"] != f.id for g in c.get("/api/groups").json() for x in g["faces"])
    photo = c.get(f"/api/photos/{f.photo_id}").json()
    assert next(x for x in photo["faces"] if x["id"] == f.id)["not_face"] is True
    c.post("/api/faces/not-face", json={"face_ids": [f.id], "value": False})
    assert _faces(lib, id=f.id)[0].not_face is False


def test_draw_a_face_box(lib):
    c = TestClient(create_app(lib))
    photo = _faces(lib)[0].photo_id
    r = c.post(f"/api/photos/{photo}/faces", json={"box": [50, 50, 110, 110]}).json()
    assert r["found"] is True
    f = _faces(lib, id=r["face_id"])[0]
    assert f.manual and abs(f.x1 - 40) < 1 and np.any(f.vector())
    # a box where the detector finds nothing is kept, without an embedding
    r = c.post(f"/api/photos/{photo}/faces", json={"box": [0, 100, 30, 150]}).json()
    assert r["found"] is False
    c.post("/api/faces/move", json={"face_ids": [r["face_id"]], "name": "Me"})
    assert c.post("/api/people/auto-file").status_code == 200  # zero vectors are skipped
    assert c.post(f"/api/photos/{photo}/faces", json={"box": [1, 1, 3, 3]}).status_code == 400


def test_delete_person_returns_faces(lib):
    me = service.name_groups(lib, [1], "Me")
    service.delete_person(lib, me)
    assert _faces(lib, person_id=me) == []
    assert len(_faces(lib, source=None)) == 8


def test_old_database_gets_new_columns(tmp_path):
    import sqlite3

    db = tmp_path / "old.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE face (id INTEGER PRIMARY KEY, photo_id INTEGER, embedding BLOB)")
    con.commit()
    con.close()
    make_engine(f"sqlite:///{db}")
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(face)")}
    assert {"not_face", "manual", "no_training"} <= cols
