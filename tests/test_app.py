"""App flow: import a folder, group faces, name, auto-file, fix by hand. Fake face model."""

import time
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from studio.api.main import create_app
from studio.library.core import Library

RNG = np.random.default_rng(7)
PEOPLE = {k: RNG.standard_normal(512) for k in ("red", "green", "blue")}


def _vec(who: str) -> np.ndarray:
    v = PEOPLE[who] + 0.5 * RNG.standard_normal(512)
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeFaces:
    """Reads who is in a photo from its dominant colour channel; one face per photo."""

    model_pack = "fake"

    def __call__(self, rgb):
        who = ("red", "green", "blue")[int(np.argmax(rgb.reshape(-1, 3).mean(0)))]
        kps = np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], float)
        return [types.SimpleNamespace(bbox=[10, 10, 110, 110], det_score=0.95, kps=kps,
                                      normed_embedding=_vec(who))]


def _photo(path, colour, seed):
    rng = np.random.default_rng(seed)
    img = (rng.random((160, 160, 3)) * 60).astype(np.uint8)
    img[..., colour] += 150
    Image.fromarray(img).save(path)


def _wait(c, job_id):
    for _ in range(200):
        j = c.get(f"/api/imports/{job_id}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError("import did not finish")


@pytest.fixture
def client(tmp_path):
    lib = Library(tmp_path / "home", face_engine_factory=FakeFaces)
    return TestClient(create_app(lib)), tmp_path


def test_import_group_name_autofile_and_fix(client):
    c, tmp = client
    batch1 = tmp / "batch1"
    batch1.mkdir()
    for i in range(8):
        _photo(batch1 / f"r{i}.jpg", 0, i)
        _photo(batch1 / f"g{i}.png", 1, 100 + i)
    job = _wait(c, c.post("/api/imports/folder", json={"path": str(batch1)}).json()["job_id"])
    assert job["status"] == "done" and job["added"] == 16 and job["faces"] == 16

    # re-importing the same folder only finds duplicates
    again = _wait(c, c.post("/api/imports/folder", json={"path": str(batch1)}).json()["job_id"])
    assert again["added"] == 0 and again["duplicates"] == 16

    assert c.post("/api/groups/refresh").json()["groups"] == 2
    groups = c.get("/api/groups").json()
    assert sorted(len(g["faces"]) for g in groups) == [8, 8]
    red_key = next(g["key"] for g in groups if g["faces"][0]["filename"].startswith("r"))
    green_key = next(g["key"] for g in groups if g["key"] != red_key)
    shivam = c.post("/api/groups/name", json={"keys": [red_key], "name": "Shivam"}).json()["person_id"]
    c.post("/api/groups/name", json={"keys": [green_key], "name": "Lesa"})

    # a second import is auto-filed against the named people
    batch2 = tmp / "batch2"
    batch2.mkdir()
    for i in range(4):
        _photo(batch2 / f"r{i}.jpg", 0, 200 + i)
    job = _wait(c, c.post("/api/imports/folder", json={"path": str(batch2)}).json()["job_id"])
    assert job["auto_filed"] == 4
    faces = c.get(f"/api/people/{shivam}/faces").json()
    assert len(faces) == 12 and sum(f["source"] == "auto" for f in faces) == 4

    # manual fixes: not them (sticks), exclude from training, move, rename-merge
    wrong = faces[-1]["id"]
    c.post("/api/faces/remove", json={"person_id": shivam, "face_ids": [wrong]})
    c.post("/api/people/auto-file")
    assert wrong not in {f["id"] for f in c.get(f"/api/people/{shivam}/faces").json()}
    filt = faces[0]["id"]
    c.post("/api/faces/training", json={"face_ids": [filt], "exclude": True})
    people = {p["name"]: p for p in c.get("/api/people").json()}
    assert people["Shivam"]["no_training"] == 1
    c.post("/api/faces/move", json={"face_ids": [wrong], "name": "Vishal"})
    vishal = {p["name"]: p for p in c.get("/api/people").json()}["Vishal"]["id"]
    c.post(f"/api/people/{vishal}/rename", json={"name": "Lesa"})  # merge
    names = [p["name"] for p in c.get("/api/people").json()]
    assert sorted(names) == ["Lesa", "Shivam"]

    st = c.get("/api/stats").json()
    assert st["photos"] == 20 and st["people"] == 2


def test_upload_and_media(client):
    c, tmp = client
    p = tmp / "b0.jpg"
    _photo(p, 2, 1)
    with open(p, "rb") as fh:
        job_id = c.post("/api/imports/upload", files=[("files", ("b0.jpg", fh, "image/jpeg"))]).json()["job_id"]
    assert _wait(c, job_id)["added"] == 1
    photo = c.get("/api/photos").json()["items"][0]
    assert c.get(photo["thumb"]).status_code == 200
    detail = c.get(f"/api/photos/{photo['id']}").json()
    assert c.get(detail["faces"][0]["crop"]).status_code == 200
    assert c.get(detail["preview"]).headers["content-type"] == "image/jpeg"
    assert c.get("/media/thumbs/../library.sqlite").status_code == 404
    assert c.post("/api/imports/folder", json={"path": str(tmp / "nope")}).status_code == 400


def test_review_answer_records_calibration_data(client):
    c, tmp = client
    d = tmp / "d"
    d.mkdir()
    for i in range(6):
        _photo(d / f"r{i}.jpg", 0, 300 + i)
    _wait(c, c.post("/api/imports/folder", json={"path": str(d)}).json()["job_id"])
    c.post("/api/groups/refresh")
    key = c.get("/api/groups").json()[0]["key"]
    c.post("/api/groups/name", json={"keys": [key], "name": "Shivam"})
    # force one face into review to exercise the answer path
    lib = c.app.state.lib
    from studio.db import Face

    with lib.session() as s:
        f = s.get(Face, 1)
        guess = f.person_id
        f.person_id, f.source, f.guess_person_id, f.score, f.margin = None, "review", guess, 0.5, 0.05
        s.add(f)
        s.commit()
    assert len(c.get("/api/review").json()) == 1
    c.post("/api/faces/answer", json={"face_id": 1, "name": "Shivam"})
    assert c.get("/api/review").json() == []
    from sqlmodel import select
    from studio.db import Answer

    with lib.session() as s:
        assert s.exec(select(Answer)).one().correct is True


def test_copy_names_from_bakeoff_workspace(client):
    import json as _json

    from studio.cli import copy_names

    c, tmp = client
    d = tmp / "named"
    d.mkdir()
    for i in range(5):
        _photo(d / f"r{i}.jpg", 0, 500 + i)
    _wait(c, c.post("/api/imports/folder", json={"path": str(d)}).json()["job_id"])
    ws = tmp / "faces2"
    ws.mkdir()
    (ws / "people.json").write_text(_json.dumps(
        {"Shivam Singh": [f"{(d / f'r{i}.jpg').resolve()}#0" for i in range(3)]}))
    assert copy_names(c.app.state.lib, ws) == 3
    people = c.get("/api/people").json()
    assert people[0]["name"] == "Shivam Singh" and people[0]["faces"] == 5  # 3 named + 2 auto


def test_failed_files_are_listed_with_reasons(client):
    c, tmp = client
    d = tmp / "mixed"
    d.mkdir()
    _photo(d / "ok.jpg", 0, 1)
    (d / "broken1.jpg").write_bytes(b"not an image")
    (d / "broken2.jpg").write_bytes(b"also not")
    job = _wait(c, c.post("/api/imports/folder", json={"path": str(d)}).json()["job_id"])
    assert job["added"] == 1 and job["failed"] == 2
    why = c.get(f"/api/imports/{job['id']}/failures").json()
    assert why["total"] == 2 and why["groups"][0]["count"] == 2
    assert "UnidentifiedImageError" in why["groups"][0]["error"]
