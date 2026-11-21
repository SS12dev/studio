import json

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from studio.faces import curate
from studio.faces.cli import main
from studio.faces.ui import create_app
from studio.faces.workspace import Workspace


def make_ws(tmp_path):
    rng = np.random.default_rng(0)
    c = rng.standard_normal(512)
    faces, embs = [], []
    photos = tmp_path / "photos"
    photos.mkdir()
    for i in range(8):
        v = c + 0.5 * rng.standard_normal(512)
        embs.append(v / np.linalg.norm(v))
        p = photos / f"{i}.jpg"
        Image.fromarray(np.full((40, 40, 3), 30 * i, np.uint8)).save(p)
        faces.append({"id": f"{p}#0", "photo": str(p), "index": 0, "bbox": [0, 0, 40, 40],
                      "usable": True, "reject_reason": None, "quality": {}})
    ws = Workspace(tmp_path / "ws")
    ws.save_faces(faces, np.array(embs))
    for f in faces:
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(ws.crops / ws.crop_name(f["id"]))
    ws.save_clusters({"clusters": {"Person 1": [f["id"] for f in faces[:4]],
                                   "Person 2": [f["id"] for f in faces[4:5]]}})
    return ws, [f["id"] for f in faces]


def test_remove_sticks_across_assign(tmp_path):
    ws, ids = make_ws(tmp_path)
    main(["name", "Person 1", "Alex", "-w", str(ws.root)])
    main(["assign", "-w", str(ws.root)])
    auto = [f for f, src in curate.members(ws)["Alex"] if src == "auto"]
    assert auto, "same-person faces should be auto-filed"
    curate.remove(ws, "Alex", [ids[0], auto[0]])
    main(["assign", "-w", str(ws.root)])  # re-run must not file them under Alex again
    filed = {f for f, _ in curate.members(ws)["Alex"]}
    assert ids[0] not in filed and auto[0] not in filed


def test_move_training_rename_answer(tmp_path):
    ws, ids = make_ws(tmp_path)
    main(["name", "Person 1", "Alex", "-w", str(ws.root)])
    curate.assign_to(ws, "Sam", [ids[1]], src="Alex")
    assert ids[1] in ws.people["Sam"] and ids[1] not in ws.people["Alex"]
    curate.set_training(ws, [ids[2]], True)
    assert ids[2] in curate.no_training(ws)
    curate.set_training(ws, [ids[2]], False)
    assert not curate.no_training(ws)
    curate.rename(ws, "Sam", "Alex")  # merge
    assert set(ws.people) == {"Alex"} and ids[1] in ws.people["Alex"]
    main(["assign", "-w", str(ws.root)])
    fid = next(iter(curate._decisions(ws)))
    curate.answer(ws, fid, None)
    assert ws.answers[-1]["correct"] is False
    assert "Alex" in curate.not_person(ws)[fid]


def test_ui_api(tmp_path):
    ws, ids = make_ws(tmp_path)
    main(["name", "Person 1", "Alex", "-w", str(ws.root)])
    c = TestClient(create_app(ws))
    assert "People" in c.get("/").text
    st = c.get("/api/state").json()
    assert len(st["people"]["Alex"]) == 4 and "Person 2" in st["unnamed_groups"]
    crop = st["people"]["Alex"][0]["crop"]
    assert c.get(f"/crops/{crop}").status_code == 200
    assert c.get("/crops/..%2Fpeople.json").status_code == 404
    assert c.get("/photo", params={"face": ids[0]}).headers["content-type"] == "image/jpeg"
    assert c.get("/photo", params={"face": "/etc/passwd#0"}).status_code == 404
    c.post("/api/remove", json={"person": "Alex", "faces": [ids[0]]})
    c.post("/api/training", json={"faces": [ids[1]], "exclude": True})
    c.post("/api/move", json={"dst": "Sam", "faces": ws.clusters()["clusters"]["Person 2"]})
    st = c.get("/api/state").json()
    assert len(st["people"]["Alex"]) == 3 and st["people"]["Sam"]
    assert any(t["no_training"] for t in st["people"]["Alex"])
    assert "Person 2" not in st["unnamed_groups"]
