"""End-to-end CLI flow on a fake workspace (no face model needed)."""

import json

import numpy as np

from studio.faces.cli import main
from studio.faces.workspace import Workspace


def test_cluster_name_assign_confirm(tmp_path, capsys):
    rng = np.random.default_rng(0)
    centres = [rng.standard_normal(512) for _ in range(2)]
    faces, embs = [], []
    for p, c in enumerate(centres):
        for i in range(8):
            v = c + 0.5 * rng.standard_normal(512)  # cos(same person) ~0.8
            embs.append(v / np.linalg.norm(v))
            faces.append({"id": f"/photos/p{p}_{i}.jpg#0", "photo": f"/photos/p{p}_{i}.jpg",
                          "index": 0, "bbox": [0, 0, 100, 100], "usable": True,
                          "reject_reason": None, "quality": {}})
    ws = Workspace(tmp_path)
    ws.save_faces(faces, np.array(embs))

    main(["cluster", "-w", str(tmp_path)])
    clusters = ws.clusters()["clusters"]
    named = [k for k in clusters if k.startswith("Person")]
    assert len(named) == 2

    # Name only one cluster, keeping half its faces back to test auto-assignment.
    first = named[0]
    ids = clusters[first]
    ws.save_clusters({"clusters": {first: ids[:4]}, "merge_suggestions": []})
    main(["name", first, "Alex", "-w", str(tmp_path)])
    main(["assign", "-w", str(tmp_path)])
    decisions = json.loads((tmp_path / "decisions.json").read_text())
    assert all(decisions[i]["route"] == "auto" and decisions[i]["person"] == "Alex"
               for i in ids[4:])
    other = [f["id"] for f in faces if f["id"] not in ids]
    assert all(decisions[i]["route"] == "new" for i in other)

    main(["confirm", ids[4], "Alex", "-w", str(tmp_path)])
    assert ws.answers[-1]["correct"] is True
    assert (tmp_path / "report.html").exists()


def test_scan_appends(tmp_path, monkeypatch):
    """A second scan adds new photos and keeps the first scan's faces and embeddings."""
    import types

    from PIL import Image

    import studio.faces.engine as engine_mod

    class FakeEngine:
        providers = ["CPU"]

        def __init__(self, **kw):
            pass

        def __call__(self, rgb):
            v = np.ones(512, np.float32) / np.sqrt(512)
            return [types.SimpleNamespace(bbox=[0, 0, 80, 80], det_score=0.9, kps=None,
                                          normed_embedding=v)]

    monkeypatch.setattr(engine_mod, "FaceEngine", FakeEngine)
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    noise = (np.random.default_rng(0).random((100, 100, 3)) * 255).astype(np.uint8)
    Image.fromarray(noise).save(a / "1.jpg")
    Image.fromarray(noise).save(b / "2.jpg")
    ws_dir = tmp_path / "ws"
    main(["scan", str(a), "-w", str(ws_dir)])
    main(["scan", str(b), "-w", str(ws_dir)])
    main(["scan", str(a), "-w", str(ws_dir)])  # re-scan is a no-op
    ws = Workspace(ws_dir)
    assert len(ws.faces) == 2 and ws.embeddings().shape == (2, 512)
