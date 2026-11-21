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
