import json

import numpy as np
from PIL import Image
from studio.bakeoff import cli
from studio.bakeoff.runners import Runner


def test_harness_with_fake_runners(tmp_path, monkeypatch):
    photos = tmp_path / "p"
    photos.mkdir()
    for i in range(3):
        Image.fromarray(np.full((64, 96, 3), i * 40, np.uint8)).save(photos / f"{i}.png")

    def good():
        return lambda rgb: rgb[:, :, 0]

    def broken():
        raise RuntimeError("no weights")

    monkeypatch.setattr(cli, "RUNNERS", {
        "fake": Runner("fake", "depth", "MIT", good),
        "broken": Runner("broken", "depth", "MIT", broken),
    })
    out = tmp_path / "out"
    cli.main(["run", str(photos), "-o", str(out), "--stage", "depth"])
    res = {r["runner"]: r for r in json.loads((out / "results.json").read_text())}
    assert res["fake"]["ok"] and res["fake"]["images"] == 3
    assert not res["broken"]["ok"] and "no weights" in res["broken"]["error"]
    assert (out / "compare.html").exists() and (out / "fake" / "002.jpg").exists()
