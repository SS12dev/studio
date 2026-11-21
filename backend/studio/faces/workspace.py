"""On-disk state for a face-sorting run: face metadata, embeddings, crops, named people.

Layout of a workspace folder:
    faces.json        one record per detected face
    embeddings.npy    N x 512 float32, row i belongs to faces.json[i]
    people.json       {"Name": ["photo#idx", ...]}  faces you confirmed
    answers.json      [{"face": id, "guess": name, "score": s, "margin": m, "correct": bool}]
    crops/            128px face thumbnails for the report
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.crops = self.root / "crops"
        self.root.mkdir(parents=True, exist_ok=True)
        self.crops.mkdir(exist_ok=True)

    # files --------------------------------------------------------------
    def _read(self, name, default):
        p = self.root / name
        return json.loads(p.read_text()) if p.exists() else default

    def _write(self, name, data):
        (self.root / name).write_text(json.dumps(data, indent=2))

    @property
    def faces(self) -> list[dict]:
        return self._read("faces.json", [])

    def save_faces(self, faces: list[dict], emb: np.ndarray) -> None:
        self._write("faces.json", faces)
        np.save(self.root / "embeddings.npy", emb.astype(np.float32))

    def embeddings(self) -> np.ndarray:
        p = self.root / "embeddings.npy"
        return np.load(p) if p.exists() else np.zeros((0, 512), np.float32)

    @property
    def people(self) -> dict[str, list[str]]:
        return self._read("people.json", {})

    def save_people(self, people: dict[str, list[str]]) -> None:
        self._write("people.json", people)

    @property
    def answers(self) -> list[dict]:
        return self._read("answers.json", [])

    def save_answers(self, answers: list[dict]) -> None:
        self._write("answers.json", answers)

    @property
    def meta(self) -> dict:
        return self._read("meta.json", {})

    def save_meta(self, meta: dict) -> None:
        self._write("meta.json", meta)

    def clusters(self) -> dict:
        return self._read("clusters.json", {})

    def save_clusters(self, data: dict) -> None:
        self._write("clusters.json", data)

    @staticmethod
    def crop_name(face_id: str) -> str:
        return hashlib.sha1(face_id.encode()).hexdigest()[:16] + ".jpg"
