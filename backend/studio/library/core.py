"""The Library: database + asset folders + the face model, shared by the API and jobs."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from studio.db import make_engine, session


class Library:
    def __init__(self, home: Path, face_engine_factory: Callable[[], Any] | None = None):
        self.home = Path(home).expanduser()
        self.assets = self.home / "assets"
        self.thumbs = self.assets / "thumbs"
        self.crops = self.assets / "crops"
        self.uploads = self.assets / "uploads"
        for d in (self.home, self.thumbs, self.crops, self.uploads):
            d.mkdir(parents=True, exist_ok=True)
        self.engine = make_engine(f"sqlite:///{self.home / 'library.sqlite'}")
        self._face_factory = face_engine_factory or _default_face_engine
        self._face_engine = None
        self._face_lock = threading.Lock()
        self.write_lock = threading.RLock()  # serialises identity changes

    def session(self):
        return session(self.engine)

    def face_engine(self):
        """Load the face model on first use and keep it (≈1GB resident)."""
        with self._face_lock:
            if self._face_engine is None:
                self._face_engine = self._face_factory()
            return self._face_engine


def _default_face_engine():
    from studio.faces.engine import FaceEngine

    return FaceEngine()
