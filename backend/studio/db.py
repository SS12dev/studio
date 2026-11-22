"""Library database (SQLite via SQLModel): photos, faces, people, imports.

Binary files (thumbnails, face crops, uploaded originals) live under STUDIO_HOME/assets;
the database only stores their file names.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from sqlalchemy import Column, LargeBinary, event
from sqlmodel import Field, Session, SQLModel, create_engine


def now() -> datetime:
    return datetime.now(UTC)


class Photo(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    sha256: str = Field(index=True, unique=True)
    path: str  # where the original lives (in place, or the uploads folder)
    filename: str
    width: int = 0
    height: int = 0
    taken_at: datetime | None = None
    imported_at: datetime = Field(default_factory=now)
    import_id: int | None = Field(default=None, foreign_key="importjob.id", index=True)
    thumb: str | None = None
    face_count: int = 0
    error: str | None = None


class Person(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=now)


class Face(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    photo_id: int = Field(foreign_key="photo.id", index=True)
    idx: int = 0
    x1: float = 0
    y1: float = 0
    x2: float = 0
    y2: float = 0
    det_score: float = 0
    sharpness: float = 0
    yaw: float = 0
    usable: bool = True
    reject_reason: str | None = None
    embedding: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    model_pack: str = "buffalo_l"
    crop: str | None = None
    # identity
    person_id: int | None = Field(default=None, foreign_key="person.id", index=True)
    source: str | None = None  # "confirmed" | "auto" | "review" | None
    guess_person_id: int | None = Field(default=None, foreign_key="person.id")
    score: float | None = None
    margin: float | None = None
    second_pass: bool = False
    group_key: int | None = Field(default=None, index=True)  # unnamed group from clustering
    no_training: bool = False  # it's them, but filtered/edited: keep out of training sets
    not_face: bool = False  # detector false positive, marked by you; hidden everywhere
    manual: bool = False  # box drawn by you in the photo viewer

    def vector(self) -> np.ndarray:
        return np.frombuffer(self.embedding, dtype=np.float32)


class NotPerson(SQLModel, table=True):
    """You said this face is not this person."""

    face_id: int = Field(foreign_key="face.id", primary_key=True)
    person_id: int = Field(foreign_key="person.id", primary_key=True)


class Answer(SQLModel, table=True):
    """Review answers, used to calibrate confidence."""

    id: int | None = Field(default=None, primary_key=True)
    face_id: int = Field(foreign_key="face.id")
    guess_person_id: int | None = None
    score: float = 0
    margin: float = 0
    correct: bool = False
    at: datetime = Field(default_factory=now)


class ImportJob(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=now)
    finished_at: datetime | None = None
    source: str = ""
    status: str = "queued"  # queued | running | done | error | cancelled
    total: int = 0
    processed: int = 0
    added: int = 0
    duplicates: int = 0
    failed: int = 0
    faces: int = 0
    auto_filed: int = 0
    message: str | None = None


class ImportFailure(SQLModel, table=True):
    """One file an import could not read, with the reason, so you can see why and retry."""

    id: int | None = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="importjob.id", index=True)
    path: str
    error: str


def make_engine(url: str):
    engine = create_engine(url, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _):  # WAL lets the UI read while an import writes
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    SQLModel.metadata.create_all(engine)
    _add_missing_columns(engine)
    return engine


def _add_missing_columns(engine) -> None:
    """Tiny forward-only migration: add new boolean/nullable columns to existing tables."""
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            have = {c["name"] for c in insp.get_columns(table.name)}
            for column in table.columns:
                if column.name in have:
                    continue
                kind = column.type.compile(engine.dialect)
                default = " NOT NULL DEFAULT 0" if kind == "BOOLEAN" else ""
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {kind}{default}'))


def session(engine) -> Session:
    return Session(engine, expire_on_commit=False)
