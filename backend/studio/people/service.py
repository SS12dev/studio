"""People: group unknown faces, name groups, auto-file new faces, and manual fixes.

Face.source tells where a face's identity came from:
    confirmed  you named its group, moved it, or accepted it in review
    auto       filed by the matcher with high confidence
    review     the matcher has a guess (guess_person_id) but wants your answer
    None       unknown: may sit in an unnamed group (group_key) or nowhere yet
"""

from __future__ import annotations

import numpy as np
from sqlmodel import col, select

from studio.db import Answer, Face, NotPerson, Person
from studio.faces.assign import Calibrator, PeopleIndex, RoutingThresholds, decide
from studio.faces.cluster import DEFAULT_LINK_SIM, cluster_embeddings
from studio.library.core import Library

STRICT_EXTRA = 0.10  # second pass for faces the quality gate set aside


def _vecs(faces: list[Face]) -> np.ndarray:
    return np.vstack([f.vector() for f in faces]) if faces else np.zeros((0, 512), np.float32)


def _has_embedding(f: Face) -> bool:
    return bool(np.any(f.vector()))


def _get_or_create(s, name: str) -> Person:
    name = name.strip()
    if not name:
        raise ValueError("name is empty")
    p = s.exec(select(Person).where(Person.name == name)).first()
    if p is None:
        p = Person(name=name)
        s.add(p)
        s.flush()
    return p


# grouping -------------------------------------------------------------------
def regroup(lib: Library, link_sim: float = DEFAULT_LINK_SIM, min_size: int = 3) -> int:
    """Cluster usable faces that have no person yet into unnamed groups. Returns group count."""
    with lib.write_lock, lib.session() as s:
        faces = s.exec(select(Face).where(Face.person_id.is_(None), Face.usable,
                                          Face.not_face.is_(False))).all()
        for f in s.exec(select(Face).where(Face.group_key.is_not(None))).all():
            f.group_key = None
            s.add(f)
        labels = cluster_embeddings(_vecs(faces), min_size, link_sim) if faces else []
        for f, lab in zip(faces, labels):
            f.group_key = int(lab) if lab >= 0 else None
            s.add(f)
        s.commit()
        return len({int(x) for x in labels if x >= 0})


def name_groups(lib: Library, keys: list[int], name: str) -> int:
    with lib.write_lock, lib.session() as s:
        person = _get_or_create(s, name)
        faces = s.exec(select(Face).where(col(Face.group_key).in_(keys))).all()
        for f in _one_per_photo(s, faces, person.id):
            _confirm(f, person.id)
            s.add(f)
        s.commit()
        return person.id


def _one_per_photo(s, faces: list[Face], person_id: int) -> list[Face]:
    """A person appears at most once per photo. When a group holds two faces from one photo
    (you and your brother, say), keep the one closest to the rest of the group and leave the
    other unnamed so you can sort it. Photos where the person is already confirmed keep that."""
    taken = set(s.exec(select(Face.photo_id).where(Face.person_id == person_id,
                                                   Face.source == "confirmed")).all())
    if not faces:
        return []
    V = _vecs(faces)
    mean = V.mean(0)
    mean /= np.linalg.norm(mean) or 1.0
    sims = V @ mean
    best: dict[int, int] = {}
    for i, f in enumerate(faces):
        if f.photo_id in taken:
            continue
        if f.photo_id not in best or sims[i] > sims[best[f.photo_id]]:
            best[f.photo_id] = i
    keep = {best[p] for p in best}
    for i, f in enumerate(faces):
        if i not in keep:  # leave it out of the group so it shows up in Unsorted
            f.group_key = None
            s.add(f)
    return [faces[i] for i in sorted(keep)]


def _confirm(f: Face, person_id: int) -> None:
    f.person_id, f.source, f.group_key = person_id, "confirmed", None
    f.guess_person_id = None


# matching -------------------------------------------------------------------
def _index(s) -> tuple[PeopleIndex, dict[int, str]]:
    names = {p.id: p.name for p in s.exec(select(Person)).all()}
    idx = PeopleIndex()
    by_person: dict[int, list[Face]] = {}
    for f in s.exec(select(Face).where(Face.source == "confirmed", Face.not_face.is_(False))).all():
        if _has_embedding(f):
            by_person.setdefault(f.person_id, []).append(f)
    for pid, fs in by_person.items():
        idx.add(str(pid), _vecs(fs))
    return idx, names


def _calibrator(s) -> Calibrator | None:
    a = s.exec(select(Answer)).all()
    if len(a) < 50:
        return None
    try:
        return Calibrator().fit([x.score for x in a], [x.margin for x in a], [x.correct for x in a])
    except ValueError:
        return None


def auto_file(lib: Library, import_id: int | None = None, th: RoutingThresholds | None = None) -> int:
    """Match faces not confirmed by you against named people. Returns how many were filed.

    Usable faces use normal thresholds; faces the quality gate set aside get a stricter second
    pass and can only be filed or queued for review, never start anything new.
    """
    th = th or RoutingThresholds()
    strict = RoutingThresholds(th.auto_score + STRICT_EXTRA, th.auto_margin + STRICT_EXTRA / 2,
                               th.new_below + STRICT_EXTRA)
    with lib.write_lock, lib.session() as s:
        idx, _ = _index(s)
        if not len(idx):
            return 0
        cal = _calibrator(s)
        refused: dict[int, set[int]] = {}
        for np_ in s.exec(select(NotPerson)).all():
            refused.setdefault(np_.face_id, set()).add(np_.person_id)
        q = select(Face).where((Face.source.is_(None)) | (col(Face.source).in_(["auto", "review"])),
                               Face.not_face.is_(False))
        if import_id is not None:
            from studio.db import Photo

            q = q.join(Photo).where(Photo.import_id == import_id)
        # who is already in each photo by your hand: nobody appears twice in one photo
        taken: dict[int, set[int]] = {}
        for pid, per in s.exec(select(Face.photo_id, Face.person_id).where(
                Face.source == "confirmed")).all():
            taken.setdefault(pid, set()).add(per)
        decided = []
        for f in s.exec(q).all():
            if not _has_embedding(f):
                continue
            d = decide(f.vector(), idx, th if f.usable else strict, cal if f.usable else None)
            decided.append((f, d))
        decided.sort(key=lambda fd: -fd[1].score)  # best match in a photo claims the person
        filed = 0
        for f, d in decided:
            no = refused.get(f.id, set()) | taken.get(f.photo_id, set())
            person = int(d.person) if d.person else None
            route = d.route
            if person in no:
                person = int(d.runner_up) if d.runner_up and int(d.runner_up) not in no else None
                route = "review" if person else "new"
            f.score, f.margin, f.second_pass = d.score, d.margin, not f.usable
            if route == "auto":
                taken.setdefault(f.photo_id, set()).add(person)
                f.person_id, f.source, f.guess_person_id, f.group_key = person, "auto", None, None
                filed += 1
            elif route == "review":
                f.person_id, f.source, f.guess_person_id = None, "review", person
            else:
                f.person_id, f.source, f.guess_person_id = None, None, None
            s.add(f)
        s.commit()
        return filed


# manual fixes -----------------------------------------------------------------
def remove(lib: Library, person_id: int, face_ids: list[int]) -> None:
    """These faces are not this person; never file them there again."""
    with lib.write_lock, lib.session() as s:
        for f in s.exec(select(Face).where(col(Face.id).in_(face_ids))).all():
            if f.person_id == person_id or f.guess_person_id == person_id:
                f.person_id, f.source, f.guess_person_id = None, None, None
                s.add(f)
            if not s.get(NotPerson, (f.id, person_id)):
                s.add(NotPerson(face_id=f.id, person_id=person_id))
        s.commit()


def move(lib: Library, face_ids: list[int], name: str) -> int:
    """These faces are `name` (created if new). Any other face in the same photo that was
    filed as `name` is taken out, since one person appears once per photo."""
    with lib.write_lock, lib.session() as s:
        person = _get_or_create(s, name)
        faces = s.exec(select(Face).where(col(Face.id).in_(face_ids))).all()
        photos = {f.photo_id for f in faces}
        for o in s.exec(select(Face).where(col(Face.photo_id).in_(photos),
                                           Face.person_id == person.id,
                                           col(Face.id).not_in(face_ids))).all():
            o.person_id, o.source, o.guess_person_id = None, None, None
            s.add(o)
            if not s.get(NotPerson, (o.id, person.id)):
                s.add(NotPerson(face_id=o.id, person_id=person.id))
        for f in faces:
            f.not_face = False
            if f.person_id and f.person_id != person.id and not s.get(NotPerson, (f.id, f.person_id)):
                s.add(NotPerson(face_id=f.id, person_id=f.person_id))
            stale = s.get(NotPerson, (f.id, person.id))
            if stale:
                s.delete(stale)
            _confirm(f, person.id)
            s.add(f)
        s.commit()
        return person.id


def set_training(lib: Library, face_ids: list[int], exclude: bool) -> None:
    with lib.write_lock, lib.session() as s:
        for f in s.exec(select(Face).where(col(Face.id).in_(face_ids))).all():
            f.no_training = exclude
            s.add(f)
        s.commit()


def rename(lib: Library, person_id: int, new_name: str) -> int:
    """Rename; if the new name already exists, merge into that person. Returns the person id."""
    with lib.write_lock, lib.session() as s:
        p = s.get(Person, person_id)
        new_name = new_name.strip()
        other = s.exec(select(Person).where(Person.name == new_name)).first()
        if other is None:
            p.name = new_name
            s.add(p)
            s.commit()
            return p.id
        if other.id == p.id:
            return p.id
        for f in s.exec(select(Face).where((Face.person_id == p.id) | (Face.guess_person_id == p.id))).all():
            if f.person_id == p.id:
                f.person_id = other.id
            if f.guess_person_id == p.id:
                f.guess_person_id = other.id
            s.add(f)
        for np_ in s.exec(select(NotPerson).where(NotPerson.person_id == p.id)).all():
            if not s.get(NotPerson, (np_.face_id, other.id)):
                s.add(NotPerson(face_id=np_.face_id, person_id=other.id))
            s.delete(np_)
        s.flush()
        s.delete(p)
        s.commit()
        return other.id


def answer(lib: Library, face_id: int, name: str | None) -> None:
    """Answer a review item. name=None means "not the guessed person"."""
    with lib.write_lock, lib.session() as s:
        f = s.get(Face, face_id)
        guess = f.guess_person_id
        guess_name = s.get(Person, guess).name if guess else None
        s.add(Answer(face_id=face_id, guess_person_id=guess, score=f.score or 0,
                     margin=f.margin or 0, correct=name is not None and name == guess_name))
        s.commit()
    if name:
        move(lib, [face_id], name)
    elif guess:
        remove(lib, guess, [face_id])


def set_not_face(lib: Library, face_ids: list[int], value: bool = True) -> None:
    """Detector false positives: hide them from people, groups and matching (or undo)."""
    with lib.write_lock, lib.session() as s:
        for f in s.exec(select(Face).where(col(Face.id).in_(face_ids))).all():
            f.not_face = value
            if value:
                f.person_id, f.source, f.guess_person_id, f.group_key = None, None, None, None
            s.add(f)
        s.commit()


def delete_person(lib: Library, person_id: int) -> None:
    """Forget a person: their faces go back to unsorted; nothing else is lost."""
    with lib.write_lock, lib.session() as s:
        for f in s.exec(select(Face).where((Face.person_id == person_id) |
                                           (Face.guess_person_id == person_id))).all():
            if f.person_id == person_id:
                f.person_id, f.source = None, None
            if f.guess_person_id == person_id:
                f.guess_person_id, f.source = None, None
            s.add(f)
        for np_ in s.exec(select(NotPerson).where(NotPerson.person_id == person_id)).all():
            s.delete(np_)
        s.flush()
        p = s.get(Person, person_id)
        if p:
            s.delete(p)
        s.commit()


def similar_in(lib: Library, face_id: int, person_id: int, min_sim: float = 0.3,
               limit: int = 60) -> list[tuple[Face, float]]:
    """Faces filed under person_id that look like face_id: after you move one wrong face out
    of a person, these are the likely other mistakes of the same kind."""
    with lib.session() as s:
        f = s.get(Face, face_id)
        if f is None or not _has_embedding(f):
            return []
        others = [o for o in s.exec(select(Face).where(Face.person_id == person_id,
                                                       Face.id != face_id)).all()
                  if _has_embedding(o)]
        if not others:
            return []
        V = _vecs(others)
        # closer to this face than to the person's own centre
        centre = V.mean(0)
        centre /= np.linalg.norm(centre) or 1.0
        to_face, to_centre = V @ f.vector(), V @ centre
        out = [(o, float(a)) for o, a, b in zip(others, to_face, to_centre)
               if a >= min_sim and a > b]
        out.sort(key=lambda t: -t[1])
        return out[:limit]


def conflicts(lib: Library) -> list[dict]:
    """Photos where one person was filed on two or more faces: at least one is wrong."""
    from sqlalchemy import func

    with lib.session() as s:
        rows = s.exec(select(Face.photo_id, Face.person_id).where(
            Face.person_id.is_not(None)).group_by(Face.photo_id, Face.person_id)
            .having(func.count() > 1)).all()
        out = []
        for photo_id, person_id in rows:
            faces = s.exec(select(Face).where(Face.photo_id == photo_id,
                                              Face.person_id == person_id)).all()
            out.append({"photo_id": photo_id, "person_id": person_id, "faces": faces})
        return out
