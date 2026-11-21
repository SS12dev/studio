"""Manual curation of the people library. Used by the web page and the CLI.

State lives in the workspace:
    people.json     {"Name": [face_id, ...]}           faces you confirmed (by naming or review)
    curation.json   {"not_person": {face_id: [names]},  faces you said are NOT that person
                     "no_training": [face_id, ...]}     it's them, but keep out of training sets
                                                         (filters, heavy edits, bad crops)
    decisions.json  last `assign` run; route "auto" faces count as filed under their person
"""

from __future__ import annotations

import json
import threading

from studio.faces.workspace import Workspace

_lock = threading.RLock()


def _curation(ws: Workspace) -> dict:
    c = ws._read("curation.json", {})
    c.setdefault("not_person", {})
    c.setdefault("no_training", [])
    return c


def _save_curation(ws: Workspace, c: dict) -> None:
    c["no_training"] = sorted(set(c["no_training"]))
    ws._write("curation.json", c)


def _decisions(ws: Workspace) -> dict:
    p = ws.root / "decisions.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_decisions(ws: Workspace, d: dict) -> None:
    (ws.root / "decisions.json").write_text(json.dumps(d, indent=2))


def not_person(ws: Workspace) -> dict[str, list[str]]:
    return _curation(ws)["not_person"]


def no_training(ws: Workspace) -> set[str]:
    return set(_curation(ws)["no_training"])


def members(ws: Workspace) -> dict[str, list[tuple[str, str]]]:
    """Every person's faces as (face_id, source) with source "confirmed" or "auto"."""
    out = {name: [(f, "confirmed") for f in ids] for name, ids in ws.people.items()}
    confirmed = {f for ids in ws.people.values() for f in ids}
    for fid, d in _decisions(ws).items():
        if d.get("route") == "auto" and d.get("person") in out and fid not in confirmed:
            out[d["person"]].append((fid, "auto"))
    return out


def _detach(ws: Workspace, people: dict, decisions: dict, cur: dict, name: str, fids) -> None:
    keep = set(people.get(name, [])) - set(fids)
    if name in people:
        people[name] = sorted(keep)
    for f in fids:
        d = decisions.get(f)
        if d and d.get("person") == name and d.get("route") == "auto":
            d["route"] = "removed"
        np_ = cur["not_person"].setdefault(f, [])
        if name not in np_:
            np_.append(name)


def remove(ws: Workspace, name: str, fids: list[str]) -> None:
    """These faces are not `name`. They will not be filed under `name` again."""
    with _lock:
        people, decisions, cur = ws.people, _decisions(ws), _curation(ws)
        _detach(ws, people, decisions, cur, name, fids)
        ws.save_people(people)
        _save_decisions(ws, decisions)
        _save_curation(ws, cur)


def assign_to(ws: Workspace, name: str, fids: list[str], src: str | None = None) -> None:
    """Confirm faces as `name`, optionally taking them away from `src` first (a move)."""
    with _lock:
        people, decisions, cur = ws.people, _decisions(ws), _curation(ws)
        if src and src != name:
            _detach(ws, people, decisions, cur, src, fids)
        people[name] = sorted(set(people.get(name, [])) | set(fids))
        for f in fids:
            if name in cur["not_person"].get(f, []):
                cur["not_person"][f].remove(name)
            if f in decisions:
                decisions[f]["route"] = "confirmed"
        ws.save_people(people)
        _save_decisions(ws, decisions)
        _save_curation(ws, cur)


def set_training(ws: Workspace, fids: list[str], exclude: bool) -> None:
    with _lock:
        cur = _curation(ws)
        s = set(cur["no_training"])
        s = s | set(fids) if exclude else s - set(fids)
        cur["no_training"] = list(s)
        _save_curation(ws, cur)


def rename(ws: Workspace, old: str, new: str) -> None:
    """Rename a person; if `new` exists the two are merged."""
    with _lock:
        people, decisions = ws.people, _decisions(ws)
        if old not in people or not new or old == new:
            return
        people[new] = sorted(set(people.get(new, [])) | set(people.pop(old)))
        for d in decisions.values():
            if d.get("person") == old:
                d["person"] = new
        cur = _curation(ws)
        for names in cur["not_person"].values():
            if old in names:
                names[names.index(old)] = new
        ws.save_people(people)
        _save_decisions(ws, decisions)
        _save_curation(ws, cur)


def answer(ws: Workspace, fid: str, name: str | None) -> None:
    """Answer a review item: `name` = who it is, None = not the guessed person / nobody known.

    Records the answer for confidence calibration (correct = you agreed with the guess).
    """
    with _lock:
        d = _decisions(ws).get(fid)
        if d is None:
            return
        answers = ws.answers
        answers.append({"face": fid, "guess": d["person"], "score": d["score"],
                        "margin": d["margin"], "correct": name == d["person"]})
        ws.save_answers(answers)
    if name:
        assign_to(ws, name, [fid])
    elif d.get("person"):
        remove(ws, d["person"], [fid])
