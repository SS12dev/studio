"""Face-sorting bake-off CLI (phase 0 / phase 1a).

    studio-faces scan    PHOTOS_DIR  -w WORKDIR   detect + embed every face, save crops
    studio-faces cluster             -w WORKDIR   group faces not yet named into Person N
    studio-faces name    CLUSTER NAME -w WORKDIR  name a cluster (or merge it into a name)
    studio-faces assign              -w WORKDIR   route unnamed faces: auto / review / new
    studio-faces confirm FACE NAME   -w WORKDIR   answer a review item (NAME or "-" for nobody)
    studio-faces stats               -w WORKDIR   timings, counts, calibration status

Open WORKDIR/report.html after cluster or assign to see the faces.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil

from studio.faces.assign import Calibrator, PeopleIndex, RoutingThresholds, decide
from studio.faces.cluster import cluster_embeddings, merge_suggestions
from studio.faces.quality import QualityThresholds, assess
from studio.faces.report import write_report
from studio.faces.workspace import Workspace
from studio.ingest import iter_images, load_rgb


# scan -------------------------------------------------------------------
def cmd_scan(args) -> None:
    from PIL import Image

    from studio.faces.engine import FaceEngine

    ws = Workspace(args.workdir)
    proc = psutil.Process()
    t0 = time.perf_counter()
    engine = FaceEngine(det_size=args.det_size)
    load_s = time.perf_counter() - t0
    print(f"model loaded in {load_s:.1f}s, providers={engine.providers}")

    th = QualityThresholds(min_face_px=args.min_face)
    # Append to an existing workspace so new folders can be filed against named people.
    faces = ws.faces
    old = ws.embeddings()
    embs = list(old) if len(faces) else []
    seen = {f["photo"] for f in faces}
    peak_rss, per_image = proc.memory_info().rss, []
    photos = [p for p in iter_images(Path(args.photos).expanduser()) if str(p) not in seen]
    print(f"{len(photos)} new photos ({len(seen)} already scanned)")
    for n, path in enumerate(photos, 1):
        t = time.perf_counter()
        try:
            rgb = load_rgb(path)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {path.name}: {e}")
            continue
        for i, f in enumerate(engine(rgb)):
            x1, y1, x2, y2 = [float(v) for v in f.bbox]
            crop = rgb[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
            ok, why, q = assess(crop, (x1, y1, x2, y2), float(f.det_score), f.kps, th)
            fid = f"{path}#{i}"
            if crop.size:
                Image.fromarray(crop).resize((128, 128)).save(ws.crops / ws.crop_name(fid), quality=85)
            faces.append({"id": fid, "photo": str(path), "index": i, "bbox": [x1, y1, x2, y2],
                          "usable": ok, "reject_reason": why, "quality": q})
            embs.append(np.asarray(f.normed_embedding, np.float32))
        per_image.append(time.perf_counter() - t)
        peak_rss = max(peak_rss, proc.memory_info().rss)
        if n % 25 == 0 or n == len(photos):
            print(f"  {n}/{len(photos)} photos, {len(faces)} faces")

    emb = np.vstack(embs) if embs else np.zeros((0, 512), np.float32)
    ws.save_faces(faces, emb)
    stats = {
        "new_photos": len(photos), "total_photos": len(seen) + len(photos), "faces": len(faces),
        "usable_faces": sum(f["usable"] for f in faces),
        "model_load_s": round(load_s, 2),
        "median_s_per_photo": round(float(np.median(per_image)), 3) if per_image else None,
        "peak_rss_gb": round(peak_rss / 1e9, 2),
        "providers": engine.providers,
    }
    (ws.root / "scan_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


# helpers ------------------------------------------------------------------
def _named_ids(ws: Workspace) -> set[str]:
    return {fid for ids in ws.people.values() for fid in ids}


def _index(ws: Workspace, faces, emb) -> PeopleIndex:
    pos = {f["id"]: i for i, f in enumerate(faces)}
    idx = PeopleIndex()
    for name, ids in ws.people.items():
        rows = [pos[i] for i in ids if i in pos]
        if rows:
            idx.add(name, emb[rows])
    return idx


def _calibrator(ws: Workspace) -> Calibrator | None:
    a = ws.answers
    if len(a) < 50:
        return None
    try:
        return Calibrator().fit([x["score"] for x in a], [x["margin"] for x in a],
                                [x["correct"] for x in a])
    except ValueError:
        return None


# cluster ------------------------------------------------------------------
def cmd_cluster(args) -> None:
    ws = Workspace(args.workdir)
    faces, emb = ws.faces, ws.embeddings()
    named = _named_ids(ws)
    rows = [i for i, f in enumerate(faces) if f["usable"] and f["id"] not in named]
    labels = cluster_embeddings(emb[rows], args.min_cluster) if rows else np.array([], int)
    clusters: dict[str, list[str]] = {}
    for r, lab in zip(rows, labels.tolist()):
        key = f"Person {lab + 1}" if lab >= 0 else "Unclustered"
        clusters.setdefault(key, []).append(faces[r]["id"])
    merges = [(f"Person {a + 1}", f"Person {b + 1}", s)
              for a, b, s in merge_suggestions(emb[rows], labels)] if rows else []
    ws.save_clusters({"clusters": clusters, "merge_suggestions": merges})

    order = sorted((k for k in clusters if k != "Unclustered"), key=lambda k: -len(clusters[k]))
    sections = [(k, f'Name it: <code>studio-faces name "{k}" "Their Name" -w {ws.root}</code>',
                 [(fid, Path(fid).name, "") for fid in clusters[k]]) for k in order]
    if "Unclustered" in clusters:
        sections.append(("Unclustered", "Seen too few times to group yet.",
                         [(fid, Path(fid).name, "new") for fid in clusters["Unclustered"]]))
    rejected = [f for f in faces if not f["usable"]]
    sections.append(("Set aside (quality)", "Too small, blurry, or turned away.",
                     [(f["id"], f["reject_reason"], "new") for f in rejected]))
    out = write_report(ws, sections)
    print(f"{len(order)} clusters, {len(clusters.get('Unclustered', []))} unclustered, "
          f"{len(rejected)} set aside")
    for a, b, s in merges:
        print(f"  maybe the same person: {a} and {b} (similarity {s})")
    print(f"open {out}")


# name -----------------------------------------------------------------------
def cmd_name(args) -> None:
    ws = Workspace(args.workdir)
    clusters = ws.clusters().get("clusters", {})
    if args.cluster not in clusters:
        sys.exit(f"no cluster called {args.cluster!r}; run cluster first")
    people = ws.people
    people.setdefault(args.name, [])
    people[args.name] = sorted(set(people[args.name]) | set(clusters[args.cluster]))
    ws.save_people(people)
    print(f"{args.name}: {len(people[args.name])} faces")


# assign -------------------------------------------------------------------
def cmd_assign(args) -> None:
    ws = Workspace(args.workdir)
    faces, emb = ws.faces, ws.embeddings()
    idx = _index(ws, faces, emb)
    if not len(idx):
        sys.exit("name at least one cluster first")
    cal = _calibrator(ws)
    th = RoutingThresholds(args.auto_score, args.auto_margin, args.new_below)
    named = _named_ids(ws)
    buckets: dict[str, list] = {"auto": [], "review": [], "new": []}
    decisions = {}
    for i, f in enumerate(faces):
        if not f["usable"] or f["id"] in named:
            continue
        d = decide(emb[i], idx, th, cal)
        decisions[f["id"]] = d.__dict__
        cap = f"{d.person} {d.score:.2f}" + (f" p={d.confidence:.2f}" if d.confidence else "")
        buckets[d.route].append((f["id"], cap, d.route))
    (ws.root / "decisions.json").write_text(json.dumps(decisions, indent=2))
    note = f'Answer with <code>studio-faces confirm "FACE_ID" "Name" -w {ws.root}</code> (hover a face for its id; use "-" for nobody you know).'
    out = write_report(ws, [
        ("Needs your answer", note, sorted(buckets["review"], key=lambda t: t[1])),
        ("Filed automatically", "", sorted(buckets["auto"], key=lambda t: t[1])),
        ("Possibly someone new", "Run cluster again to group these.", buckets["new"]),
    ])
    mode = "calibrated" if cal else "fixed thresholds"
    print(f"auto={len(buckets['auto'])} review={len(buckets['review'])} "
          f"new={len(buckets['new'])} ({mode})\nopen {out}")


# confirm ------------------------------------------------------------------
def cmd_confirm(args) -> None:
    ws = Workspace(args.workdir)
    dec_path = ws.root / "decisions.json"
    decisions = json.loads(dec_path.read_text()) if dec_path.exists() else {}
    d = decisions.get(args.face)
    if d is None:
        sys.exit("face not in the last assign run")
    people = ws.people
    if args.name != "-":
        people.setdefault(args.name, [])
        if args.face not in people[args.name]:
            people[args.name].append(args.face)
        ws.save_people(people)
    answers = ws.answers
    answers.append({"face": args.face, "guess": d["person"], "score": d["score"],
                    "margin": d["margin"], "correct": args.name == d["person"]})
    ws.save_answers(answers)
    print(f"recorded ({len(answers)} answers; calibration starts at 50)")


# stats --------------------------------------------------------------------
def cmd_stats(args) -> None:
    ws = Workspace(args.workdir)
    p = ws.root / "scan_stats.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    a = ws.answers
    out["people"] = {k: len(v) for k, v in ws.people.items()}
    out["answers"] = len(a)
    if a:
        out["guess_accuracy_on_reviewed"] = round(sum(x["correct"] for x in a) / len(a), 3)
    cal = _calibrator(ws)
    if cal:
        probs = [cal.predict(x["score"], x["margin"]) for x in a]
        out["auto_threshold_for_99pct_precision"] = Calibrator.threshold_for_precision(
            probs, [x["correct"] for x in a], 0.99)
    print(json.dumps(out, indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="studio-faces", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        sp = sub.add_parser(name, **kw)
        sp.add_argument("-w", "--workdir", type=Path, default=Path("faces-run"))
        sp.set_defaults(fn=fn)
        return sp

    s = add("scan", cmd_scan)
    s.add_argument("photos")
    s.add_argument("--det-size", type=int, default=640)
    s.add_argument("--min-face", type=int, default=64)
    c = add("cluster", cmd_cluster)
    c.add_argument("--min-cluster", type=int, default=3)
    n = add("name", cmd_name)
    n.add_argument("cluster")
    n.add_argument("name")
    a = add("assign", cmd_assign)
    a.add_argument("--auto-score", type=float, default=0.55)
    a.add_argument("--auto-margin", type=float, default=0.10)
    a.add_argument("--new-below", type=float, default=0.35)
    f = add("confirm", cmd_confirm)
    f.add_argument("face")
    f.add_argument("name")
    add("stats", cmd_stats)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
