"""Face-sorting bake-off CLI (phase 0 / phase 1a).

    studio-faces scan    PHOTOS_DIR  -w WORKDIR   detect + embed every face, save crops
    studio-faces cluster             -w WORKDIR   group faces not yet named into Person N
    studio-faces name    CLUSTER NAME -w WORKDIR  name a cluster (or merge it into a name)
    studio-faces assign              -w WORKDIR   route unnamed faces: auto / review / new
    studio-faces confirm FACE NAME   -w WORKDIR   answer a review item (NAME or "-" for nobody)
    studio-faces stats               -w WORKDIR   timings, counts, calibration status
    studio-faces ui                  -w WORKDIR   people page: fix groups, review, filters

Open WORKDIR/report.html after cluster or assign to see the faces.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
import sys
import time
from pathlib import Path

import numpy as np
import psutil

from studio.faces import curate
from studio.faces.assign import Calibrator, PeopleIndex, RoutingThresholds, decide
from studio.faces.cluster import DEFAULT_LINK_SIM, cluster_embeddings, merge_suggestions
from studio.faces.quality import QualityThresholds, assess
from studio.faces.report import write_report
from studio.faces.workspace import Workspace
from studio.ingest import iter_images, load_rgb


# scan -------------------------------------------------------------------
def cmd_scan(args) -> None:
    from PIL import Image

    from studio.faces.engine import FaceEngine

    ws = Workspace(args.workdir)
    pack = ws.meta.get("model_pack")
    if pack and pack != args.model:
        sys.exit(f"this workspace was scanned with {pack}; use a new -w folder for {args.model}")
    ws.save_meta({**ws.meta, "model_pack": args.model})
    proc = psutil.Process()
    t0 = time.perf_counter()
    engine = FaceEngine(model_pack=args.model, det_size=args.det_size)
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
        "model_pack": args.model,
        "set_aside_reasons": dict(Counter(f["reject_reason"] for f in faces if not f["usable"])),
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
    labels = (cluster_embeddings(emb[rows], args.min_cluster, args.link_sim)
              if rows else np.array([], int))
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
    reasons = dict(Counter(f["reject_reason"] for f in rejected))
    print(f"{len(order)} groups, {len(clusters.get('Unclustered', []))} unclustered, "
          f"{len(rejected)} set aside {reasons}")
    print("largest groups: " + ", ".join(f"{k} ({len(clusters[k])})" for k in order[:10]))
    if merges:
        print("maybe the same person (name both with the same name to merge):")
    for a, b, s in merges:
        print(f"  {a} + {b}  (avg similarity {s})")
    print(f"open {out}")


# name -----------------------------------------------------------------------
def cmd_name(args) -> None:
    """Name one or more groups; "Person 3,Person 7" gives both groups the same name (merge)."""
    ws = Workspace(args.workdir)
    clusters = ws.clusters().get("clusters", {})
    keys = [k.strip() for k in args.cluster.split(",") if k.strip()]
    missing = [k for k in keys if k not in clusters]
    if missing:
        sys.exit(f"no group called {missing}; run cluster first (names are like \"Person 3\")")
    people = ws.people
    ids = set(people.get(args.name, []))
    for k in keys:
        ids |= set(clusters[k])
    people[args.name] = sorted(ids)
    ws.save_people(people)
    print(f"{args.name}: {len(ids)} faces")


# assign -------------------------------------------------------------------
def cmd_assign(args) -> None:
    ws = Workspace(args.workdir)
    faces, emb = ws.faces, ws.embeddings()
    idx = _index(ws, faces, emb)
    if not len(idx):
        sys.exit("name at least one cluster first")
    cal = _calibrator(ws)
    th = RoutingThresholds(args.auto_score, args.auto_margin, args.new_below)
    # Second pass for faces the quality gate set aside (small, blurry, turned away): they are
    # only matched against people you already named, with stricter thresholds, and they never
    # start new groups or become prototypes for others.
    strict = RoutingThresholds(args.auto_score + args.strict_extra,
                               args.auto_margin + args.strict_extra / 2,
                               args.new_below + args.strict_extra)
    named = _named_ids(ws)
    refused = curate.not_person(ws)  # faces you said are not a given person
    buckets: dict[str, list] = {"auto": [], "review": [], "new": [], "still_set_aside": []}
    decisions = {}
    second_pass = 0
    for i, f in enumerate(faces):
        if f["id"] in named:
            continue
        if not f["usable"]:
            if not args.include_set_aside:
                continue
            second_pass += 1
            d = decide(emb[i], idx, strict, None)
            if d.route == "new":
                buckets["still_set_aside"].append((f["id"], f["reject_reason"], "new"))
                continue
        else:
            d = decide(emb[i], idx, th, cal)
        if d.person in refused.get(f["id"], []):
            # You already said this face is not them: never auto-file it there again.
            alt = d.runner_up if d.runner_up not in refused.get(f["id"], []) else None
            d = d.__class__("review", alt, d.score, None, d.margin, d.confidence)
            if alt is None:
                continue
        decisions[f["id"]] = {**d.__dict__, "second_pass": not f["usable"]}
        tag = " (2nd pass)" if not f["usable"] else ""
        cap = (f"{d.person} {d.score:.2f}" + (f" p={d.confidence:.2f}" if d.confidence else "")
               + tag)
        buckets[d.route].append((f["id"], cap, d.route))
    (ws.root / "decisions.json").write_text(json.dumps(decisions, indent=2))
    note = f'Answer with <code>studio-faces confirm "FACE_ID" "Name" -w {ws.root}</code> (hover a face for its id; use "-" for nobody you know).'
    out = write_report(ws, [
        ("Needs your answer", note, sorted(buckets["review"], key=lambda t: t[1])),
        ("Filed automatically", "", sorted(buckets["auto"], key=lambda t: t[1])),
        ("Possibly someone new", "Run cluster again to group these.", buckets["new"]),
        ("Still set aside", "Low-quality faces that matched nobody strongly enough.",
         buckets["still_set_aside"]),
    ])
    mode = "calibrated" if cal else "fixed thresholds"
    print(f"auto={len(buckets['auto'])} review={len(buckets['review'])} "
          f"new={len(buckets['new'])} ({mode})")
    if args.include_set_aside:
        rescued = second_pass - len(buckets["still_set_aside"])
        print(f"second pass over {second_pass} set-aside faces: {rescued} matched a named person "
              f"(stricter thresholds), {len(buckets['still_set_aside'])} left aside")
    print(f"open {out}")


# confirm ------------------------------------------------------------------
def cmd_confirm(args) -> None:
    ws = Workspace(args.workdir)
    if args.face not in curate._decisions(ws):
        sys.exit("face not in the last assign run")
    curate.answer(ws, args.face, None if args.name == "-" else args.name)
    print(f"recorded ({len(ws.answers)} answers; calibration starts at 50)")


def cmd_ui(args) -> None:
    import threading
    import webbrowser

    import uvicorn

    from studio.faces.ui import create_app

    url = f"http://127.0.0.1:{args.port}"
    print(f"people page for {args.workdir} at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(Workspace(args.workdir)), host="127.0.0.1", port=args.port,
                log_level="warning")


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
    s.add_argument("--min-face", type=int, default=40)
    s.add_argument("--model", default="buffalo_l", choices=["buffalo_l", "antelopev2"])
    c = add("cluster", cmd_cluster)
    c.add_argument("--min-cluster", type=int, default=3)
    c.add_argument("--link-sim", type=float, default=DEFAULT_LINK_SIM,
                   help="join groups whose average similarity is at least this (lower = fewer, "
                        "bigger groups; raise it if different people get merged)")
    n = add("name", cmd_name)
    n.add_argument("cluster", help='a group like "Person 3", or several: "Person 3,Person 7"')
    n.add_argument("name")
    a = add("assign", cmd_assign)
    a.add_argument("--auto-score", type=float, default=0.55)
    a.add_argument("--auto-margin", type=float, default=0.10)
    a.add_argument("--new-below", type=float, default=0.35)
    a.add_argument("--no-set-aside", dest="include_set_aside", action="store_false",
                   help="skip the second pass over faces the quality gate set aside")
    a.add_argument("--strict-extra", type=float, default=0.10,
                   help="how much stricter the second pass is than the normal thresholds")
    f = add("confirm", cmd_confirm)
    f.add_argument("face")
    f.add_argument("name")
    add("stats", cmd_stats)
    u = add("ui", cmd_ui)
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--no-browser", action="store_true")

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
