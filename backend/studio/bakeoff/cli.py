"""Extraction bake-off.

    studio-bakeoff list
    studio-bakeoff run PHOTOS -o runs/bake1 [--stage matting,depth,albedo] [--only a,b] [--limit 12]

Runs one model at a time (so memory numbers are honest), writes per-model outputs, results.json,
and compare.html with every model's output side by side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from studio.bakeoff.harness import run_one, write_compare
from studio.bakeoff.runners import RUNNERS
from studio.ingest import iter_images, load_rgb


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="studio-bakeoff", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("photos")
    r.add_argument("-o", "--out", type=Path, default=Path("runs/bake1"))
    r.add_argument("--stage", default="matting,depth,albedo")
    r.add_argument("--only", default="", help="comma-separated runner names")
    r.add_argument("--limit", type=int, default=12)
    r.add_argument("--max-side", type=int, default=1536)
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for x in RUNNERS.values():
            print(f"{x.name:28} {x.stage:8} {x.licence}")
        return

    stages = set(args.stage.split(","))
    only = set(filter(None, args.only.split(",")))
    chosen = [x for x in RUNNERS.values() if x.stage in stages and (not only or x.name in only)]
    paths = list(iter_images(Path(args.photos).expanduser()))[: args.limit]
    photos = [(str(p), load_rgb(p, args.max_side)) for p in paths]
    print(f"{len(photos)} photos, {len(chosen)} models")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for x in chosen:
        print(f"-> {x.name} ({x.stage})", flush=True)
        res = run_one(x, photos, args.out / x.name)
        print(json.dumps(res))
        results.append(res)
        (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"open {write_compare(args.out, photos, results)}")


if __name__ == "__main__":
    main()
