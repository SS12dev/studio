"""`studio` command: run the app, and bring names over from a studio-faces bake-off run.

    studio serve [--port 8000] [--no-browser]
    studio import-names runs/faces2      after importing the same photos in the app
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from pathlib import Path


def serve(args) -> None:
    import uvicorn

    from studio.api.main import FRONTEND_DIST, create_app

    if not FRONTEND_DIST.exists():
        print("frontend not built yet: run `make build` (or `make app`) first")
    url = f"http://127.0.0.1:{args.port}"
    print(f"Studio at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")


def import_names(args) -> None:
    """Copy names (and "not them" answers) from a studio-faces workspace into the library.

    Faces are matched by photo path and face index, which line up because the app runs the
    same detector. Photos must already be imported in the app.
    """
    from studio.config import settings
    from studio.library.core import Library

    copy_names(Library(settings.home), Path(args.workdir))


def copy_names(lib, ws: Path) -> int:
    from sqlmodel import select

    from studio.db import Face, Photo
    from studio.people import service

    people = json.loads((ws / "people.json").read_text())
    with lib.session() as s:
        by_key = {(str(Path(p.path)), f.idx): f.id
                  for f, p in s.exec(select(Face, Photo).join(Photo)).all()}
    total = 0
    for name, ids in people.items():
        face_ids = []
        for fid in ids:
            path, _, idx = fid.rpartition("#")
            key = (str(Path(path).resolve()), int(idx))
            if key in by_key:
                face_ids.append(by_key[key])
        if face_ids:
            service.move(lib, face_ids, name)
            total += len(face_ids)
        print(f"{name}: {len(face_ids)} of {len(ids)} faces matched")
    filed = service.auto_file(lib)
    print(f"{total} faces named, {filed} more filed automatically")
    return total


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="studio", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=serve)
    n = sub.add_parser("import-names")
    n.add_argument("workdir")
    n.set_defaults(fn=import_names)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
