"""Static HTML contact sheets so you can eyeball clusters and the review queue."""

from __future__ import annotations

import html
from pathlib import Path

from studio.faces.workspace import Workspace

CSS = """
body{font:14px -apple-system,system-ui,sans-serif;margin:24px;background:#111;color:#eee}
h2{margin-top:32px}.grid{display:flex;flex-wrap:wrap;gap:8px}
figure{margin:0;width:128px}figure img{width:128px;height:128px;object-fit:cover;border-radius:6px}
figcaption{font-size:11px;color:#aaa;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.auto{outline:2px solid #3a3}.review{outline:2px solid #da3}.new{outline:2px solid #888}
code{background:#222;padding:2px 4px;border-radius:4px}
"""


def _tile(ws: Workspace, face_id: str, caption: str, cls: str = "") -> str:
    src = f"crops/{ws.crop_name(face_id)}"
    title = html.escape(face_id)
    return (f'<figure class="{cls}" title="{title}"><img src="{src}" loading="lazy">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def write_report(ws: Workspace, sections: list[tuple[str, str, list[tuple[str, str, str]]]]) -> Path:
    """sections: [(heading, note, [(face_id, caption, css_class), ...]), ...]"""
    parts = [f"<!doctype html><meta charset=utf-8><title>Face sorting</title><style>{CSS}</style>"]
    for heading, note, tiles in sections:
        parts.append(f"<h2>{html.escape(heading)} <small>({len(tiles)})</small></h2>")
        if note:
            parts.append(f"<p>{note}</p>")
        parts.append('<div class="grid">' + "".join(_tile(ws, *t) for t in tiles) + "</div>")
    out = ws.root / "report.html"
    out.write_text("\n".join(parts))
    return out
