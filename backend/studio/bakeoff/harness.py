"""Run runners over photos one model at a time, record load time, latency and peak memory."""

from __future__ import annotations

import gc
import html
import json
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

from studio.bakeoff.runners import Runner


def _gpu_bytes() -> int:
    try:
        import torch

        if torch.backends.mps.is_available():
            return int(torch.mps.driver_allocated_memory())
    except Exception:  # noqa: BLE001
        pass
    return 0


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def run_one(runner: Runner, photos: list[tuple[str, np.ndarray]], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process()
    t0 = time.perf_counter()
    try:
        fn = runner.load()
    except Exception as e:  # noqa: BLE001 - a model that will not load is a result
        return {"runner": runner.name, "stage": runner.stage, "ok": False,
                "error": f"load: {type(e).__name__}: {e}"}
    load_s = time.perf_counter() - t0
    peak_rss, peak_gpu, times, errors = proc.memory_info().rss, _gpu_bytes(), [], []
    for i, (name, rgb) in enumerate(photos):
        t = time.perf_counter()
        try:
            out = fn(rgb)
            Image.fromarray(out).save(out_dir / f"{i:03d}.jpg", quality=90)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue
        finally:
            peak_rss = max(peak_rss, proc.memory_info().rss)
            peak_gpu = max(peak_gpu, _gpu_bytes())
        times.append(time.perf_counter() - t)
    del fn
    _free_gpu()
    # first image includes warm-up (graph compile, kernel cache); report it separately
    return {
        "runner": runner.name, "stage": runner.stage, "licence": runner.licence,
        "ok": not errors, "load_s": round(load_s, 1),
        "first_image_s": round(times[0], 2) if times else None,
        "median_s": round(float(np.median(times[1:] or times)), 2) if times else None,
        "peak_rss_gb": round(peak_rss / 1e9, 2), "peak_gpu_gb": round(peak_gpu / 1e9, 2),
        "images": len(times), "errors": errors[:5],
    }


def write_compare(out_root: Path, photos: list[tuple[str, np.ndarray]], results: list[dict]) -> Path:
    inputs = out_root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for i, (_, rgb) in enumerate(photos):
        Image.fromarray(rgb).save(inputs / f"{i:03d}.jpg", quality=90)
    ok = [r for r in results if r.get("images")]
    head = "".join(f"<th>{html.escape(r['runner'])}<br><small>{r['median_s']}s, "
                   f"{r['peak_gpu_gb']}GB GPU</small></th>" for r in ok)
    rows = []
    for i, (name, _) in enumerate(photos):
        cells = "".join(f'<td><img src="{html.escape(r["runner"])}/{i:03d}.jpg"></td>' for r in ok)
        rows.append(f'<tr><td><img src="inputs/{i:03d}.jpg"><br><small>{html.escape(Path(name).name)}'
                    f"</small></td>{cells}</tr>")
    page = ("<!doctype html><meta charset=utf-8><title>Bake-off</title><style>"
            "body{font:13px system-ui;background:#111;color:#eee;margin:16px}"
            "img{width:220px;border-radius:4px}td,th{vertical-align:top;padding:4px}</style>"
            f"<table><tr><th>input</th>{head}</tr>{''.join(rows)}</table>"
            f"<pre>{html.escape(json.dumps(results, indent=2))}</pre>")
    out = out_root / "compare.html"
    out.write_text(page)
    return out
