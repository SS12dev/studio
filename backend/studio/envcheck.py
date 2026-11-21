"""Print what this machine can run: chip, memory, disk, and which ML backends work.

Run on the Mac with `uv run studio-check`. Paste the output back into the project thread.
"""

from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
import sys

import psutil

from studio.config import settings


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return None


def _try(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - report any failure as data
        return f"unavailable ({type(e).__name__}: {e})"


def _torch():
    torch = importlib.import_module("torch")
    return {"version": torch.__version__, "mps": bool(torch.backends.mps.is_available())}


def _mlx():
    mx = importlib.import_module("mlx.core")
    return {"version": getattr(mx, "__version__", "?"), "default_device": str(mx.default_device())}


def _onnx():
    ort = importlib.import_module("onnxruntime")
    return {"version": ort.__version__, "providers": ort.get_available_providers()}


def collect() -> dict:
    home = settings.home
    probe = home if home.exists() else home.parent
    disk = shutil.disk_usage(probe)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "chip": _sysctl("machdep.cpu.brand_string") or platform.processor(),
        "memory_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "studio_home": str(home),
        "disk_free_gb": round(disk.free / 1e9, 1),
        "ffmpeg": shutil.which("ffmpeg") or "not found",
        "torch": _try(_torch),
        "mlx": _try(_mlx),
        "onnxruntime": _try(_onnx),
        "insightface": _try(lambda: importlib.import_module("insightface").__version__),
        "pillow_heif": _try(lambda: importlib.import_module("pillow_heif").__version__),
        "rawpy": _try(lambda: importlib.import_module("rawpy").__version__),
    }


def main() -> None:
    print(json.dumps(collect(), indent=2))


if __name__ == "__main__":
    main()
