"""Model registry and a memory-budgeted model manager.

Every model the app can use is described by a ModelSpec. The ModelManager loads models on demand
and evicts least-recently-used ones so the estimated resident total stays under the budget
(16GB by default on a 24GB Mac). Pinned models (in use by a running job) are never evicted.
"""

from __future__ import annotations

import gc
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

Backend = Literal["mlx", "mps", "coreml", "onnx", "cpu"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    stage: str  # e.g. "face_detect", "face_embed", "matting", "generator"
    backend: Backend
    est_gb: float  # estimated resident memory, measured in the bake-off
    licence: str
    loader: Callable[[], Any] = field(compare=False, repr=False)
    notes: str = ""


class BudgetExceeded(RuntimeError):
    pass


class ModelManager:
    def __init__(self, budget_gb: float):
        self.budget_gb = budget_gb
        self._specs: dict[str, ModelSpec] = {}
        self._loaded: OrderedDict[str, Any] = OrderedDict()  # LRU order: oldest first
        self._pins: dict[str, int] = {}
        self._lock = threading.RLock()

    # registry -----------------------------------------------------------
    def register(self, spec: ModelSpec) -> None:
        self._specs[spec.name] = spec

    def specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    # residency ----------------------------------------------------------
    @property
    def resident_gb(self) -> float:
        return sum(self._specs[n].est_gb for n in self._loaded)

    def loaded(self) -> list[str]:
        return list(self._loaded)

    def get(self, name: str) -> Any:
        """Return a loaded model, loading (and evicting others) if needed."""
        with self._lock:
            if name in self._loaded:
                self._loaded.move_to_end(name)
                return self._loaded[name]
            spec = self._specs[name]
            if spec.est_gb > self.budget_gb:
                raise BudgetExceeded(f"{name} needs ~{spec.est_gb}GB, budget is {self.budget_gb}GB")
            self._evict_until(self.budget_gb - spec.est_gb)
            model = spec.loader()
            self._loaded[name] = model
            return model

    def unload(self, name: str) -> None:
        with self._lock:
            if self._pins.get(name):
                raise RuntimeError(f"{name} is pinned by a running job")
            self._loaded.pop(name, None)
            gc.collect()
            _empty_device_caches()

    def _evict_until(self, target_gb: float) -> None:
        for victim in list(self._loaded):
            if self.resident_gb <= target_gb:
                return
            if not self._pins.get(victim):
                self.unload(victim)
        if self.resident_gb > target_gb:
            raise BudgetExceeded("pinned models leave no room; finish the running job first")

    # pinning ------------------------------------------------------------
    def pin(self, name: str):
        """Context manager: load and keep a model resident for the duration of a job."""
        manager = self

        class _Pin:
            def __enter__(self_inner):
                with manager._lock:
                    model = manager.get(name)
                    manager._pins[name] = manager._pins.get(name, 0) + 1
                    return model

            def __exit__(self_inner, *exc):
                with manager._lock:
                    manager._pins[name] -= 1

        return _Pin()


def _empty_device_caches() -> None:
    """Release cached GPU memory for whichever backends are installed."""
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
