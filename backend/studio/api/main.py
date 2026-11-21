"""FastAPI entry point. Phase 0 exposes health and the model registry; nodes arrive in phase 1."""

from __future__ import annotations

from fastapi import FastAPI

from studio import __version__
from studio.config import settings
from studio.models.registry import ModelManager

app = FastAPI(title="Studio", version=__version__)
manager = ModelManager(budget_gb=settings.model_budget_gb)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "version": __version__, "home": str(settings.home)}


@app.get("/models")
def models() -> dict:
    return {
        "budget_gb": manager.budget_gb,
        "resident_gb": manager.resident_gb,
        "loaded": manager.loaded(),
        "registered": [
            {"name": s.name, "stage": s.stage, "backend": s.backend, "est_gb": s.est_gb,
             "licence": s.licence}
            for s in manager.specs()
        ],
    }
