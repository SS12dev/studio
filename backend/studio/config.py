"""Paths and settings. Everything lives under STUDIO_HOME (default ~/StudioData)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _home() -> Path:
    return Path(os.environ.get("STUDIO_HOME", Path.home() / "StudioData")).expanduser()


@dataclass(frozen=True)
class Settings:
    home: Path = field(default_factory=_home)
    # Working budget for resident models, in GB. 24GB Mac minus OS, browser and app.
    model_budget_gb: float = float(os.environ.get("STUDIO_MODEL_BUDGET_GB", "16"))

    @property
    def db_path(self) -> Path:
        return self.home / "studio.sqlite"

    @property
    def assets_dir(self) -> Path:
        return self.home / "assets"

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    def ensure(self) -> Settings:
        for p in (self.home, self.assets_dir, self.cache_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self


settings = Settings()
