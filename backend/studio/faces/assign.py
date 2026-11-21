"""Assign new faces to named people and route each one by confidence.

auto    -> filed without asking
review  -> shown to you with the best guess pre-filled
new     -> probably someone not in the library yet; re-clustered as a batch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Route = Literal["auto", "review", "new"]


@dataclass(frozen=True)
class RoutingThresholds:
    # Typical ArcFace starting points. Tune on your own photos in the bake-off.
    auto_score: float = 0.55
    auto_margin: float = 0.10
    new_below: float = 0.35
    top_k: int = 5


@dataclass(frozen=True)
class Decision:
    route: Route
    person: str | None  # best guess
    score: float  # mean cosine similarity to the best person's top-k prototypes
    runner_up: str | None
    margin: float  # score minus the runner-up's score
    confidence: float | None = None  # calibrated probability, once a calibrator exists


class PeopleIndex:
    """Prototype embeddings per named person."""

    def __init__(self):
        self._protos: dict[str, np.ndarray] = {}

    def add(self, name: str, emb: np.ndarray) -> None:
        emb = np.atleast_2d(emb).astype(np.float32)
        cur = self._protos.get(name)
        self._protos[name] = emb if cur is None else np.vstack([cur, emb])

    def names(self) -> list[str]:
        return list(self._protos)

    def __len__(self) -> int:
        return len(self._protos)

    def scores(self, e: np.ndarray, top_k: int = 5) -> dict[str, float]:
        out = {}
        for name, P in self._protos.items():
            sims = P @ e
            k = min(top_k, len(sims))
            out[name] = float(np.sort(sims)[-k:].mean())
        return out


class Calibrator:
    """Turns (score, margin) into P(correct) using your past accept/reject answers."""

    def __init__(self):
        self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def fit(self, scores, margins, correct) -> Calibrator:
        from sklearn.linear_model import LogisticRegression

        X = np.column_stack([scores, margins])
        y = np.asarray(correct, dtype=int)
        if len(set(y.tolist())) < 2:
            raise ValueError("need both correct and incorrect examples to calibrate")
        # Scores live in a narrow range, so use weak regularisation to let the slope be steep.
        self._model = LogisticRegression(C=100.0).fit(X, y)
        return self

    def predict(self, score: float, margin: float) -> float:
        return float(self._model.predict_proba([[score, margin]])[0, 1])

    @staticmethod
    def threshold_for_precision(probs, correct, target: float = 0.99) -> float | None:
        """Lowest probability cut-off whose auto-filed set reaches the target precision."""
        order = np.argsort(-np.asarray(probs))
        p = np.asarray(probs)[order]
        c = np.asarray(correct, dtype=float)[order]
        prec = np.cumsum(c) / np.arange(1, len(c) + 1)
        ok = np.where(prec >= target)[0]
        return float(p[ok.max()]) if len(ok) else None


def decide(
    e: np.ndarray,
    index: PeopleIndex,
    th: RoutingThresholds = RoutingThresholds(),
    calibrator: Calibrator | None = None,
    auto_prob: float = 0.99,
) -> Decision:
    if len(index) == 0:
        return Decision("new", None, 0.0, None, 0.0)
    ranked = sorted(index.scores(e, th.top_k).items(), key=lambda kv: -kv[1])
    best, score = ranked[0]
    runner_up, second = ranked[1] if len(ranked) > 1 else (None, -1.0)
    margin = score - second
    conf = calibrator.predict(score, margin) if calibrator and calibrator.ready else None

    if score < th.new_below:
        route: Route = "new"
    elif conf is not None:
        route = "auto" if conf >= auto_prob else "review"
    elif score >= th.auto_score and margin >= th.auto_margin:
        route = "auto"
    else:
        route = "review"
    return Decision(route, best, round(score, 4), runner_up, round(margin, 4),
                    None if conf is None else round(conf, 4))
