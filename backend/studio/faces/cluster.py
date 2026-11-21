"""Group unknown faces into candidate people."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN


def cluster_embeddings(emb: np.ndarray, min_cluster_size: int = 3) -> np.ndarray:
    """HDBSCAN over L2-normalised embeddings. Returns a label per row, -1 = unclustered.

    For unit vectors Euclidean distance is sqrt(2 - 2cos), a monotonic function of cosine
    distance, so Euclidean HDBSCAN gives the same grouping as cosine and is faster.
    """
    if len(emb) < max(2, min_cluster_size):
        return np.full(len(emb), -1, dtype=int)
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2, metric="euclidean", copy=True)
    return model.fit_predict(emb.astype(np.float64))


def centroids(emb: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    out = {}
    for lab in sorted(set(labels.tolist()) - {-1}):
        c = emb[labels == lab].mean(axis=0)
        out[lab] = c / (np.linalg.norm(c) + 1e-12)
    return out


def merge_suggestions(emb: np.ndarray, labels: np.ndarray, min_sim: float = 0.6):
    """Pairs of clusters whose centroids are close enough that they may be the same person."""
    cents = centroids(emb, labels)
    keys = list(cents)
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            sim = float(cents[a] @ cents[b])
            if sim >= min_sim:
                pairs.append((a, b, round(sim, 3)))
    return sorted(pairs, key=lambda p: -p[2])
