"""Group unknown faces into candidate people.

Average-linkage agglomerative clustering on cosine distance: two groups join only if the
*average* similarity between all their faces clears the threshold. This keeps one person's
different looks (glasses, beard, years apart) together, where density-based HDBSCAN split
them into many small groups on real photo libraries.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

# ArcFace: the same person usually scores 0.4 to 0.8, different people below about 0.2.
# 0.25 keeps very different looks of one person together; average linkage stops strangers
# chaining together because a whole group has to clear it on average, not one pair.
DEFAULT_LINK_SIM = 0.25


def cluster_embeddings(
    emb: np.ndarray, min_cluster_size: int = 3, link_sim: float = DEFAULT_LINK_SIM
) -> np.ndarray:
    """Label per row; groups smaller than min_cluster_size are returned as -1 (unclustered).

    Labels are renumbered so 0 is the largest group, 1 the next, and so on.
    """
    n = len(emb)
    if n < 2:
        return np.full(n, -1, dtype=int)
    raw = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=1 - link_sim
    ).fit_predict(_safe_unit(emb))
    ids, counts = np.unique(raw, return_counts=True)
    big = [i for i, c in sorted(zip(ids, counts), key=lambda t: -t[1]) if c >= min_cluster_size]
    remap = {old: new for new, old in enumerate(big)}
    return np.array([remap.get(r, -1) for r in raw], dtype=int)


def _safe_unit(emb: np.ndarray) -> np.ndarray:
    """float64 unit rows with zero rows left at zero.

    numpy 2.x on macOS (Accelerate BLAS) can raise spurious divide-by-zero / overflow warnings
    from float32 matmul even on clean data; float64 plus errstate keeps the output quiet and
    nan_to_num guards against any genuinely bad row.
    """
    e = np.asarray(emb, dtype=np.float64)
    n = np.linalg.norm(e, axis=1, keepdims=True)
    return np.divide(e, n, out=np.zeros_like(e), where=n > 0)


def centroids(emb: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    out = {}
    for lab in sorted(set(labels.tolist()) - {-1}):
        c = emb[labels == lab].mean(axis=0)
        out[lab] = c / (np.linalg.norm(c) + 1e-12)
    return out


def merge_suggestions(emb: np.ndarray, labels: np.ndarray, min_sim: float = 0.55, top: int = 15):
    """Pairs of groups whose faces are, on average, nearly as similar as a single group's.

    Uses the mean pairwise similarity between the two groups (not centroid similarity, which
    is inflated by averaging and made almost every pair look alike).
    """
    emb = _safe_unit(emb)
    groups = {lab: emb[labels == lab] for lab in sorted(set(labels.tolist()) - {-1})}
    keys = list(groups)
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            with np.errstate(all="ignore"):  # see _safe_unit
                sim = float(np.nan_to_num(groups[a] @ groups[b].T).mean())
            if sim >= min_sim:
                pairs.append((a, b, round(sim, 3)))
    return sorted(pairs, key=lambda p: -p[2])[:top]
