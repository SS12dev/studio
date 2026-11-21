import numpy as np
import pytest
from studio.faces.cluster import cluster_embeddings, merge_suggestions
from studio.faces.quality import assess, estimate_yaw_from_landmarks, laplacian_variance
from studio.models.registry import BudgetExceeded, ModelManager, ModelSpec


def test_clusters_three_people():
    rng = np.random.default_rng(0)
    groups = []
    for _ in range(3):
        c = rng.standard_normal(512)
        g = c + 0.05 * rng.standard_normal((8, 512)) * np.linalg.norm(c)
        groups.append(g / np.linalg.norm(g, axis=1, keepdims=True))
    emb = np.vstack(groups)
    labels = cluster_embeddings(emb, 3)
    assert len(set(labels.tolist()) - {-1}) == 3
    for k in range(3):
        assert len(set(labels[k * 8:(k + 1) * 8].tolist())) == 1
    assert merge_suggestions(emb, labels) == []


def test_quality_rejects_small_and_blurry():
    sharp = (np.indices((96, 96)).sum(0) % 2 * 255).astype(np.uint8)
    rgb = np.repeat(sharp[:, :, None], 3, 2)
    assert assess(rgb, (0, 0, 96, 96), 0.9, None)[0]
    assert assess(rgb, (0, 0, 40, 40), 0.9, None)[1] == "too_small"
    flat = np.full((96, 96, 3), 128, np.uint8)
    assert assess(flat, (0, 0, 96, 96), 0.9, None)[1] == "blurry"
    assert laplacian_variance(sharp) > laplacian_variance(flat[:, :, 0])


def test_yaw_estimate():
    frontal = np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], float)
    profile = np.array([[30, 40], [70, 40], [72, 60], [35, 80], [65, 80]], float)
    assert abs(estimate_yaw_from_landmarks(frontal)) < 1
    assert abs(estimate_yaw_from_landmarks(profile)) > 60


def test_model_manager_lru_and_pins():
    mm = ModelManager(budget_gb=10)
    for name, gb in [("a", 6), ("b", 4), ("c", 5)]:
        mm.register(ModelSpec(name, "x", "cpu", gb, "test", loader=lambda n=name: n))
    mm.get("a"); mm.get("b")
    assert mm.loaded() == ["a", "b"]
    mm.get("c")  # evicts a (oldest)
    assert mm.loaded() == ["b", "c"]
    with mm.pin("b"), pytest.raises(BudgetExceeded):
        mm.register(ModelSpec("d", "x", "cpu", 7, "t", loader=lambda: "d"))
        mm.get("d")  # needs c and b gone, but b is pinned
    mm.register(ModelSpec("huge", "x", "cpu", 11, "t", loader=lambda: 0))
    with pytest.raises(BudgetExceeded):
        mm.get("huge")
