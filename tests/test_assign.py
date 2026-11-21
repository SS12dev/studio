import numpy as np
from studio.faces.assign import Calibrator, PeopleIndex, RoutingThresholds, decide


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def person(rng, centre, n, noise=0.15):
    return np.stack([unit(centre + noise * rng.standard_normal(centre.shape)) for _ in range(n)])


def test_routes_auto_review_new():
    rng = np.random.default_rng(0)
    a, b = unit(rng.standard_normal(512)), unit(rng.standard_normal(512))
    idx = PeopleIndex()
    idx.add("A", person(rng, a, 10, 0.03))
    idx.add("B", person(rng, b, 10, 0.03))

    assert decide(person(rng, a, 1, 0.03)[0], idx).route == "auto"
    assert decide(person(rng, a, 1, 0.03)[0], idx).person == "A"
    # halfway between A and B: plausible but ambiguous
    mid = decide(unit(a + b), idx, RoutingThresholds(new_below=0.3))
    assert mid.route == "review"
    # an unrelated face
    assert decide(unit(rng.standard_normal(512)), idx).route == "new"


def test_empty_index_is_new():
    assert decide(unit(np.ones(512)), PeopleIndex()).route == "new"


def test_calibrator_and_precision_threshold():
    rng = np.random.default_rng(1)
    scores = np.r_[rng.uniform(0.5, 0.9, 80), rng.uniform(0.2, 0.5, 40)]
    margins = np.r_[rng.uniform(0.1, 0.4, 80), rng.uniform(0.0, 0.1, 40)]
    correct = np.r_[np.ones(80), np.zeros(40)]
    cal = Calibrator().fit(scores, margins, correct)
    assert cal.predict(0.8, 0.3) > 0.9
    assert cal.predict(0.3, 0.02) < 0.1
    probs = [cal.predict(s, m) for s, m in zip(scores, margins)]
    t = Calibrator.threshold_for_precision(probs, correct, 0.99)
    assert t is not None and 0 < t < 1
