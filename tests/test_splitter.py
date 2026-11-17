"""Splitter: fast scan == brute force (DESIGN.md §9.1), vectors, edge cases, max_features."""

from __future__ import annotations

import numpy as np
import pytest

from treesfs.criteria import CRITERIA
from treesfs.splitter import (
    best_split_brute_force,
    best_split_one_feature,
    find_best_split,
    partition_cost,
    resolve_max_features,
)


def _random_case(rng: np.random.Generator, name: str):
    n = int(rng.integers(2, 60))
    msl = int(rng.integers(1, 6))
    if rng.random() < 0.5:
        x = rng.normal(size=n)
    else:  # heavily tied integer feature
        x = rng.integers(0, 5, size=n).astype(float)
    if name == "squared_error":
        y = rng.normal(size=n) + 1e4  # offset stresses cancellation
        stats = y - y.mean()
        k = None
    else:
        k = int(rng.integers(2, 5))
        y = rng.integers(0, k, size=n)
        stats = np.eye(k)[y]
    return x, y, stats, k, msl


@pytest.mark.parametrize("name", ["gini", "entropy", "squared_error"])
def test_fast_equals_brute_force(name):
    rng = np.random.default_rng({"gini": 11, "entropy": 12, "squared_error": 13}[name])
    crit = CRITERIA[name]
    n_none = 0
    for _ in range(300):
        x, y, stats, k, msl = _random_case(rng, name)
        fast = best_split_one_feature(x, stats, crit, msl)
        brute = best_split_brute_force(x, y, name, msl, n_classes=k)
        assert (fast is None) == (brute is None)
        if fast is None:
            n_none += 1
            continue
        cost, tau, n_left = fast
        assert cost == pytest.approx(brute[0], rel=1e-9, abs=1e-8)
        # Different thresholds can tie, so check the returned one really achieves the cost.
        assert partition_cost(x, y, tau, name, k) == pytest.approx(cost, rel=1e-9, abs=1e-8)
        left = x <= tau
        assert left.sum() == n_left
        assert n_left >= msl and len(x) - n_left >= msl
    assert n_none < 300  # the comparison was not vacuous


def test_design_vector_classification(tiny_clf):
    x, y = tiny_clf
    cost, tau, n_left = best_split_one_feature(x, np.eye(2)[y], CRITERIA["gini"], 1)
    assert tau == 3.5 and n_left == 3
    assert cost == pytest.approx(4 / 3)


def test_design_vector_regression():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 1.0, 5.0, 5.0])
    cost, tau, _ = best_split_one_feature(x, y - y.mean(), CRITERIA["squared_error"], 1)
    assert tau == 2.5
    assert cost == pytest.approx(0.0, abs=1e-12)


def test_constant_feature_and_too_few_rows():
    stats = np.eye(2)[[0, 1, 0, 1]]
    assert best_split_one_feature(np.ones(4), stats, CRITERIA["gini"], 1) is None
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert best_split_one_feature(x, stats, CRITERIA["gini"], 3) is None  # needs 6 rows
    assert best_split_one_feature(x, stats, CRITERIA["gini"], 2) is not None


def test_min_samples_leaf_always_respected(rng):
    crit = CRITERIA["gini"]
    for _ in range(100):
        n, msl = int(rng.integers(5, 50)), int(rng.integers(1, 6))
        x = rng.normal(size=n)
        res = best_split_one_feature(x, np.eye(3)[rng.integers(0, 3, size=n)], crit, msl)
        if res is not None:
            assert res[2] >= msl and n - res[2] >= msl


def test_threshold_rounding_guard():
    crit = CRITERIA["gini"]
    stats = np.eye(2)[[0, 1]]
    # Adjacent floats near 1.0 are closer than FEATURE_THRESHOLD (1e-7): no valid threshold.
    assert best_split_one_feature(np.array([1.0, np.nextafter(1.0, 2.0)]), stats, crit, 1) is None
    # Near 4e9 adjacent floats are ~4.8e-7 apart, so a split exists, but the midpoint rounds
    # onto one of the two values; the guard must keep "x <= tau" separating them.
    a = 4e9
    b = float(np.nextafter(a, 5e9))
    assert b > a + 1e-7
    x = np.array([a, b])
    cost, tau, n_left = best_split_one_feature(x, stats, crit, 1)
    assert tau == a and n_left == 1
    assert list(x <= tau) == [True, False]


@pytest.mark.parametrize(
    ("mf", "d", "expected"),
    [
        (None, 10, 10),
        ("sqrt", 100, 10),
        ("sqrt", 10, 3),
        ("sqrt", 1, 1),
        ("log2", 64, 6),
        ("log2", 10, 3),
        ("log2", 1, 1),
        (3, 10, 3),
        (10, 10, 10),
        (0.5, 10, 5),
        (1.0, 7, 7),
        (0.01, 10, 1),
        (0.25, 10, 2),
    ],
)
def test_resolve_max_features(mf, d, expected):
    assert resolve_max_features(mf, d) == expected


@pytest.mark.parametrize("bad", [0, 11, -1, 0.0, 1.5, -0.2, "auto", True, [3]])
def test_resolve_max_features_invalid(bad):
    with pytest.raises(ValueError):
        resolve_max_features(bad, 10)


def test_find_best_split_deterministic_scan(blobs_clf):
    X, y = blobs_clf
    stats = np.eye(3)[y]
    a = find_best_split(X, stats, CRITERIA["gini"], 1, X.shape[1], None)
    b = find_best_split(X, stats, CRITERIA["gini"], 1, X.shape[1], None)
    assert a == b
    # best over all features equals the min over per-feature scans (lowest index wins ties)
    per_feature = [best_split_one_feature(X[:, j], stats, CRITERIA["gini"], 1) for j in range(6)]
    j_best = int(np.argmin([p[0] for p in per_feature]))
    assert a.feature == j_best and a.cost == pytest.approx(per_feature[j_best][0])
    assert a.n_left + a.n_right == len(y)


def test_find_best_split_same_seed_same_split(blobs_clf):
    X, y = blobs_clf
    stats = np.eye(3)[y]
    s1 = find_best_split(X, stats, CRITERIA["gini"], 1, 2, np.random.default_rng(5))
    s2 = find_best_split(X, stats, CRITERIA["gini"], 1, 2, np.random.default_rng(5))
    assert s1 == s2 and s1 is not None


def test_find_best_split_requires_rng_for_subsampling(blobs_clf):
    X, y = blobs_clf
    with pytest.raises(ValueError, match="rng"):
        find_best_split(X, np.eye(3)[y], CRITERIA["gini"], 1, 2, None)


def test_find_best_split_continues_past_constant_features():
    g = np.random.default_rng(0)
    n = 40
    informative = g.normal(size=n)
    y = (informative > 0).astype(int)
    X = np.column_stack([np.ones(n), informative, np.full(n, 3.0)])  # only column 1 can split
    stats = np.eye(2)[y]
    for seed in range(20):  # whichever feature is drawn first, the F=1 rule must keep going
        s = find_best_split(X, stats, CRITERIA["gini"], 1, 1, np.random.default_rng(seed))
        assert s is not None and s.feature == 1
        assert s.cost == pytest.approx(0.0, abs=1e-9)
