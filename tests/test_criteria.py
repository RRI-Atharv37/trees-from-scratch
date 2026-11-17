"""Criteria: hand-computed values (DESIGN.md §1.4, §2.3) and properties (§2.2, §3.2)."""

from __future__ import annotations

import numpy as np
import pytest

from treesfs.criteria import CRITERIA, entropy, gini, impurity_decrease, squared_error


@pytest.mark.parametrize(
    ("counts", "g", "h"),
    [
        ([5, 5], 0.5, 1.0),
        ([10, 0], 0.0, 0.0),
        ([2, 3, 5], 0.62, 1.4854752972273344),
        ([1, 1, 1, 1], 0.75, 2.0),
    ],
)
def test_design_values_classification(counts, g, h):
    assert gini(counts) == pytest.approx(g, rel=1e-12, abs=1e-15)
    assert entropy(counts) == pytest.approx(h, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize(("y", "v"), [([1, 2, 3, 4], 1.25), ([2, 2], 0.0)])
def test_design_values_variance(y, v):
    assert squared_error(y) == pytest.approx(v, rel=1e-12, abs=1e-15)


def test_pure_node_is_zero_for_all_criteria():
    assert gini([7, 0, 0]) == 0.0
    assert entropy([0, 7, 0]) == 0.0
    assert squared_error([3.5, 3.5, 3.5]) == 0.0
    assert gini([0, 0]) == 0.0 and entropy([0, 0]) == 0.0  # empty node: no NaN


@pytest.mark.parametrize("k", [2, 3, 5, 10])
def test_uniform_classes(k):
    counts = np.full(k, 4)
    assert gini(counts) == pytest.approx(1 - 1 / k)
    assert entropy(counts) == pytest.approx(np.log2(k))


def test_batched_equals_loop(rng):
    counts = rng.integers(0, 20, size=(25, 4))
    assert np.allclose(gini(counts), [gini(c) for c in counts])
    assert np.allclose(entropy(counts), [entropy(c) for c in counts])
    assert gini(counts).shape == (25,)
    assert gini(rng.integers(0, 5, size=(3, 4, 5))).shape == (3, 4)


def test_range_properties(rng):
    for _ in range(200):
        k = int(rng.integers(2, 8))
        c = rng.integers(0, 30, size=k)
        if c.sum() == 0:
            continue
        assert 0.0 <= gini(c) <= 1 - 1 / k + 1e-12
        assert 0.0 <= entropy(c) <= np.log2(k) + 1e-12


def test_impurity_decrease_design_vectors():
    # classification: parent [4, 2], children [3, 0] and [1, 2]
    parent, left, right = gini([4, 2]), gini([3, 0]), gini([1, 2])
    assert impurity_decrease(6, 6, parent, 3, left, 3, right) == pytest.approx(2 / 9)
    # regression: y = [1, 1, 5, 5] split in the middle
    y = np.array([1.0, 1.0, 5.0, 5.0])
    d = impurity_decrease(4, 4, squared_error(y), 2, squared_error(y[:2]), 2, squared_error(y[2:]))
    assert d == pytest.approx(4.0)


@pytest.mark.parametrize("name", ["gini", "entropy"])
def test_concavity_classification(rng, name):
    fn = gini if name == "gini" else entropy
    for _ in range(300):
        k = int(rng.integers(2, 6))
        cl, cr = rng.integers(0, 15, size=k), rng.integers(0, 15, size=k)
        nl, nr = cl.sum(), cr.sum()
        if nl == 0 or nr == 0:
            continue
        d = impurity_decrease(nl + nr, nl + nr, fn(cl + cr), nl, fn(cl), nr, fn(cr))
        assert d >= -1e-12


def test_concavity_regression(rng):
    for _ in range(300):
        yl, yr = (
            rng.normal(size=int(rng.integers(1, 20))),
            rng.normal(size=int(rng.integers(1, 20))),
        )
        y = np.concatenate([yl, yr])
        d = impurity_decrease(
            len(y), len(y), squared_error(y), len(yl), squared_error(yl), len(yr), squared_error(yr)
        )
        assert d >= -1e-12


@pytest.mark.parametrize("name", ["gini", "entropy"])
def test_children_cost_matches_direct_classification(rng, name):
    crit = CRITERIA[name]
    fn = gini if name == "gini" else entropy
    for _ in range(30):
        m, k = int(rng.integers(2, 40)), int(rng.integers(2, 5))
        labels = rng.integers(0, k, size=m)
        stats = np.eye(k)[labels]
        n_left = np.arange(1, m, dtype=float)
        cost = crit.children_cost(np.cumsum(stats, axis=0), n_left, m - n_left)
        for i in range(m - 1):  # test oracle: direct per-position computation
            cl = stats[: i + 1].sum(axis=0)
            cr = stats[i + 1 :].sum(axis=0)
            expected = (i + 1) * fn(cl) + (m - i - 1) * fn(cr)
            assert cost[i] == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_children_cost_matches_direct_regression(rng):
    crit = CRITERIA["squared_error"]
    for _ in range(30):
        m = int(rng.integers(2, 40))
        y = rng.normal(loc=1e4, size=m)
        ys = y - y.mean()
        n_left = np.arange(1, m, dtype=float)
        cost = crit.children_cost(np.cumsum(ys), n_left, m - n_left, float(ys @ ys))
        for i in range(m - 1):
            expected = (i + 1) * squared_error(y[: i + 1]) + (m - i - 1) * squared_error(y[i + 1 :])
            assert cost[i] == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_mse_alias_and_unknown():
    assert CRITERIA["mse"] is CRITERIA["squared_error"]
    from treesfs.criteria import get_criterion

    with pytest.raises(ValueError, match="Unknown criterion"):
        get_criterion("nope")
