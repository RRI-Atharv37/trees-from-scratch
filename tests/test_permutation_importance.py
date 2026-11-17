"""Permutation importance: exact agreement with a slow loop, statistics, and sklearn."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.inspection import permutation_importance as sk_permutation_importance

from treesfs import (
    DecisionTreeClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    mse,
    permutation_importance,
)


@pytest.fixture
def signal_and_noise() -> tuple[np.ndarray, np.ndarray]:
    """Column 0 determines y; columns 1-3 are pure noise."""
    g = np.random.default_rng(0)
    X = g.normal(size=(400, 4))
    return X, (X[:, 0] > 0).astype(int)


def test_signal_column_dominates_noise(signal_and_noise):
    X, y = signal_and_noise
    model = DecisionTreeClassifier(max_depth=3).fit(X[:200], y[:200])
    res = permutation_importance(model, X[200:], y[200:], n_repeats=10, random_state=0)
    assert res.importances.shape == (4, 10)
    assert res.importances_mean[0] > 0.3  # shuffling the informative column ~halves accuracy
    assert np.abs(res.importances_mean[1:]).max() < 0.05
    assert res.baseline_score == pytest.approx(model.score(X[200:], y[200:]))
    assert res.n_rows == 200


def test_matches_slow_reference_loop_exactly(signal_and_noise):
    """Replicate the documented seeding with an obviously-correct double loop."""
    X, y = signal_and_noise
    X, y = X[:60], y[:60]
    model = DecisionTreeClassifier(max_depth=3).fit(X, y)
    res = permutation_importance(model, X, y, n_repeats=3, random_state=7)

    _, feature_root = np.random.SeedSequence(7).spawn(2)
    seeds = feature_root.spawn(X.shape[1])
    base = model.score(X, y)
    ref = np.empty((X.shape[1], 3))
    for j in range(X.shape[1]):  # slow on purpose: tiny data, clarity over speed
        rng = np.random.default_rng(seeds[j])
        for r in range(3):
            Xp = X.copy()
            Xp[:, j] = X[rng.permutation(len(y)), j]
            ref[j, r] = base - model.score(Xp, y)
    assert np.array_equal(res.importances, ref)
    assert np.allclose(res.importances_mean, ref.mean(axis=1))
    assert np.allclose(res.importances_std, ref.std(axis=1))


def test_input_is_not_modified(signal_and_noise):
    X, y = signal_and_noise
    model = DecisionTreeClassifier(max_depth=3).fit(X, y)
    before = X.copy()
    permutation_importance(model, X, y, n_repeats=2, random_state=0)
    assert np.array_equal(X, before)


def test_result_independent_of_n_jobs_and_seeded(signal_and_noise):
    X, y = signal_and_noise
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    a = permutation_importance(model, X, y, n_repeats=3, random_state=1, n_jobs=1)
    b = permutation_importance(model, X, y, n_repeats=3, random_state=1, n_jobs=3)
    c = permutation_importance(model, X, y, n_repeats=3, random_state=2, n_jobs=1)
    assert np.array_equal(a.importances, b.importances)
    assert not np.array_equal(a.importances, c.importances)


def test_max_rows_subsamples_deterministically(signal_and_noise):
    X, y = signal_and_noise
    model = DecisionTreeClassifier(max_depth=3).fit(X, y)
    a = permutation_importance(model, X, y, n_repeats=2, random_state=0, max_rows=100)
    b = permutation_importance(model, X, y, n_repeats=2, random_state=0, max_rows=100)
    assert a.n_rows == 100 and np.array_equal(a.importances, b.importances)
    full = permutation_importance(model, X, y, n_repeats=1, random_state=0, max_rows=10_000)
    assert full.n_rows == 400  # cap above n keeps every row


def test_regression_and_custom_scoring(reg_data):
    X, y = reg_data
    model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X[:200], y[:200])
    res = permutation_importance(model, X[200:], y[200:], n_repeats=4, random_state=0)
    # y depends on columns 0, 1, 2 only
    assert res.importances_mean[:3].min() > res.importances_mean[3:].max()
    neg_mse = lambda m, X_, y_: -mse(y_, m.predict(X_))  # noqa: E731
    res2 = permutation_importance(
        model, X[200:], y[200:], n_repeats=2, random_state=0, scoring=neg_mse
    )
    assert res2.importances_mean[:3].min() > 0  # error rises when a useful column is shuffled


def test_agrees_with_sklearn_in_expectation(blobs_clf):
    """Different RNG streams, so compare means over many repeats, not exact values."""
    X, y = blobs_clf
    model = RandomForestClassifier(n_estimators=15, random_state=0).fit(X[:200], y[:200])
    mine = permutation_importance(model, X[200:], y[200:], n_repeats=40, random_state=0)
    ref = sk_permutation_importance(
        model,
        X[200:],
        y[200:],
        scoring=lambda m, X_, y_: m.score(X_, y_),
        n_repeats=40,
        random_state=0,
    )
    assert np.abs(mine.importances_mean - ref.importances_mean).max() < 0.03


def test_validation(signal_and_noise):
    X, y = signal_and_noise
    model = DecisionTreeClassifier(max_depth=2).fit(X, y)
    with pytest.raises(ValueError, match="2-D"):
        permutation_importance(model, X[:, 0], y)
    with pytest.raises(ValueError, match="shape"):
        permutation_importance(model, X, y[:5])
    with pytest.raises(ValueError, match="n_repeats"):
        permutation_importance(model, X, y, n_repeats=0)
    with pytest.raises(ValueError, match="max_rows"):
        permutation_importance(model, X, y, max_rows=1)
