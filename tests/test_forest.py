"""Random forests: wiring, determinism, OOB correctness, and agreement with scikit-learn."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier as SkRFC
from sklearn.ensemble import RandomForestRegressor as SkRFR

from treesfs import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    forest_from_config,
)
from treesfs.models.forest import _bootstrap_indices


def _noisy_blobs(n: int = 600, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """3-class data with heavy overlap, so a single deep tree overfits."""
    g = np.random.default_rng(seed)
    y = g.integers(0, 3, size=n)
    X = g.normal(size=(n, 8))
    X[:, :3] += np.array([[0, 0, 0], [1.0, 0.6, -0.6], [-0.6, 1.0, 0.8]])[y] * 0.9
    return X.astype(np.float32).astype(np.float64), y


# ----------------------------------------------------------------------------- basic API
def test_classifier_api_shapes_and_labels(blobs_clf):
    X, y = blobs_clf
    labels = np.array(["a", "b", "c"])[y]
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, labels)
    proba = rf.predict_proba(X)
    assert proba.shape == (300, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(rf.classes_) == ["a", "b", "c"]
    assert np.array_equal(rf.predict(X), rf.classes_[np.argmax(proba, axis=1)])
    assert rf.score(X, labels) > 0.95  # forests fit the training data well
    assert len(rf.estimators_) == 10


def test_regressor_api(reg_data):
    X, y = reg_data
    rf = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    assert rf.predict(X).shape == (300,)
    assert rf.score(X, y) > 0.8


def test_trees_have_aligned_value_width_when_bootstrap_misses_a_class():
    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = np.array([0] * 19 + [1] * 20 + [2])  # class 2 is a single row
    rf = RandomForestClassifier(n_estimators=30, random_state=3).fit(X, y)
    assert {t.tree_.value.shape[1] for t in rf.estimators_} == {3}
    assert np.allclose(rf.predict_proba(X).sum(axis=1), 1.0)


# ----------------------------------------------------------------------------- determinism
def test_same_seed_same_forest_and_different_seed_differs(blobs_clf):
    X, y = blobs_clf
    a = RandomForestClassifier(n_estimators=8, random_state=1).fit(X, y).predict_proba(X)
    b = RandomForestClassifier(n_estimators=8, random_state=1).fit(X, y).predict_proba(X)
    c = RandomForestClassifier(n_estimators=8, random_state=2).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_n_jobs_does_not_change_the_forest(blobs_clf):
    X, y = blobs_clf
    serial = RandomForestClassifier(n_estimators=6, random_state=4, n_jobs=1).fit(X, y)
    parallel = RandomForestClassifier(n_estimators=6, random_state=4, n_jobs=2).fit(X, y)
    for s, p in zip(serial.estimators_, parallel.estimators_, strict=True):
        assert np.array_equal(s.tree_.feature, p.tree_.feature)
        assert np.array_equal(s.tree_.threshold, p.tree_.threshold)
    assert np.array_equal(serial.predict_proba(X), parallel.predict_proba(X))


def test_refit_with_same_seedsequence_object_is_reproducible(blobs_clf):
    X, y = blobs_clf
    ss = np.random.SeedSequence(11)
    rf = RandomForestClassifier(n_estimators=4, random_state=ss)
    first = rf.fit(X, y).predict_proba(X)
    second = rf.fit(X, y).predict_proba(X)  # spawn() must not have advanced the shared object
    assert np.array_equal(first, second)


def test_tree_i_is_the_tree_grown_from_its_own_seeds(blobs_clf):
    """Re-grow tree 0 by hand from the seed tree and compare arrays exactly."""
    X, y = blobs_clf
    rf = RandomForestClassifier(n_estimators=3, random_state=9).fit(X, y)
    child = np.random.SeedSequence(9).spawn(3)[0]
    boot_seed, tree_seed = child.spawn(2)
    boot = _bootstrap_indices(boot_seed, len(y))
    expected = DecisionTreeClassifier(max_features="sqrt", random_state=tree_seed)
    expected._fit(X[boot], y[boot], 3)
    got = rf.estimators_[0].tree_
    assert np.array_equal(got.feature, expected.tree_.feature)
    assert np.array_equal(got.threshold, expected.tree_.threshold)
    assert np.array_equal(got.n_node_samples, expected.tree_.n_node_samples)


def test_no_bootstrap_all_features_reduces_to_a_single_tree(blobs_clf):
    X, y = blobs_clf
    rf = RandomForestClassifier(
        n_estimators=3, bootstrap=False, max_features=1.0, max_depth=4, random_state=0
    ).fit(X, y)
    tree = DecisionTreeClassifier(max_depth=4).fit(X, y)
    # mean of 3 identical trees is (p + p + p) / 3, which can differ from p by 1 ulp
    assert np.allclose(rf.predict_proba(X), tree.predict_proba(X), rtol=0, atol=1e-15)


def test_regressor_no_bootstrap_equals_single_tree(reg_data):
    X, y = reg_data
    rf = RandomForestRegressor(n_estimators=2, bootstrap=False, max_depth=5).fit(X, y)
    tree = DecisionTreeRegressor(max_depth=5).fit(X, y)
    assert np.allclose(rf.predict(X), tree.predict(X))


# ----------------------------------------------------------------------------- OOB
def _brute_force_oob(rf: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """Independent OOB aggregation: for each row, average the trees that never saw it."""
    n = X.shape[0]
    out = np.full((n, rf.n_classes_), np.nan)
    # Python loop over rows is the point: a slow, obviously-correct oracle for the vectorised code.
    for r in range(n):
        votes = [
            tree.tree_.value[tree.tree_.apply(X[r : r + 1])[0]]
            for i, tree in enumerate(rf.estimators_)
            if r in set(rf.oob_indices(i).tolist())
        ]
        if votes:
            out[r] = np.mean(votes, axis=0)
    return out


def test_oob_decision_function_matches_brute_force(blobs_clf):
    X, y = blobs_clf
    X, y = X[:120], y[:120]
    rf = RandomForestClassifier(n_estimators=15, oob_score=True, random_state=0).fit(X, y)
    ref = _brute_force_oob(rf, X)
    assert np.allclose(rf.oob_decision_function_, ref, equal_nan=True)
    covered = ~np.isnan(ref[:, 0])
    expected = np.mean(rf.classes_[np.argmax(ref[covered], axis=1)] == y[covered])
    assert rf.oob_score_ == pytest.approx(expected)


def test_oob_fraction_is_about_one_over_e(blobs_clf):
    X, y = blobs_clf
    rf = RandomForestClassifier(n_estimators=20, oob_score=True, random_state=0).fit(X, y)
    frac = np.mean([len(rf.oob_indices(i)) / len(y) for i in range(20)])
    assert frac == pytest.approx(np.exp(-1), abs=0.03)


def test_oob_rows_were_not_in_the_bootstrap_sample(blobs_clf):
    X, y = blobs_clf
    rf = RandomForestClassifier(n_estimators=5, oob_score=True, random_state=0).fit(X, y)
    for i in range(5):
        boot = _bootstrap_indices(rf._boot_seeds[i], len(y))
        assert np.intersect1d(boot, rf.oob_indices(i)).size == 0
        assert np.union1d(boot, rf.oob_indices(i)).size == len(y)


def test_oob_score_close_to_sklearn_and_to_held_out_accuracy():
    X, y = _noisy_blobs(600)
    Xtr, ytr, Xte, yte = X[:400], y[:400], X[400:], y[400:]
    mine = RandomForestClassifier(n_estimators=30, oob_score=True, random_state=0).fit(Xtr, ytr)
    ref = SkRFC(n_estimators=30, oob_score=True, random_state=0).fit(Xtr, ytr)
    assert abs(mine.oob_score_ - ref.oob_score_) < 0.05
    assert abs(mine.oob_score_ - mine.score(Xte, yte)) < 0.07  # OOB ~ honest generalisation


def test_regressor_oob_close_to_sklearn(reg_data):
    X, y = reg_data
    mine = RandomForestRegressor(n_estimators=20, oob_score=True, random_state=0).fit(X, y)
    ref = SkRFR(n_estimators=20, oob_score=True, random_state=0).fit(X, y)
    assert mine.oob_prediction_.shape == (300,) and not np.isnan(mine.oob_prediction_).any()
    assert abs(mine.oob_score_ - ref.oob_score_) < 0.08


def test_rows_never_oob_warn_and_are_nan(blobs_clf):
    X, y = blobs_clf
    with pytest.warns(UserWarning, match="never out-of-bag"):
        rf = RandomForestClassifier(n_estimators=2, oob_score=True, random_state=0).fit(X, y)
    assert np.isnan(rf.oob_decision_function_).any()
    assert np.isfinite(rf.oob_score_)


def test_no_warning_with_enough_trees(blobs_clf):
    X, y = blobs_clf
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RandomForestClassifier(n_estimators=20, oob_score=True, random_state=0).fit(X, y)


# ----------------------------------------------------------------------------- quality
def test_forest_beats_a_single_overfit_tree_and_tracks_sklearn():
    X, y = _noisy_blobs(600)
    Xtr, ytr, Xte, yte = X[:400], y[:400], X[400:], y[400:]
    tree_acc = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr).score(Xte, yte)
    rf = RandomForestClassifier(n_estimators=30, random_state=0).fit(Xtr, ytr)
    ref = SkRFC(n_estimators=30, random_state=0).fit(Xtr, ytr)
    assert rf.score(Xte, yte) > tree_acc + 0.03  # variance reduction is the whole point
    assert abs(rf.score(Xte, yte) - ref.score(Xte, yte)) < 0.05


def test_regression_forest_tracks_sklearn(reg_data):
    X, y = reg_data
    Xtr, ytr, Xte, yte = X[:200], y[:200], X[200:], y[200:]
    mine = RandomForestRegressor(n_estimators=20, random_state=0).fit(Xtr, ytr)
    ref = SkRFR(n_estimators=20, random_state=0).fit(Xtr, ytr)
    assert abs(mine.score(Xte, yte) - ref.score(Xte, yte)) < 0.06


def test_feature_importances(blobs_clf):
    X, y = blobs_clf
    rf = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    imp = rf.feature_importances_
    assert imp.shape == (6,) and (imp >= 0).all() and imp.sum() == pytest.approx(1.0)
    assert imp[:3].sum() > imp[3:].sum()  # only the first three columns carry signal
    ref = SkRFC(n_estimators=30, random_state=0).fit(X, y).feature_importances_
    assert np.corrcoef(imp, ref)[0, 1] > 0.9


# ----------------------------------------------------------------------------- staged scores
def test_staged_scores_endpoints_and_order(blobs_clf):
    X, y = blobs_clf
    rf = RandomForestClassifier(n_estimators=12, oob_score=True, random_state=0).fit(X, y)
    grid = [12, 1, 5]  # any order is returned in the order given
    test = rf.staged_scores(X, y, grid)
    assert test[0] == pytest.approx(rf.score(X, y))
    first = rf.estimators_[0].tree_  # k = 1 is just the first bootstrap tree
    assert test[1] == pytest.approx(np.mean(np.argmax(first.value[first.apply(X)], axis=1) == y))
    oob = rf.staged_scores(X, y, [12], oob=True)
    assert oob[0] == pytest.approx(rf.oob_score_)


def test_staged_scores_regressor(reg_data):
    X, y = reg_data
    rf = RandomForestRegressor(n_estimators=8, oob_score=True, random_state=0).fit(X, y)
    assert rf.staged_scores(X, y, [8])[0] == pytest.approx(rf.score(X, y))
    assert rf.staged_scores(X, y, [8], oob=True)[0] == pytest.approx(rf.oob_score_)


def test_staged_scores_validation(blobs_clf):
    X, y = blobs_clf
    rf = RandomForestClassifier(n_estimators=4, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="n_trees_grid"):
        rf.staged_scores(X, y, [5])
    with pytest.raises(ValueError, match="n_trees_grid"):
        rf.staged_scores(X, y, [0])
    with pytest.raises(ValueError, match="training data"):
        rf.staged_scores(X[:10], y[:10], [2], oob=True)
    with pytest.raises(ValueError, match="shape"):
        rf.staged_scores(X, y[:5], [2])


# ----------------------------------------------------------------------------- errors / config
def test_errors(blobs_clf):
    X, y = blobs_clf
    with pytest.raises(RuntimeError, match="not fitted"):
        RandomForestClassifier().predict(X)
    rf = RandomForestClassifier(n_estimators=2, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        rf.predict(X[:, :3])
    with pytest.raises(ValueError, match="oob_indices requires bootstrap"):
        RandomForestClassifier(n_estimators=2, bootstrap=False).fit(X, y).oob_indices(0)
    with pytest.raises(IndexError):
        rf.oob_indices(5)
    with pytest.raises(ValueError, match="n_estimators"):
        RandomForestClassifier(n_estimators=0).fit(X, y)
    with pytest.raises(ValueError, match="oob_score=True requires bootstrap"):
        RandomForestClassifier(oob_score=True, bootstrap=False).fit(X, y)
    with pytest.raises(TypeError, match="random_state"):
        RandomForestClassifier(random_state=np.random.default_rng(0)).fit(X, y)
    with pytest.raises(ValueError, match="rows"):
        RandomForestClassifier().fit(X, y[:5])
    with pytest.raises(ValueError, match="NaN"):
        RandomForestRegressor().fit(X, np.full(300, np.nan))
    with pytest.raises(ValueError, match="criterion"):
        RandomForestClassifier(n_estimators=1, criterion="mse").fit(X, y)


def test_repr_and_get_params():
    rf = RandomForestClassifier(n_estimators=7, max_depth=3)
    assert rf.get_params()["n_estimators"] == 7
    assert "RandomForestClassifier(n_estimators=7" in repr(rf)


def test_forest_from_config_uses_task_defaults(default_config):
    clf = forest_from_config("classification", default_config.forest, default_config.tree, 3)
    reg = forest_from_config("regression", default_config.forest, default_config.tree, 3)
    assert isinstance(clf, RandomForestClassifier) and isinstance(reg, RandomForestRegressor)
    assert clf.n_estimators == default_config.forest.n_estimators and clf.random_state == 3
    assert clf._prototype().max_features == "sqrt"
    assert reg._prototype().max_features == 1.0
    assert (
        forest_from_config(
            "classification", default_config.forest, default_config.tree, 3, n_estimators=5
        ).n_estimators
        == 5
    )
    with pytest.raises(ValueError, match="task"):
        forest_from_config("ranking", default_config.forest, default_config.tree, 3)
