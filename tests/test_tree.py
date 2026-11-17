"""Tree: constraints, perfect fit, tie-aware match vs scikit-learn (DESIGN.md §9.3), errors."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier as SkClassifier
from sklearn.tree import DecisionTreeRegressor as SkRegressor

from treesfs import DecisionTreeClassifier, DecisionTreeRegressor
from treesfs.criteria import entropy, gini
from treesfs.models.tree import TreeArrays, tree_from_config

# ----------------------------------------------------------------------------- basics


def test_pure_y_gives_single_leaf():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    clf = DecisionTreeClassifier().fit(X, np.ones(10, dtype=int))
    assert clf.tree_.node_count == 1 and clf.get_depth() == 0 and clf.get_n_leaves() == 1
    assert clf.predict(X).tolist() == [1] * 10
    reg = DecisionTreeRegressor().fit(X, np.full(10, 2.5))
    assert reg.tree_.node_count == 1
    assert reg.predict(X).tolist() == [2.5] * 10
    assert not np.any(reg.feature_importances_)


@pytest.mark.parametrize("k", [0, 1, 2, 5])
def test_max_depth_respected(blobs_clf, k):
    X, y = blobs_clf
    clf = DecisionTreeClassifier(max_depth=k).fit(X, y)
    assert clf.get_depth() <= k
    if k == 0:
        assert clf.tree_.node_count == 1
        assert clf.predict(X).tolist() == [np.bincount(y).argmax()] * len(y)
    else:
        assert clf.get_depth() == k  # blobs overlap, so the tree really uses every level


@pytest.mark.parametrize("msl", [1, 3, 10, 40])
def test_min_samples_leaf_respected(blobs_clf, msl):
    X, y = blobs_clf
    t = DecisionTreeClassifier(min_samples_leaf=msl).fit(X, y).tree_
    leaves = t.children_left == -1
    assert t.n_node_samples[leaves].min() >= msl


@pytest.mark.parametrize("mss", [2, 5, 30, 100])
def test_min_samples_split_respected(blobs_clf, mss):
    X, y = blobs_clf
    t = DecisionTreeClassifier(min_samples_split=mss).fit(X, y).tree_
    internal = t.children_left != -1
    assert t.n_node_samples[internal].min() >= mss


def test_unlimited_tree_fits_training_data(blobs_clf, reg_data):
    X, y = blobs_clf
    assert DecisionTreeClassifier().fit(X, y).score(X, y) == 1.0
    assert DecisionTreeClassifier(criterion="entropy").fit(X, y).score(X, y) == 1.0
    Xr, yr = reg_data
    assert DecisionTreeRegressor().fit(Xr, yr).score(Xr, yr) == pytest.approx(1.0)


def test_predict_proba_shape_sum_and_classes(blobs_clf):
    X, y = blobs_clf
    clf = DecisionTreeClassifier(max_depth=4).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (300, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert clf.classes_.tolist() == [0, 1, 2]
    assert (clf.predict(X) == clf.classes_[proba.argmax(axis=1)]).all()


def test_string_labels_roundtrip():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array(["no", "no", "no", "yes", "yes", "yes"])
    clf = DecisionTreeClassifier().fit(X, y)
    assert clf.classes_.tolist() == ["no", "yes"]
    assert clf.predict(X).tolist() == y.tolist()
    assert clf.score(X, y) == 1.0
    assert clf.predict_proba(X).shape == (6, 2)


def test_apply_returns_leaves(blobs_clf):
    X, y = blobs_clf
    clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
    leaf_ids = clf.apply(X)
    assert (clf.tree_.children_left[leaf_ids] == -1).all()
    assert np.array_equal(clf.tree_.value[leaf_ids], clf.predict_proba(X))


def test_tree_arrays_invariants(blobs_clf):
    X, y = blobs_clf
    t = DecisionTreeClassifier(max_depth=5).fit(X, y).tree_
    assert isinstance(t, TreeArrays)
    leaf = t.children_left == -1
    assert (t.feature[leaf] == -2).all() and (t.threshold[leaf] == -2.0).all()
    assert t.n_node_samples[0] == len(y)
    internal = np.flatnonzero(~leaf)
    # children's samples add up; preorder ids: left child is always parent + 1
    sums = (
        t.n_node_samples[t.children_left[internal]] + t.n_node_samples[t.children_right[internal]]
    )
    assert np.array_equal(sums, t.n_node_samples[internal])
    assert np.array_equal(t.children_left[internal], internal + 1)
    assert t.node_count == 2 * t.n_leaves() - 1


def test_get_params_and_repr():
    clf = DecisionTreeClassifier(max_depth=3, random_state=1)
    assert clf.get_params()["max_depth"] == 3
    assert "DecisionTreeClassifier(criterion='gini'" in repr(clf)


def test_tree_from_config(default_config):
    clf = tree_from_config("classification", default_config.tree, 7, max_depth=4)
    reg = tree_from_config("regression", default_config.tree, 7)
    assert isinstance(clf, DecisionTreeClassifier) and clf.max_depth == 4
    assert isinstance(reg, DecisionTreeRegressor) and reg.random_state == 7
    with pytest.raises(ValueError):
        tree_from_config("ranking", default_config.tree, 0)


# ------------------------------------------------------------------ vs scikit-learn


def _split_cost(X, y, rows, f, thr, name):
    """Children cost of splitting ``rows`` at ``X[:, f] <= thr`` (direct partition)."""
    fn = gini if name == "gini" else entropy
    mask = X[rows, f] <= thr
    k = int(y.max()) + 1
    cost = 0.0
    for part in (rows[mask], rows[~mask]):
        cost += len(part) * float(fn(np.bincount(y[part], minlength=k)))
    return cost


def _tie_aware_match(mine: TreeArrays, sk, X, y, name):
    """Walk both trees in lockstep; return number of nodes compared and ties encountered."""
    stack = [(0, 0, np.arange(len(y)))]
    compared = ties = 0
    while stack:
        a, b, rows = stack.pop()
        a_leaf = mine.children_left[a] == -1
        b_leaf = sk.children_left[b] == -1
        assert a_leaf == b_leaf, "leaf-ness differs at a node reached by the same rows"
        if a_leaf:
            continue
        compared += 1
        fa, ta = int(mine.feature[a]), float(mine.threshold[a])
        fb, tb = int(sk.feature[b]), float(sk.threshold[b])
        if fa == fb and np.isclose(ta, tb, rtol=1e-12, atol=0):
            left = X[rows, fa] <= ta
            stack.append((int(mine.children_right[a]), int(sk.children_right[b]), rows[~left]))
            stack.append((int(mine.children_left[a]), int(sk.children_left[b]), rows[left]))
        else:
            ca, cb = _split_cost(X, y, rows, fa, ta, name), _split_cost(X, y, rows, fb, tb, name)
            assert ca == pytest.approx(cb, abs=1e-9), "different splits with different cost"
            ties += 1  # legitimate tie: subtrees may diverge from here, stop descending
    return compared, ties


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize(("depth", "msl"), [(3, 1), (4, 5)])
def test_tie_aware_structural_match_vs_sklearn(blobs_clf, criterion, depth, msl):
    X, y = blobs_clf
    mine = DecisionTreeClassifier(criterion=criterion, max_depth=depth, min_samples_leaf=msl)
    mine.fit(X, y)
    sk = SkClassifier(
        criterion=criterion, max_depth=depth, min_samples_leaf=msl, random_state=0
    ).fit(X, y)
    assert mine.tree_.feature[0] == sk.tree_.feature[0]
    assert mine.tree_.threshold[0] == pytest.approx(sk.tree_.threshold[0], rel=1e-12)
    compared, _ = _tie_aware_match(mine.tree_, sk.tree_, X, y, criterion)
    assert compared >= 1
    # whatever the ties, predictions on the training data should (nearly) agree
    assert (mine.predict(X) == sk.predict(X)).mean() >= 0.9


def test_regression_structure_equals_sklearn(reg_data):
    X, y = reg_data
    mine = DecisionTreeRegressor(max_depth=4, min_samples_leaf=3).fit(X, y)
    sk = SkRegressor(max_depth=4, min_samples_leaf=3, random_state=0).fit(X, y)
    a, b = mine.tree_, sk.tree_
    assert np.array_equal(a.children_left, b.children_left)
    assert np.array_equal(a.children_right, b.children_right)
    assert np.array_equal(a.feature, b.feature)
    assert np.array_equal(a.threshold, b.threshold)
    assert np.array_equal(a.n_node_samples, b.n_node_samples)
    assert np.allclose(a.impurity, b.impurity)
    assert np.allclose(a.value[:, 0], b.value[:, 0, 0])
    assert np.allclose(mine.predict(X), sk.predict(X))


def test_unlimited_depth_agrees_with_sklearn(blobs_clf):
    X, y = blobs_clf
    Xtr, ytr, Xte, yte = X[:200], y[:200], X[200:], y[200:]
    mine = DecisionTreeClassifier().fit(Xtr, ytr)
    sk = SkClassifier(random_state=0).fit(Xtr, ytr)
    assert (mine.predict(Xtr) == sk.predict(Xtr)).mean() == 1.0
    assert abs(mine.score(Xte, yte) - sk.score(Xte, yte)) <= 0.02 + 1e-9  # float slack


@pytest.mark.parametrize("mid", [0.005, 0.02])
def test_min_impurity_decrease_matches_sklearn_node_count(blobs_clf, mid):
    X, y = blobs_clf
    mine = DecisionTreeClassifier(min_impurity_decrease=mid).fit(X, y)
    # sklearn breaks cross-feature ties randomly, which can change the node count at 0.005
    # (49 or 53 depending on its seed), so compare with the set of counts it can produce.
    sk_counts = {
        SkClassifier(min_impurity_decrease=mid, random_state=s).fit(X, y).tree_.node_count
        for s in range(8)
    }
    assert mine.tree_.node_count in sk_counts
    assert mine.tree_.node_count > 1  # the threshold prunes, but does not erase, the tree
    # every split we kept really clears the threshold (weighted by N, DESIGN.md §2.1)
    t = mine.tree_
    for i in np.flatnonzero(t.children_left != -1):
        left, right = t.children_left[i], t.children_right[i]
        gain = (
            t.n_node_samples[i] * t.impurity[i]
            - t.n_node_samples[left] * t.impurity[left]
            - t.n_node_samples[right] * t.impurity[right]
        ) / len(y)
        assert gain >= mid - 1e-9


def test_feature_importances(reg_data):
    X, y = reg_data
    mine = DecisionTreeRegressor(max_depth=3).fit(X, y)
    sk = SkRegressor(max_depth=3, random_state=0).fit(X, y)
    imp = mine.feature_importances_
    assert imp.shape == (5,) and (imp >= 0).all() and imp.sum() == pytest.approx(1.0)
    assert np.array_equal(mine.tree_.feature, sk.tree_.feature)  # structures match
    assert np.allclose(imp, sk.feature_importances_)


def test_feature_importances_classifier_sum_to_one(blobs_clf):
    X, y = blobs_clf
    imp = DecisionTreeClassifier(max_depth=5).fit(X, y).feature_importances_
    assert imp.sum() == pytest.approx(1.0) and (imp >= 0).all()
    assert imp[:3].sum() > imp[3:].sum()  # only the first three features carry signal


# ------------------------------------------------------------------ determinism


def test_default_tree_independent_of_random_state(blobs_clf):
    X, y = blobs_clf
    a = DecisionTreeClassifier(random_state=0).fit(X, y).tree_
    b = DecisionTreeClassifier(random_state=123).fit(X, y).tree_
    for name in ("children_left", "children_right", "feature", "threshold", "value"):
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_max_features_seeded(blobs_clf):
    X, y = blobs_clf

    def grow(seed):
        return DecisionTreeClassifier(max_features="sqrt", max_depth=6, random_state=seed).fit(X, y)

    a, b = grow(3).tree_, grow(3).tree_
    assert np.array_equal(a.feature, b.feature) and np.array_equal(a.threshold, b.threshold)
    features = {tuple(grow(s).tree_.feature.tolist()) for s in range(6)}
    assert len(features) > 1  # different seeds usually differ


def test_max_features_changes_tree_but_still_valid(blobs_clf):
    X, y = blobs_clf
    clf = DecisionTreeClassifier(max_features=1, random_state=0).fit(X, y)
    assert clf.score(X, y) == 1.0  # unlimited depth still memorizes


# ------------------------------------------------------------------ truncation identity


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_truncation_identity_classifier(blobs_clf, k):
    X, y = blobs_clf
    full = DecisionTreeClassifier().fit(X, y)
    fresh = DecisionTreeClassifier(max_depth=k).fit(X, y)
    leaf_values = full.tree_.value[full.tree_.apply(X, max_depth=k)]
    assert np.allclose(leaf_values, fresh.predict_proba(X))


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_truncation_identity_regressor(reg_data, k):
    X, y = reg_data
    full = DecisionTreeRegressor().fit(X, y)
    fresh = DecisionTreeRegressor(max_depth=k).fit(X, y)
    leaf_values = full.tree_.value[full.tree_.apply(X, max_depth=k), 0]
    assert np.allclose(leaf_values, fresh.predict(X))


# ------------------------------------------------------------------ errors


def test_predict_before_fit_raises():
    X = np.zeros((2, 2))
    for model in (DecisionTreeClassifier(), DecisionTreeRegressor()):
        with pytest.raises(RuntimeError, match="is not fitted"):
            model.predict(X)
        with pytest.raises(RuntimeError, match="is not fitted"):
            model.get_depth()


def test_wrong_n_features_raises(blobs_clf):
    X, y = blobs_clf
    clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        clf.predict(X[:, :3])


def test_invalid_inputs_raise(blobs_clf):
    X, y = blobs_clf
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        DecisionTreeClassifier().fit(bad, y)
    with pytest.raises(ValueError, match="rows"):
        DecisionTreeClassifier().fit(X, y[:-1])
    with pytest.raises(ValueError, match="2-D"):
        DecisionTreeClassifier().fit(X[:, 0], y)
    with pytest.raises(ValueError, match="1-D"):
        DecisionTreeClassifier().fit(X, y.reshape(-1, 1))
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(X[:0], y[:0])
    with pytest.raises(ValueError, match="NaN"):
        DecisionTreeRegressor().fit(X, np.where(np.arange(len(y)) == 0, np.nan, 1.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_samples_leaf": 0},
        {"min_samples_split": 1},
        {"max_depth": -1},
        {"min_impurity_decrease": -0.1},
        {"criterion": "squared_error"},
        {"max_features": 99},
    ],
)
def test_bad_hyperparameters_raise(blobs_clf, kwargs):
    X, y = blobs_clf
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**kwargs).fit(X, y)


def test_regressor_rejects_classification_criterion(reg_data):
    X, y = reg_data
    with pytest.raises(ValueError, match="criterion"):
        DecisionTreeRegressor(criterion="gini").fit(X, y)
    DecisionTreeRegressor(criterion="mse", max_depth=2).fit(X, y)  # alias accepted


def test_single_sample_and_constant_features():
    clf = DecisionTreeClassifier().fit(np.array([[1.0, 2.0]]), np.array([3]))
    assert clf.tree_.node_count == 1 and clf.predict(np.array([[0.0, 0.0]])).tolist() == [3]
    clf = DecisionTreeClassifier().fit(np.ones((6, 2)), np.array([0, 1, 0, 1, 0, 0]))
    assert clf.tree_.node_count == 1  # nothing to split on
    assert clf.predict(np.ones((1, 2))).tolist() == [0]
