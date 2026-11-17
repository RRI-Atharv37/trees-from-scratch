"""Random forests: bagged CART trees with out-of-bag (OOB) evaluation. NumPy only.

A forest grows ``B`` decorrelated trees and averages them (DESIGN.md §11):

* **Bagging.** Tree ``b`` is trained on a bootstrap sample of the rows, ``n`` draws with
  replacement. Each row is missed by a given tree with probability ``(1 - 1/n)^n -> 1/e ~ 0.368``.
* **Feature subsampling.** Every node looks at ``F`` randomly chosen features (``sqrt(d)`` for
  classification, all ``d`` for regression, as in scikit-learn >= 1.1), which decorrelates trees.
* **Averaging.** ``predict_proba`` is the mean of the trees' leaf class proportions; the
  regressor averages leaf means. If the trees' errors have variance ``s^2`` and pairwise
  correlation ``rho``, the average has variance ``rho s^2 + (1 - rho) s^2 / B``: more trees
  shrink the second term, and bagging + feature subsampling shrink ``rho``.
* **OOB estimate.** A row's OOB prediction averages only the trees that did *not* see it, so
  ``oob_score_`` is a nearly free, honest estimate of generalisation (no held-out set needed).

Reproducibility: one ``random_state`` feeds a ``SeedSequence`` that is spawned into one
independent child per tree (and each child into a bootstrap stream and a tree stream). A tree
therefore depends only on its own seed, never on ``n_jobs`` or on scheduling order.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from joblib import Parallel, delayed

from treesfs.metrics import accuracy, r2_score
from treesfs.models.tree import (
    BaseDecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    _check_X,
)

if TYPE_CHECKING:  # pragma: no cover
    from treesfs.config import ForestConfig, TreeConfig


def _bootstrap_indices(seed: np.random.SeedSequence, n: int) -> np.ndarray:
    """``n`` row indices drawn with replacement from ``[0, n)``, shape ``(n,)``.

    Deterministic in ``seed`` alone, so the OOB rows of a fitted tree can be recomputed later
    instead of being stored.
    """
    return np.random.default_rng(seed).integers(0, n, size=n)


def _oob_rows(boot: np.ndarray, n: int) -> np.ndarray:
    """Rows never drawn by the bootstrap sample ``boot``: ``bincount(boot) == 0``."""
    return np.flatnonzero(np.bincount(boot, minlength=n) == 0)


def _grow_tree(
    proto: BaseDecisionTree,
    boot_seed: np.random.SeedSequence,
    tree_seed: np.random.SeedSequence,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int | None,
    bootstrap: bool,
    want_oob: bool,
) -> tuple[BaseDecisionTree, np.ndarray | None, np.ndarray | None]:
    """Grow one tree (module-level so joblib can pickle it).

    Returns the fitted tree and, if ``want_oob``, the OOB row indices ``(m,)`` and the tree's
    leaf values on those rows ``(m, K or 1)``, so OOB aggregation is parallel too.
    """
    n = X.shape[0]
    params = {**proto.get_params(), "random_state": tree_seed}
    tree = type(proto)(**params)
    if bootstrap:
        boot = _bootstrap_indices(boot_seed, n)
        tree._fit(X[boot], y[boot], n_classes)  # forest hook, DESIGN.md §5.3 / §11
    else:
        tree._fit(X, y, n_classes)
    if not want_oob:
        return tree, None, None
    oob = _oob_rows(boot, n)
    values = tree.tree_.value[tree.tree_.apply(X[oob])]
    return tree, oob, values


class BaseForest:
    """Shared bagging machinery; use :class:`RandomForestClassifier` / ``Regressor``.

    Parameters
    ----------
    n_estimators : int
        Number of trees ``B >= 1``.
    criterion, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease
        Passed to every tree (see :class:`~treesfs.models.tree.BaseDecisionTree`).
    max_features : None, int, float or {"sqrt", "log2"}
        Features tried per node; ``None`` selects the task default (``"sqrt"`` for the
        classifier, ``1.0`` for the regressor).
    bootstrap : bool
        Train each tree on a bootstrap sample (``True``) or on all rows (``False``).
    oob_score : bool
        Compute ``oob_score_`` and the OOB predictions (requires ``bootstrap=True``).
    n_jobs : int or None
        Worker processes for tree growing (joblib semantics; ``None`` / ``1`` = serial,
        ``-1`` = all cores). Never changes the result.
    random_state : int, SeedSequence or None
        Root seed. ``None`` draws fresh OS entropy.
    """

    _is_classifier: bool = False
    _tree_cls: type[BaseDecisionTree]
    _default_max_features: Any = None

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: None | int | float | str = None,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int | None = None,
        random_state: int | np.random.SeedSequence | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

    # ------------------------------------------------------------------ params
    def get_params(self) -> dict[str, Any]:
        """Constructor parameters as a dict."""
        return {
            "n_estimators": self.n_estimators,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "oob_score": self.oob_score,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"

    def _validate_params(self) -> None:
        if isinstance(self.n_estimators, bool) or int(self.n_estimators) != self.n_estimators:
            raise ValueError(f"n_estimators must be an int >= 1, got {self.n_estimators!r}")
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}")
        if self.oob_score and not self.bootstrap:
            raise ValueError("oob_score=True requires bootstrap=True (otherwise no row is OOB)")
        if self.random_state is not None and not isinstance(
            self.random_state, int | np.integer | np.random.SeedSequence
        ):
            raise TypeError(
                "random_state must be an int, a numpy SeedSequence or None, "
                f"got {type(self.random_state).__name__}"
            )

    def _prototype(self) -> BaseDecisionTree:
        """An unfitted tree carrying the shared hyper-parameters (validated at construction)."""
        max_features = self.max_features
        if max_features is None:
            max_features = self._default_max_features
        return self._tree_cls(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=max_features,
        )

    # ------------------------------------------------------------------ growth
    def _fit(self, X: np.ndarray, y: np.ndarray, n_classes: int | None) -> BaseForest:
        """Grow ``n_estimators`` trees on validated data.

        Parameters
        ----------
        X : ndarray, shape (n, d) float64
        y : ndarray, shape (n,)
            Class codes in ``[0, n_classes)`` or float targets.
        n_classes : int or None
            ``K`` shared by all trees, so each tree's ``value`` is aligned even when a
            bootstrap sample misses a class and a plain mean of ``value`` is correct.
        """
        self._validate_params()
        proto = self._prototype()
        proto._validate_params()
        n, d = X.shape
        root = (
            self.random_state
            if isinstance(self.random_state, np.random.SeedSequence)
            else np.random.SeedSequence(
                None if self.random_state is None else int(self.random_state)
            )
        )
        # spawn() mutates the parent's counter; copy so refitting with the same SeedSequence
        # object reproduces the same forest.
        root = np.random.SeedSequence(root.entropy, spawn_key=root.spawn_key)
        children = root.spawn(self.n_estimators)
        seeds = [tuple(c.spawn(2)) for c in children]  # (bootstrap stream, tree stream)

        results = Parallel(n_jobs=self.n_jobs)(  # ordered: result i is tree i for any n_jobs
            delayed(_grow_tree)(proto, b, t, X, y, n_classes, self.bootstrap, self.oob_score)
            for b, t in seeds
        )
        self.estimators_ = [r[0] for r in results]
        self._boot_seeds = [s[0] for s in seeds]
        self.n_samples_fit_ = n
        self.n_features_in_ = d

        # Mean of per-tree normalised impurity importances, renormalised (DESIGN.md §7).
        imp = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        total = imp.sum()
        self.feature_importances_ = imp / total if total > 0 else imp

        if self.oob_score:
            self._aggregate_oob(results, y, n)
        return self

    def _aggregate_oob(
        self,
        results: list[tuple[BaseDecisionTree, np.ndarray | None, np.ndarray | None]],
        y: np.ndarray,
        n: int,
    ) -> None:
        """Average each row's OOB leaf values over the trees that did not see it."""
        width = results[0][0].tree_.value.shape[1]
        total = np.zeros((n, width))
        count = np.zeros(n)
        # Loop over trees (<= a few hundred), each update vectorised over its OOB rows.
        for _, idx, vals in results:
            total[idx] += vals
            count[idx] += 1
        covered = count > 0
        if not covered.all():
            warnings.warn(
                f"{int((~covered).sum())} of {n} training rows were never out-of-bag; their OOB "
                "prediction is NaN and they are excluded from oob_score_. "
                "Use more estimators.",
                UserWarning,
                stacklevel=4,
            )
        mean = np.full((n, width), np.nan)
        mean[covered] = total[covered] / count[covered, None]
        self._set_oob_outputs(mean, covered, y)

    # ------------------------------------------------------------------ inference
    def _check_fitted(self) -> None:
        if not hasattr(self, "estimators_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")

    def _mean_leaf_values(self, X: Any) -> np.ndarray:
        """Mean over trees of the reached leaf's ``value``, shape ``(n, K or 1)``."""
        self._check_fitted()
        X = _check_X(X, self.n_features_in_)
        total = 0.0
        for tree in self.estimators_:  # loop over trees; each apply() is vectorised over rows
            total = total + tree.tree_.value[tree.tree_.apply(X)]
        return total / len(self.estimators_)

    def oob_indices(self, i: int) -> np.ndarray:
        """Training rows that tree ``i`` never saw, recomputed from its seed. Shape ``(m,)``."""
        self._check_fitted()
        if not self.bootstrap:
            raise ValueError("oob_indices requires bootstrap=True")
        if not 0 <= i < len(self.estimators_):
            raise IndexError(f"tree index {i} out of range for {len(self.estimators_)} trees")
        boot = _bootstrap_indices(self._boot_seeds[i], self.n_samples_fit_)
        return _oob_rows(boot, self.n_samples_fit_)

    def staged_scores(
        self, X: Any, y: Any, n_trees_grid: Sequence[int], oob: bool = False
    ) -> np.ndarray:
        """Score of the sub-forest made of the first ``k`` trees, for every ``k`` in the grid.

        One pass over the trees, so a whole "score vs number of trees" curve costs about as
        much as a single prediction. With ``oob=True``, ``X``/``y`` must be the training data and
        each tree only votes on its own out-of-bag rows; rows with no vote yet are excluded
        (early points of the curve cover fewer rows).

        Parameters
        ----------
        X : array_like, shape (n, d)
        y : array_like, shape (n,)
        n_trees_grid : sequence of int
            Values of ``k`` in ``[1, n_estimators]`` (any order).
        oob : bool

        Returns
        -------
        ndarray, shape (len(grid),)
            Accuracy (classifier) or R² (regressor), in the order of ``n_trees_grid``.
        """
        self._check_fitted()
        X = _check_X(X, self.n_features_in_)
        y = np.asarray(y)
        if y.shape != (X.shape[0],):
            raise ValueError(f"y must have shape ({X.shape[0]},), got {y.shape}")
        grid = [int(k) for k in n_trees_grid]
        if not grid or min(grid) < 1 or max(grid) > len(self.estimators_):
            raise ValueError(f"n_trees_grid must lie in [1, {len(self.estimators_)}], got {grid}")
        if oob and X.shape[0] != self.n_samples_fit_:
            raise ValueError("oob=True needs the training data (same number of rows as in fit)")
        wanted = set(grid)
        scores: dict[int, float] = {}
        width = self.estimators_[0].tree_.value.shape[1]
        total = np.zeros((X.shape[0], width))
        count = np.zeros(X.shape[0])
        for k, tree in enumerate(self.estimators_[: max(grid)], start=1):
            if oob:
                idx = self.oob_indices(k - 1)
                total[idx] += tree.tree_.value[tree.tree_.apply(X[idx])]
                count[idx] += 1
            else:
                total += tree.tree_.value[tree.tree_.apply(X)]
                count += 1
            if k in wanted:
                covered = count > 0
                scores[k] = self._score_from_mean(total[covered] / count[covered, None], y[covered])
        return np.array([scores[k] for k in grid])

    # hooks implemented by the two subclasses ---------------------------------------------
    def _set_oob_outputs(self, mean: np.ndarray, covered: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def _score_from_mean(self, mean_values: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError


class RandomForestClassifier(BaseForest):
    """Random forest classifier (Gini or entropy trees, ``max_features="sqrt"`` by default).

    Attributes (after ``fit``)
    --------------------------
    estimators_ : list of fitted trees
    classes_ : ndarray (K,)
    feature_importances_ : ndarray (d,)
        Mean impurity importance over trees, sums to 1.
    oob_score_ : float
        OOB accuracy (if ``oob_score=True``).
    oob_decision_function_ : ndarray (n, K)
        OOB class probabilities; NaN for rows that were never out-of-bag.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]] * 5); y = np.array([0, 0, 1, 1] * 5)
    >>> RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y).predict(X[:4]).tolist()
    [0, 0, 1, 1]
    """

    _is_classifier = True
    _tree_cls = DecisionTreeClassifier
    _default_max_features = "sqrt"

    def fit(self, X: Any, y: Any) -> RandomForestClassifier:
        """Grow the forest on features ``X (n, d)`` and labels ``y (n,)``; returns ``self``."""
        self._validate_params()
        X = _check_X(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
        if X.shape[0] < 1 or X.shape[1] < 1:
            raise ValueError("X must have at least one row and one column")
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = int(self.classes_.shape[0])
        return self._fit(X, y_enc.ravel(), self.n_classes_)  # type: ignore[return-value]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Mean of the trees' leaf class proportions, shape ``(n, K)``; rows sum to 1."""
        return self._mean_leaf_values(X)

    def predict(self, X: Any) -> np.ndarray:
        """Labels of the highest mean probability, shape ``(n,)`` (lowest class on ties)."""
        mean = self._mean_leaf_values(X)  # checks fitted-ness before classes_ is touched
        return self.classes_[np.argmax(mean, axis=1)]

    def score(self, X: Any, y: Any) -> float:
        """Accuracy of ``predict(X)``."""
        return accuracy(np.asarray(y), self.predict(X))

    def _set_oob_outputs(self, mean: np.ndarray, covered: np.ndarray, y: np.ndarray) -> None:
        self.oob_decision_function_ = mean
        pred = self.classes_[np.argmax(mean[covered], axis=1)]
        self.oob_score_ = accuracy(self.classes_[y[covered]], pred)

    def _score_from_mean(self, mean_values: np.ndarray, y: np.ndarray) -> float:
        return accuracy(y, self.classes_[np.argmax(mean_values, axis=1)])


class RandomForestRegressor(BaseForest):
    """Random forest regressor (squared-error trees, all features per node by default).

    Attributes (after ``fit``)
    --------------------------
    estimators_ : list of fitted trees
    feature_importances_ : ndarray (d,)
    oob_score_ : float
        OOB R² (if ``oob_score=True``).
    oob_prediction_ : ndarray (n,)
        OOB mean prediction; NaN for rows that were never out-of-bag.
    """

    _tree_cls = DecisionTreeRegressor
    _default_max_features = 1.0

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: None | int | float | str = None,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int | None = None,
        random_state: int | np.random.SeedSequence | None = None,
    ) -> None:
        super().__init__(
            n_estimators,
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            max_features,
            bootstrap,
            oob_score,
            n_jobs,
            random_state,
        )

    def fit(self, X: Any, y: Any) -> RandomForestRegressor:
        """Grow the forest on features ``X (n, d)`` and finite float targets ``y (n,)``."""
        self._validate_params()
        X = _check_X(X)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
        if X.shape[0] < 1 or X.shape[1] < 1:
            raise ValueError("X must have at least one row and one column")
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or infinity")
        return self._fit(X, y, None)  # type: ignore[return-value]

    def predict(self, X: Any) -> np.ndarray:
        """Mean over trees of the leaf means, shape ``(n,)``."""
        return self._mean_leaf_values(X)[:, 0]

    def score(self, X: Any, y: Any) -> float:
        """R² of ``predict(X)``."""
        return r2_score(np.asarray(y, dtype=np.float64), self.predict(X))

    def _set_oob_outputs(self, mean: np.ndarray, covered: np.ndarray, y: np.ndarray) -> None:
        self.oob_prediction_ = mean[:, 0]
        self.oob_score_ = r2_score(y[covered], mean[covered, 0])

    def _score_from_mean(self, mean_values: np.ndarray, y: np.ndarray) -> float:
        return r2_score(y.astype(np.float64), mean_values[:, 0])


def forest_from_config(
    task: str, forest: ForestConfig, tree: TreeConfig, seed: int, **overrides: Any
) -> RandomForestClassifier | RandomForestRegressor:
    """Build an unfitted forest from the ``forest`` and ``tree`` config sections.

    Depth / leaf-size / impurity limits come from ``tree``; ``max_features`` comes from
    ``forest`` (``None`` = the task default, deliberately *not* ``tree.max_features``, which is
    ``None`` = all features for a single tree).

    Parameters
    ----------
    task : {"classification", "regression"}
    forest, tree : ForestConfig, TreeConfig
    seed : int
        Root ``random_state``.
    **overrides
        Constructor parameters that replace the config's (e.g. ``n_estimators=200``).
    """
    params: dict[str, Any] = {
        "n_estimators": forest.n_estimators,
        "max_depth": tree.max_depth,
        "min_samples_split": tree.min_samples_split,
        "min_samples_leaf": tree.min_samples_leaf,
        "min_impurity_decrease": tree.min_impurity_decrease,
        "max_features": forest.max_features,
        "bootstrap": forest.bootstrap,
        "oob_score": forest.oob_score,
        "n_jobs": forest.n_jobs,
        "random_state": seed,
    }
    params.update(overrides)
    if task == "classification":
        return RandomForestClassifier(criterion=tree.criterion_classification, **params)
    if task == "regression":
        return RandomForestRegressor(criterion=tree.criterion_regression, **params)
    raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
