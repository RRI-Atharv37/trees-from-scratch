"""CART decision trees on flat arrays (see DESIGN.md §5–§7). NumPy only.

* :class:`TreeArrays`: the fitted tree as 7 parallel arrays (sklearn's ``tree_`` layout) with a
  vectorized :meth:`TreeArrays.apply`.
* :class:`BaseDecisionTree`: the shared depth-first growth loop with an explicit stack.
* :class:`DecisionTreeClassifier` / :class:`DecisionTreeRegressor`: thin sklearn-style wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from treesfs.criteria import EPS, Criterion, get_criterion
from treesfs.metrics import accuracy, r2_score
from treesfs.splitter import find_best_split, resolve_max_features

if TYPE_CHECKING:  # pragma: no cover
    from treesfs.config import TreeConfig

TREE_LEAF = -1  # children_left / children_right value of a leaf
TREE_UNDEFINED = -2  # feature value of a leaf (sklearn's TREE_UNDEFINED)


@dataclass(frozen=True)
class TreeArrays:
    """A fitted tree stored as parallel arrays of length ``node_count`` (DESIGN.md §6).

    Node ids are assigned in **preorder** (root 0, then the whole left subtree), the same order
    as scikit-learn's depth-first builder.

    Attributes
    ----------
    children_left, children_right : ndarray (n_nodes,) int64
        Child ids, ``-1`` for a leaf.
    feature : ndarray (n_nodes,) int64
        Split feature, ``-2`` for a leaf.
    threshold : ndarray (n_nodes,) float64
        Rows go left iff ``x[feature] <= threshold``; ``-2.0`` for a leaf.
    value : ndarray (n_nodes, K) or (n_nodes, 1)
        Class proportions (classifier) or the mean target (regressor); stored for internal
        nodes too (used by truncated traversal and by the exporters).
    n_node_samples : ndarray (n_nodes,) int64
    impurity : ndarray (n_nodes,) float64
    """

    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray
    n_node_samples: np.ndarray
    impurity: np.ndarray

    @property
    def node_count(self) -> int:
        """Total number of nodes (internal + leaves)."""
        return int(self.children_left.shape[0])

    def node_depths(self) -> np.ndarray:
        """Depth of every node (root = 0), shape ``(n_nodes,)``.

        One pass in preorder: parents always precede their children, so
        ``depth[child] = depth[parent] + 1`` is final when it is assigned. The loop is over
        nodes, not samples.
        """
        depths = np.zeros(self.node_count, dtype=np.int64)
        for t in range(self.node_count):
            left = self.children_left[t]
            if left != TREE_LEAF:
                depths[left] = depths[t] + 1
                depths[self.children_right[t]] = depths[t] + 1
        return depths

    def depth(self) -> int:
        """Depth of the deepest leaf (a single-leaf tree has depth 0)."""
        return int(self.node_depths().max())

    def n_leaves(self) -> int:
        """Number of leaves."""
        return int((self.children_left == TREE_LEAF).sum())

    def apply(self, X: np.ndarray, max_depth: int | None = None) -> np.ndarray:
        """Node id reached by each row (DESIGN.md §6.1).

        All rows advance one level per iteration: ``go_left = X[i, feature[t]] <= threshold[t]``.
        The loop runs over *depth levels*, never over samples, so cost is
        ``O(n * depth)`` inside NumPy.

        Parameters
        ----------
        X : ndarray, shape (n, d)
        max_depth : int, optional
            Stop descending after this many levels and return the internal node reached
            (DESIGN.md §11). With a deterministic tree, ``value[apply(X, k)]`` equals the
            predictions of a tree freshly grown with ``max_depth=k``.

        Returns
        -------
        ndarray, shape (n,) of intp
        """
        n = X.shape[0]
        node = np.zeros(n, dtype=np.intp)
        active = np.flatnonzero(self.children_left[node] != TREE_LEAF)
        level = 0
        while active.size and (max_depth is None or level < max_depth):
            nd = node[active]
            go_left = X[active, self.feature[nd]] <= self.threshold[nd]
            node[active] = np.where(go_left, self.children_left[nd], self.children_right[nd])
            active = active[self.children_left[node[active]] != TREE_LEAF]
            level += 1
        return node


def _check_X(X: Any, n_features: int | None = None) -> np.ndarray:
    """Validate a feature matrix: 2-D, finite, float64, optionally ``n_features`` columns."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features), got shape {X.shape}")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} features, but the model was fitted with {n_features}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinity")
    return X


class BaseDecisionTree:
    """Shared CART growth loop (DESIGN.md §5.1); use the classifier/regressor subclasses.

    Parameters
    ----------
    criterion : str
        Impurity criterion name (see :data:`treesfs.criteria.CRITERIA`).
    max_depth : int or None
        Maximum depth. ``None`` = unlimited. ``0`` is allowed and gives a single leaf (a
        majority-class / mean baseline; scikit-learn requires >= 1).
    min_samples_split : int
        A node with fewer rows is a leaf. Must be >= 2.
    min_samples_leaf : int
        Every leaf keeps at least this many rows. Must be >= 1.
    min_impurity_decrease : float
        A split must reduce the *whole-dataset* weighted impurity by at least this much
        (DESIGN.md §2.1).
    max_features : None, int, float or {"sqrt", "log2"}
        Features examined per node (DESIGN.md §4).
    random_state : int, SeedSequence, Generator or None
        Seeds feature subsampling; irrelevant when every feature is tried.
    """

    _is_classifier: bool = False
    _valid_criteria: tuple[str, ...] = ()

    def __init__(
        self,
        criterion: str,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: None | int | float | str = None,
        random_state: Any = None,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state

    # ------------------------------------------------------------------ params
    def get_params(self) -> dict[str, Any]:
        """Constructor parameters as a dict (handy for cloning inside a forest)."""
        return {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
            "max_features": self.max_features,
            "random_state": self.random_state,
        }

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"

    def _validate_params(self) -> None:
        """Range-check hyper-parameters; raises ``ValueError``."""
        if self.criterion not in self._valid_criteria:
            raise ValueError(
                f"criterion={self.criterion!r} is invalid for {type(self).__name__}; "
                f"choose from {list(self._valid_criteria)}"
            )
        if self.max_depth is not None and (
            isinstance(self.max_depth, bool)
            or int(self.max_depth) != self.max_depth
            or self.max_depth < 0
        ):
            raise ValueError(f"max_depth must be None or an int >= 0, got {self.max_depth!r}")
        if int(self.min_samples_split) != self.min_samples_split or self.min_samples_split < 2:
            raise ValueError(f"min_samples_split must be an int >= 2, got {self.min_samples_split}")
        if int(self.min_samples_leaf) != self.min_samples_leaf or self.min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be an int >= 1, got {self.min_samples_leaf}")
        if not self.min_impurity_decrease >= 0:
            raise ValueError(
                f"min_impurity_decrease must be >= 0, got {self.min_impurity_decrease}"
            )

    # ------------------------------------------------------------------ growth
    def _fit(self, X: np.ndarray, y: np.ndarray, n_classes: int | None) -> BaseDecisionTree:
        """Grow the tree on validated data (the hook a random forest calls, DESIGN.md §5.3/§11).

        Parameters
        ----------
        X : ndarray, shape (n, d) float64
        y : ndarray, shape (n,)
            Integer class codes in ``[0, n_classes)`` (classifier) or float targets
            (regressor).
        n_classes : int or None
            ``K`` for the classifier (a forest passes its own ``K``, so every tree's ``value`` is
            aligned even if a bootstrap sample misses a class); ``None`` for the regressor.

        Notes
        -----
        Explicit-stack depth-first growth. The RIGHT child is pushed first so the LEFT child is
        popped first, giving preorder node ids. The loop is over *nodes*; inside a node all work
        is vectorized (DESIGN.md §3). ``X[idx]`` is a fancy-index copy per node: a known cost
        versus scikit-learn's in-place partition (DESIGN.md §8).
        """
        self._validate_params()
        n, d = X.shape
        crit: Criterion = get_criterion(self.criterion)
        n_try = resolve_max_features(self.max_features, d)
        rng = np.random.default_rng(self.random_state) if n_try < d else None
        classify = crit.is_classification
        if classify:
            assert n_classes is not None
            one_hot = np.eye(n_classes, dtype=np.float64)[y]  # (n, K), built once per fit
        y_float = None if classify else np.asarray(y, dtype=np.float64)

        left: list[int] = []
        right: list[int] = []
        feat: list[int] = []
        thr: list[float] = []
        values: list[np.ndarray] = []
        n_samples: list[int] = []
        impurity: list[float] = []
        importances = np.zeros(d, dtype=np.float64)

        # (row indices, depth, parent id, is_left_child)
        stack: list[tuple[np.ndarray, int, int, bool]] = [(np.arange(n), 0, -1, False)]
        while stack:  # loop over nodes
            idx, depth, parent, is_left = stack.pop()
            node_id = len(feat)
            if parent >= 0:
                (left if is_left else right)[parent] = node_id
            m = idx.size

            # Per-node statistics, built once and shared by all features (DESIGN.md §5.1).
            if classify:
                stats = one_hot[idx]  # (m, K)
                value = stats.sum(axis=0) / m  # class proportions
            else:
                y_node = y_float[idx]
                mean = y_node.mean()
                stats = y_node - mean  # (m,) centered, DESIGN.md §3.4
                value = np.array([mean])
            node_imp = crit.node_impurity(stats)

            left.append(TREE_LEAF)
            right.append(TREE_LEAF)
            feat.append(TREE_UNDEFINED)
            thr.append(float(TREE_UNDEFINED))
            values.append(value)
            n_samples.append(m)
            impurity.append(node_imp)

            is_leaf = (
                (self.max_depth is not None and depth >= self.max_depth)
                or m < self.min_samples_split
                or m < 2 * self.min_samples_leaf
                or node_imp <= EPS  # pure node; never compare to 0 exactly (DESIGN.md §5.1)
            )
            if is_leaf:
                continue
            X_node = X[idx]
            split = find_best_split(X_node, stats, crit, self.min_samples_leaf, n_try, rng)
            if split is None:
                continue
            gain = m * node_imp - split.cost  # N_t I(t) - N_L I(L) - N_R I(R)
            if gain / n + EPS < self.min_impurity_decrease:  # sklearn's tolerance
                continue

            feat[node_id] = split.feature
            thr[node_id] = split.threshold
            importances[split.feature] += max(gain, 0.0)
            go_left = X_node[:, split.feature] <= split.threshold
            stack.append((idx[~go_left], depth + 1, node_id, False))
            stack.append((idx[go_left], depth + 1, node_id, True))

        self.tree_ = TreeArrays(
            children_left=np.asarray(left, dtype=np.int64),
            children_right=np.asarray(right, dtype=np.int64),
            feature=np.asarray(feat, dtype=np.int64),
            threshold=np.asarray(thr, dtype=np.float64),
            value=np.vstack(values),
            n_node_samples=np.asarray(n_samples, dtype=np.int64),
            impurity=np.asarray(impurity, dtype=np.float64),
        )
        self.n_features_in_ = d
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances  # §7
        return self

    # ------------------------------------------------------------------ inference
    def _check_fitted(self) -> None:
        if not hasattr(self, "tree_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")

    def _leaf_values(self, X: Any) -> np.ndarray:
        """``value`` of the leaf reached by each row, shape ``(n, K or 1)``."""
        self._check_fitted()
        X = _check_X(X, self.n_features_in_)
        return self.tree_.value[self.tree_.apply(X)]

    def apply(self, X: Any) -> np.ndarray:
        """Leaf id reached by each row, shape ``(n,)``."""
        self._check_fitted()
        return self.tree_.apply(_check_X(X, self.n_features_in_))

    def get_depth(self) -> int:
        """Depth of the fitted tree (single leaf = 0)."""
        self._check_fitted()
        return self.tree_.depth()

    def get_n_leaves(self) -> int:
        """Number of leaves of the fitted tree."""
        self._check_fitted()
        return self.tree_.n_leaves()


class DecisionTreeClassifier(BaseDecisionTree):
    """CART classifier with Gini or entropy impurity.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0], [2.0], [3.0], [4.0]]); y = np.array([0, 0, 1, 1])
    >>> DecisionTreeClassifier().fit(X, y).predict(X).tolist()
    [0, 0, 1, 1]
    """

    _is_classifier = True
    _valid_criteria = ("gini", "entropy")

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: None | int | float | str = None,
        random_state: Any = None,
    ) -> None:
        super().__init__(
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            max_features,
            random_state,
        )

    def fit(self, X: Any, y: Any) -> DecisionTreeClassifier:
        """Grow the tree.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Finite numeric features.
        y : array_like, shape (n,)
            Labels of any sortable type; encoded with ``np.unique`` into ``classes_``.

        Returns
        -------
        self
        """
        self._validate_params()
        X = _check_X(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
        if X.shape[0] < 1 or X.shape[1] < 1:
            raise ValueError("X must have at least one row and one column")
        classes, y_enc = np.unique(y, return_inverse=True)
        self.classes_ = classes
        self.n_classes_ = int(classes.shape[0])
        return self._fit(X, y_enc.ravel(), self.n_classes_)  # ravel: NumPy 2.x inverse shape

    def predict_proba(self, X: Any) -> np.ndarray:
        """Class proportions of the reached leaf, shape ``(n, K)``; columns follow ``classes_``."""
        return self._leaf_values(X).copy()

    def predict(self, X: Any) -> np.ndarray:
        """Predicted labels, shape ``(n,)``: argmax of leaf proportions (lowest index on ties)."""
        leaf_values = self._leaf_values(X)  # also checks fitted-ness before classes_ is used
        return self.classes_[np.argmax(leaf_values, axis=1)]

    def score(self, X: Any, y: Any) -> float:
        """Accuracy of ``predict(X)`` against ``y``."""
        return accuracy(np.asarray(y), self.predict(X))


class DecisionTreeRegressor(BaseDecisionTree):
    """CART regressor with squared-error impurity (leaf value = mean target).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0], [2.0], [3.0], [4.0]]); y = np.array([1.0, 1.0, 5.0, 5.0])
    >>> DecisionTreeRegressor().fit(X, y).predict(X).tolist()
    [1.0, 1.0, 5.0, 5.0]
    """

    _valid_criteria = ("squared_error", "mse")

    def __init__(
        self,
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: None | int | float | str = None,
        random_state: Any = None,
    ) -> None:
        super().__init__(
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            max_features,
            random_state,
        )

    def fit(self, X: Any, y: Any) -> DecisionTreeRegressor:
        """Grow the tree on features ``X (n, d)`` and finite float targets ``y (n,)``."""
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
        return self._fit(X, y, None)

    def predict(self, X: Any) -> np.ndarray:
        """Mean target of the reached leaf, shape ``(n,)``."""
        return self._leaf_values(X)[:, 0]

    def score(self, X: Any, y: Any) -> float:
        """Coefficient of determination R² of ``predict(X)``."""
        return r2_score(np.asarray(y, dtype=np.float64), self.predict(X))


def tree_from_config(
    task: str, cfg: TreeConfig, seed: int, **overrides: Any
) -> DecisionTreeClassifier | DecisionTreeRegressor:
    """Build an unfitted tree from a :class:`~treesfs.config.TreeConfig`.

    Parameters
    ----------
    task : {"classification", "regression"}
    cfg : TreeConfig
    seed : int
        Used as ``random_state``.
    **overrides
        Constructor parameters that replace the config's (e.g. ``max_depth=None`` for a sweep).
    """
    params: dict[str, Any] = {
        "max_depth": cfg.max_depth,
        "min_samples_split": cfg.min_samples_split,
        "min_samples_leaf": cfg.min_samples_leaf,
        "min_impurity_decrease": cfg.min_impurity_decrease,
        "max_features": cfg.max_features,
        "random_state": seed,
    }
    params.update(overrides)
    if task == "classification":
        return DecisionTreeClassifier(criterion=cfg.criterion_classification, **params)
    if task == "regression":
        return DecisionTreeRegressor(criterion=cfg.criterion_regression, **params)
    raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
