"""Best-split search: the O(n)-after-sort scan (see DESIGN.md §3–§4).

For one feature, sort the node's rows by that feature once; every candidate threshold is then a
*prefix* of the sorted order, its class counts / target sums are prefix sums, and the children
cost of **all** ``m - 1`` thresholds is evaluated in one vectorized shot. No Python loop over
samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from treesfs.criteria import (
    CRITERIA,
    FEATURE_THRESHOLD,
    Criterion,
    entropy,
    get_criterion,
    gini,
    squared_error,
)


@dataclass(frozen=True)
class Split:
    """A chosen split: rows go left iff ``x[:, feature] <= threshold``."""

    feature: int
    threshold: float
    cost: float  # N_L I(L) + N_R I(R); lower is better (DESIGN.md §3.1)
    n_left: int
    n_right: int


def _scan_feature(
    x: np.ndarray,
    stats: np.ndarray,
    criterion: Criterion,
    min_samples_leaf: int,
    total_ss: float,
) -> tuple[bool, tuple[float, float, int] | None]:
    """Scan one feature; returns ``(is_constant, best)``.

    ``is_constant`` lets :func:`find_best_split` tell a constant feature (does not count toward
    ``max_features``) from a non-constant one that merely has no split satisfying
    ``min_samples_leaf`` (does count).
    """
    m = x.shape[0]
    order = np.argsort(x, kind="stable")  # stable: bit-reproducible across NumPy versions
    xs = x[order]
    if xs[-1] <= xs[0] + FEATURE_THRESHOLD:
        return True, None
    n_left = np.arange(1, m, dtype=np.float64)
    n_right = m - n_left
    valid = (
        (xs[1:] > xs[:-1] + FEATURE_THRESHOLD)
        & (n_left >= min_samples_leaf)
        & (n_right >= min_samples_leaf)
    )
    if not valid.any():
        return False, None
    cost = criterion.children_cost(np.cumsum(stats[order], axis=0), n_left, n_right, total_ss)
    cost = np.where(valid, cost, np.inf)
    i = int(np.argmin(cost))  # first minimum -> smallest threshold wins ties (DESIGN.md §3.3)
    tau = 0.5 * (xs[i] + xs[i + 1])
    if tau == xs[i + 1] or not np.isfinite(tau):  # adjacent floats / overflow guard (§3.4)
        tau = xs[i]
    return False, (float(cost[i]), float(tau), i + 1)


def _total_ss(stats: np.ndarray, criterion: Criterion) -> float:
    """``Q_t = sum(ys**2)`` for regression (centered targets); unused (0) for classification."""
    return 0.0 if criterion.is_classification else float(np.dot(stats, stats))


def best_split_one_feature(
    x: np.ndarray, stats: np.ndarray, criterion: Criterion, min_samples_leaf: int
) -> tuple[float, float, int] | None:
    """Best threshold on a single feature (DESIGN.md §3.2–§3.5).

    Sorts ``x`` once, takes prefix sums of ``stats`` in that order and evaluates
    ``cost(i) = N_L I(L) + N_R I(R)`` for all split positions ``i = 0..m-2`` simultaneously.
    A position is valid iff ``xs[i+1] > xs[i] + 1e-7`` and both sides keep at least
    ``min_samples_leaf`` rows. Ties go to the smallest threshold. The threshold is the midpoint
    of the two neighbouring sorted values (falling back to the lower one if rounding would
    make it equal to the upper).

    Parameters
    ----------
    x : ndarray, shape (m,)
        Feature values of the node's rows.
    stats : ndarray, shape (m, K) or (m,)
        One-hot labels (classification) or node-mean-**centered** targets (regression;
        centering removes cancellation, DESIGN.md §3.4), in the same row order as ``x``.
    criterion : Criterion
        Strategy from :data:`treesfs.criteria.CRITERIA`.
    min_samples_leaf : int
        Minimum rows on each side.

    Returns
    -------
    (cost, threshold, n_left) or None
        ``None`` if the feature is constant or no position is valid.
    """
    return _scan_feature(x, stats, criterion, min_samples_leaf, _total_ss(stats, criterion))[1]


def _canonical(criterion_name: str) -> str:
    """Normalize criterion aliases (``"mse"`` -> ``"squared_error"``) and validate."""
    get_criterion(criterion_name)
    return "squared_error" if criterion_name == "mse" else criterion_name


def partition_cost(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    criterion_name: str,
    n_classes: int | None = None,
) -> float:
    """``N_L I(L) + N_R I(R)`` obtained by really partitioning ``x <= threshold`` (test oracle).

    Computes impurities directly from the raw labels/targets with the pure functions in
    :mod:`treesfs.criteria`, independent of the prefix-sum formulas. Returns ``inf`` if one side
    is empty.

    Parameters
    ----------
    x : ndarray, shape (m,)
    y : ndarray, shape (m,)
        Integer class codes (classification) or raw float targets (regression).
    threshold : float
    criterion_name : str
    n_classes : int, optional
        Number of classes; defaults to ``y.max() + 1``.
    """
    name = _canonical(criterion_name)
    left = x <= threshold
    n_l, n_r = int(left.sum()), int((~left).sum())
    if n_l == 0 or n_r == 0:
        return math.inf
    parts = (y[left], y[~left])
    if name == "squared_error":
        return n_l * squared_error(parts[0]) + n_r * squared_error(parts[1])
    k = int(y.max()) + 1 if n_classes is None else n_classes
    impurity = gini if name == "gini" else entropy
    return sum(len(part) * float(impurity(np.bincount(part, minlength=k))) for part in parts)


def best_split_brute_force(
    x: np.ndarray,
    y: np.ndarray,
    criterion_name: str,
    min_samples_leaf: int,
    n_classes: int | None = None,
) -> tuple[float, float] | None:
    """O(m^2) reference: try every midpoint between distinct values and repartition.

    Used ONLY to validate :func:`best_split_one_feature` (DESIGN.md §9.1) and in notebooks, never
    in the tree. The loop over thresholds is deliberate: the point is an independent,
    obviously-correct implementation that shares no prefix-sum code with the fast path.

    Parameters
    ----------
    x : ndarray, shape (m,)
    y : ndarray, shape (m,)
        Integer class codes (classification) or raw float targets (regression).
    criterion_name : str
        ``"gini"``, ``"entropy"`` or ``"squared_error"`` (alias ``"mse"``).
    min_samples_leaf : int
    n_classes : int, optional

    Returns
    -------
    (cost, threshold) or None
        Best children cost and one threshold achieving it; ``None`` if no valid split.
    """
    name = _canonical(criterion_name)
    u = np.unique(x)
    best: tuple[float, float] | None = None
    for a, b in zip(u[:-1], u[1:], strict=True):  # loop over thresholds, not samples
        if not b > a + FEATURE_THRESHOLD:
            continue
        tau = 0.5 * (a + b)
        if tau == b or not np.isfinite(tau):
            tau = a
        n_l = int((x <= tau).sum())
        if n_l < min_samples_leaf or len(x) - n_l < min_samples_leaf:
            continue
        cost = partition_cost(x, y, tau, name, n_classes)
        if best is None or cost < best[0]:
            best = (cost, float(tau))
    return best


def resolve_max_features(max_features: None | int | float | str, n_features: int) -> int:
    """Turn a ``max_features`` setting into an integer ``1 <= F <= d`` (DESIGN.md §4).

    ========================  ===========================
    ``max_features``          ``F``
    ========================  ===========================
    ``None``                  ``d``
    ``"sqrt"``                ``max(1, int(sqrt(d)))``
    ``"log2"``                ``max(1, int(log2(d)))``
    ``int``                   that value (``1 <= F <= d``)
    ``float`` in ``(0, 1]``   ``max(1, int(F * d))``
    ========================  ===========================

    Raises
    ------
    ValueError
        For unknown strings, out-of-range numbers, or other types (including ``bool``).
    """
    d = n_features
    if max_features is None:
        return d
    if isinstance(max_features, bool):
        raise ValueError("max_features must not be a bool")
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(math.sqrt(d)))
        if max_features == "log2":
            return max(1, int(math.log2(d)))
        raise ValueError(
            f"max_features must be 'sqrt' or 'log2' when a string, got {max_features!r}"
        )
    if isinstance(max_features, (int, np.integer)):
        if not 1 <= max_features <= d:
            raise ValueError(f"max_features={max_features} must be in [1, {d}]")
        return int(max_features)
    if isinstance(max_features, (float, np.floating)):
        if not 0.0 < max_features <= 1.0:
            raise ValueError(f"float max_features must be in (0, 1], got {max_features}")
        return max(1, int(max_features * d))
    raise ValueError(f"Invalid max_features: {max_features!r}")


def find_best_split(
    X_node: np.ndarray,
    stats: np.ndarray,
    criterion: Criterion,
    min_samples_leaf: int,
    n_features_to_try: int,
    rng: np.random.Generator | None,
) -> Split | None:
    """Best split over (a random subset of) the features of one node (DESIGN.md §4).

    With ``n_features_to_try == d`` (``rng`` may be ``None``) features are scanned in the order
    ``0..d-1``, so ties break to the lowest feature index and the tree is fully deterministic.
    Otherwise a fresh ``rng.permutation(d)`` fixes the visiting order and the scan stops once at
    least ``n_features_to_try`` *non-constant* features were evaluated **and** a valid split was
    found (if the first F give none, it keeps going rather than make a premature leaf).

    The loop is over features (a few dozen), never over samples.

    Parameters
    ----------
    X_node : ndarray, shape (m, d)
    stats : ndarray, shape (m, K) or (m,)
        Built once per node: one-hot labels or node-centered targets.
    criterion : Criterion
    min_samples_leaf : int
    n_features_to_try : int
        ``F`` from :func:`resolve_max_features`.
    rng : numpy.random.Generator or None
        Required when ``n_features_to_try < d``.

    Returns
    -------
    Split or None
    """
    d = X_node.shape[1]
    if n_features_to_try >= d:
        order = range(d)
    else:
        if rng is None:
            raise ValueError("rng is required when n_features_to_try < n_features")
        order = rng.permutation(d)
    total_ss = _total_ss(stats, criterion)
    best: Split | None = None
    n_visited = 0
    for j in order:  # loop over features, not samples
        if n_visited >= n_features_to_try and best is not None:
            break
        is_constant, res = _scan_feature(X_node[:, j], stats, criterion, min_samples_leaf, total_ss)
        if is_constant:
            continue
        n_visited += 1
        if res is not None and (best is None or res[0] < best.cost):  # strict <: first wins ties
            m = X_node.shape[0]
            best = Split(int(j), res[1], res[0], res[2], m - res[2])
    return best


__all__ = [
    "CRITERIA",
    "Split",
    "best_split_brute_force",
    "best_split_one_feature",
    "find_best_split",
    "partition_cost",
    "resolve_max_features",
]
