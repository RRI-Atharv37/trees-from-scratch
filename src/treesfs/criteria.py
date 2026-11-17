"""Impurity criteria: Gini, entropy and squared error (see DESIGN.md §1–§3.2).

Two layers live here:

* **Pure functions** (:func:`gini`, :func:`entropy`, :func:`squared_error`,
  :func:`impurity_decrease`) that take sufficient statistics (class counts or raw targets) and
  return the impurity. They are the readable definitions and the oracle used by the tests.
* A **strategy registry** (:data:`CRITERIA`) whose members evaluate the splitter's
  *children cost* ``N_L I(L) + N_R I(R)`` for all ``m - 1`` candidate thresholds of a sorted
  column at once, from prefix sums (DESIGN.md §3.2).

Regression convention (answers the ``Q_t`` question): ``stats`` handed to the criterion are the
node's targets **centered on the node mean**, ``ys = y - mean(y)``. Then

    N_t * V(t) = Q_t = sum(ys ** 2)

so the squared error's children cost ``Q_t - s_L^2/n_L - s_R^2/n_R`` needs one scalar ``Q_t``
per node. The splitter computes it once from ``stats`` and passes it to
:meth:`Criterion.children_cost` as ``total_ss``; classification criteria ignore it.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

#: Machine epsilon for float64; a node whose impurity is <= EPS is treated as pure.
EPS: float = float(np.finfo(np.float64).eps)

#: Minimum gap between adjacent sorted feature values for a threshold to exist between them.
FEATURE_THRESHOLD: float = 1e-7


def _scalar_if_0d(a: np.ndarray) -> np.ndarray | float:
    """Return a Python float for 0-d results, else the array unchanged."""
    return float(a) if a.ndim == 0 else a


def gini(counts: ArrayLike) -> np.ndarray | float:
    """Gini impurity from class counts.

    .. math:: G = \\sum_k p_k (1 - p_k) = 1 - \\sum_k p_k^2, \\qquad p_k = c_k / \\sum_j c_j

    The probability that two labels drawn with replacement from the node disagree. It lies in
    ``[0, 1 - 1/K]`` and is 0 iff the node is pure (DESIGN.md §1.1). An empty node gives 0.

    Parameters
    ----------
    counts : array_like, shape (..., K)
        Class counts (non-negative). Any leading batch axes are preserved.

    Returns
    -------
    ndarray of shape (...) or float
        Impurity per row of counts.
    """
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum(axis=-1)
    safe_n = np.where(n > 0, n, 1.0)
    out = np.where(n > 0, 1.0 - np.einsum("...k,...k->...", c, c) / (safe_n * safe_n), 0.0)
    return _scalar_if_0d(out)


def _xlog2x(c: np.ndarray) -> np.ndarray:
    """Elementwise ``c * log2(c)`` with the convention ``0 log 0 = 0`` (masked, no NaN)."""
    pos = c > 0
    return np.where(pos, c * np.log2(np.where(pos, c, 1.0)), 0.0)


def entropy(counts: ArrayLike) -> np.ndarray | float:
    """Shannon entropy in **bits** from class counts.

    .. math:: H = -\\sum_{k: p_k > 0} p_k \\log_2 p_k

    ``0 log 0 = 0`` is enforced by masking, not by adding an epsilon. Range ``[0, log2 K]``
    (DESIGN.md §1.2). An empty node gives 0.

    Parameters
    ----------
    counts : array_like, shape (..., K)
        Class counts (non-negative).

    Returns
    -------
    ndarray of shape (...) or float
        Entropy per row of counts.
    """
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum(axis=-1, keepdims=True)
    p = c / np.where(n > 0, n, 1.0)
    out = 0.0 - _xlog2x(p).sum(axis=-1)  # "0.0 -" avoids returning -0.0 for pure nodes
    return _scalar_if_0d(out)


def squared_error(y: ArrayLike) -> float:
    """Variance of the targets, the squared-error impurity (two-pass form).

    .. math:: V = \\frac{1}{N}\\sum_i (y_i - \\bar y)^2

    The leaf prediction ``mean(y)`` minimizes this. The two-pass form is used instead of
    ``Q/N - mean**2`` because the latter cancels catastrophically when ``|mean| >> std``
    (DESIGN.md §1.3).

    Parameters
    ----------
    y : array_like, shape (N,)
        Targets of the node, ``N >= 1``.

    Returns
    -------
    float
        Mean squared deviation from the mean.
    """
    a = np.asarray(y, dtype=np.float64)
    if a.size == 0:
        raise ValueError("squared_error needs at least one target")
    return float(np.mean((a - a.mean()) ** 2))


def impurity_decrease(
    n_total: float,
    n_node: float,
    imp_node: float,
    n_left: float,
    imp_left: float,
    n_right: float,
    imp_right: float,
) -> float:
    """Weighted impurity decrease of a split (identical to scikit-learn; DESIGN.md §2.1).

    .. math::
        \\Delta I = \\frac{N_t}{N}\\Big[I(t) - \\frac{N_L}{N_t} I(L) - \\frac{N_R}{N_t} I(R)\\Big]
        = \\frac{1}{N}\\big[N_t I(t) - N_L I(L) - N_R I(R)\\big]

    It is >= 0 for gini/entropy/variance because they are concave (DESIGN.md §2.2).

    Parameters
    ----------
    n_total : float
        ``N``, rows at the tree root.
    n_node, n_left, n_right : float
        Rows at the node and at its children (``n_left + n_right == n_node``).
    imp_node, imp_left, imp_right : float
        Impurities of the node and of its children.
    """
    return (n_node * imp_node - n_left * imp_left - n_right * imp_right) / n_total


class Criterion(Protocol):
    """Strategy interface used by the splitter and the tree."""

    is_classification: bool

    def node_impurity(self, stats: np.ndarray) -> float:
        """Impurity ``I(t)`` of ONE node from its per-row statistics.

        ``stats`` is ``(m, K)`` one-hot rows (classification) or ``(m,)`` node-mean-centered
        targets (regression).
        """
        ...

    def children_cost(
        self, cum: np.ndarray, n_left: np.ndarray, n_right: np.ndarray, total_ss: float = 0.0
    ) -> np.ndarray:
        """``N_L I(L) + N_R I(R)`` for each of the ``m - 1`` prefix splits, shape ``(m-1,)``.

        ``cum`` is the full prefix-sum array of the sorted ``stats`` (``(m, K)`` counts or
        ``(m,)`` centered sums); it is sliced here as ``cum[:-1]`` (left) and
        ``cum[-1] - cum[:-1]`` (right). ``n_left = 1..m-1`` and ``n_right = m - n_left`` are
        float64. ``total_ss`` is ``Q_t`` and is used by regression only.
        """
        ...


def _split_cum(cum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left/right sufficient statistics for every prefix split of ``cum``."""
    left = cum[:-1]
    return left, cum[-1] - left


class _GiniCriterion:
    """Gini: ``N G = N - ||c||^2 / N`` so cost = ``n_L - |c_L|^2/n_L + n_R - |c_R|^2/n_R``."""

    is_classification = True
    name = "gini"

    def node_impurity(self, stats: np.ndarray) -> float:
        return float(gini(stats.sum(axis=0)))

    def children_cost(
        self, cum: np.ndarray, n_left: np.ndarray, n_right: np.ndarray, total_ss: float = 0.0
    ) -> np.ndarray:
        cl, cr = _split_cum(cum)
        ssq_l = np.einsum("ij,ij->i", cl, cl)
        ssq_r = np.einsum("ij,ij->i", cr, cr)
        return (n_left - ssq_l / n_left) + (n_right - ssq_r / n_right)


class _EntropyCriterion:
    """Entropy: ``N H = N log2 N - sum_k c_k log2 c_k`` (DESIGN.md §3.2)."""

    is_classification = True
    name = "entropy"

    def node_impurity(self, stats: np.ndarray) -> float:
        return float(entropy(stats.sum(axis=0)))

    def children_cost(
        self, cum: np.ndarray, n_left: np.ndarray, n_right: np.ndarray, total_ss: float = 0.0
    ) -> np.ndarray:
        cl, cr = _split_cum(cum)
        left = n_left * np.log2(n_left) - _xlog2x(cl).sum(axis=1)
        right = n_right * np.log2(n_right) - _xlog2x(cr).sum(axis=1)
        return left + right


class _SquaredErrorCriterion:
    """Squared error: cost = ``Q_t - s_L^2/n_L - s_R^2/n_R`` on node-centered targets."""

    is_classification = False
    name = "squared_error"

    def node_impurity(self, stats: np.ndarray) -> float:
        return squared_error(stats)

    def children_cost(
        self, cum: np.ndarray, n_left: np.ndarray, n_right: np.ndarray, total_ss: float = 0.0
    ) -> np.ndarray:
        sl, sr = _split_cum(cum)
        return total_ss - sl * sl / n_left - sr * sr / n_right


_SQUARED = _SquaredErrorCriterion()

#: Registry of criteria by name. ``"mse"`` is an alias for ``"squared_error"``.
CRITERIA: dict[str, Criterion] = {
    "gini": _GiniCriterion(),
    "entropy": _EntropyCriterion(),
    "squared_error": _SQUARED,
    "mse": _SQUARED,
}


def get_criterion(name: str) -> Criterion:
    """Look up a criterion by name (``"gini"``, ``"entropy"``, ``"squared_error"``/``"mse"``)."""
    try:
        return CRITERIA[name]
    except KeyError:
        raise ValueError(f"Unknown criterion {name!r}; choose from {sorted(CRITERIA)}") from None
