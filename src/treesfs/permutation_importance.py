"""Permutation feature importance on held-out data. NumPy only (joblib for threads).

Impurity importance (DESIGN.md §7) is computed from **training** data and rewards features that
offer many candidate thresholds (a near-continuous column such as Adult's ``fnlwgt`` can always
find *some* split that lowers training impurity, even if it generalises to nothing). Permutation
importance asks the question that matters instead: *how much does the held-out score drop when
this column's link to the target is destroyed?*

For a feature ``j``, repeat ``r = 1..R``: shuffle column ``j`` of ``X`` (rows keep every other
feature, so the marginal distribution of ``x_j`` is unchanged but its association with ``y`` and
with the other features is broken), re-score, and record

    importance[j, r] = score(X, y) - score(X with column j permuted, y).

A large positive value means the model relies on the feature; ~0 means it does not; a negative
value is noise (a permuted column that happened to help). Caveat: strongly correlated features
share their importance (permuting one leaves its twin to compensate), and permuting a single
one-hot column produces rows that are not valid one-hot rows.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from joblib import Parallel, delayed

Scoring = Callable[[Any, np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class PermutationImportanceResult:
    """Output of :func:`permutation_importance`.

    Attributes
    ----------
    importances : ndarray (d, R)
        Score drop for every feature and repeat.
    importances_mean, importances_std : ndarray (d,)
        Mean and (population) standard deviation over the ``R`` repeats.
    baseline_score : float
        Score of the unpermuted data.
    n_rows : int
        Number of held-out rows actually used.
    """

    importances: np.ndarray
    importances_mean: np.ndarray
    importances_std: np.ndarray
    baseline_score: float
    n_rows: int


def _default_scoring(model: Any, X: np.ndarray, y: np.ndarray) -> float:
    return float(model.score(X, y))


def _score_features(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    features: list[int],
    seeds: list[np.random.SeedSequence],
    n_repeats: int,
    scoring: Scoring,
    baseline: float,
) -> np.ndarray:
    """Score drops for ``features``; returns shape ``(len(features), n_repeats)``."""
    n = X.shape[0]
    X_perm = X.copy()  # one private copy per worker; one column is shuffled and restored
    out = np.empty((len(features), n_repeats))
    # Loops over features and repeats (each iteration is one full vectorised prediction).
    for row, (j, seed) in enumerate(zip(features, seeds, strict=True)):
        rng = np.random.default_rng(seed)
        original = X[:, j].copy()
        for r in range(n_repeats):
            X_perm[:, j] = original[rng.permutation(n)]
            out[row, r] = baseline - scoring(model, X_perm, y)
        X_perm[:, j] = original
    return out


def permutation_importance(
    model: Any,
    X: Any,
    y: Any,
    n_repeats: int = 5,
    random_state: int | np.random.SeedSequence | None = None,
    scoring: Scoring | None = None,
    max_rows: int | None = None,
    n_jobs: int | None = None,
) -> PermutationImportanceResult:
    """Permutation importance of every column of ``X`` for a fitted ``model``.

    Parameters
    ----------
    model : fitted estimator
        Needs ``predict``; ``score(X, y)`` is used unless ``scoring`` is given (higher = better:
        accuracy for classifiers, R² for regressors).
    X : array_like, shape (n, d)
        **Held-out** data. Using training data measures memorisation, not usefulness.
    y : array_like, shape (n,)
    n_repeats : int
        Shuffles per feature ``R >= 1``; the std over them is a noise estimate.
    random_state : int, SeedSequence or None
        Each feature gets its own spawned seed, so the result does not depend on ``n_jobs``.
    scoring : callable ``(model, X, y) -> float``, optional
    max_rows : int, optional
        Randomly keep at most this many rows (without replacement) to bound the cost
        ``d * R`` predictions.
    n_jobs : int, optional
        Worker threads over feature chunks (the model is shared, never pickled).

    Returns
    -------
    PermutationImportanceResult
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features), got shape {X.shape}")
    if y.shape != (X.shape[0],):
        raise ValueError(f"y must have shape ({X.shape[0]},), got {y.shape}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    if max_rows is not None and max_rows < 2:
        raise ValueError(f"max_rows must be >= 2 or None, got {max_rows}")
    scoring = scoring or _default_scoring
    n, d = X.shape

    root = (
        random_state
        if isinstance(random_state, np.random.SeedSequence)
        else np.random.SeedSequence(None if random_state is None else int(random_state))
    )
    root = np.random.SeedSequence(root.entropy, spawn_key=root.spawn_key)
    rows_seed, feature_root = root.spawn(2)
    feature_seeds = feature_root.spawn(d)

    if max_rows is not None and n > max_rows:
        keep = np.sort(np.random.default_rng(rows_seed).choice(n, size=max_rows, replace=False))
        X, y, n = X[keep], y[keep], max_rows

    baseline = scoring(model, X, y)
    n_chunks = max(1, min(d, _effective_jobs(n_jobs)))
    chunks = [list(range(c, d, n_chunks)) for c in range(n_chunks)]  # interleaved: balanced work
    parts = Parallel(n_jobs=n_chunks, prefer="threads")(
        delayed(_score_features)(
            model, X, y, ch, [feature_seeds[j] for j in ch], n_repeats, scoring, baseline
        )
        for ch in chunks
    )
    importances = np.empty((d, n_repeats))
    for ch, part in zip(chunks, parts, strict=True):
        importances[ch] = part
    return PermutationImportanceResult(
        importances=importances,
        importances_mean=importances.mean(axis=1),
        importances_std=importances.std(axis=1),
        baseline_score=float(baseline),
        n_rows=n,
    )


def _effective_jobs(n_jobs: int | None) -> int:
    """Resolve joblib-style ``n_jobs`` (``None``/1 -> 1, ``-1`` -> all cores) to a positive int."""
    if n_jobs is None or n_jobs == 1:
        return 1
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return int(n_jobs)
