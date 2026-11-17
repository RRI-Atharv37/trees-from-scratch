"""Evaluation metrics (NumPy only)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(y_true), np.asarray(y_pred)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: y_true {a.shape} vs y_pred {b.shape}")
    if a.size == 0:
        raise ValueError("metrics need at least one sample")
    return a, b


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction of exactly matching labels, ``mean(y_true == y_pred)``.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n,)
    """
    a, b = _pair(y_true, y_pred)
    return float(np.mean(a == b))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination.

    .. math:: R^2 = 1 - \\frac{\\sum_i (y_i - \\hat y_i)^2}{\\sum_i (y_i - \\bar y)^2}

    Degenerate case ``SS_tot == 0`` (constant targets) follows scikit-learn's
    ``force_finite=True`` convention: ``1.0`` for a perfect prediction, ``0.0`` otherwise.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n,)
    """
    a, b = _pair(y_true, y_pred)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean squared error, ``mean((y_true - y_pred)**2)``."""
    a, b = _pair(y_true, y_pred)
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error, ``mean(|y_true - y_pred|)``."""
    a, b = _pair(y_true, y_pred)
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))
