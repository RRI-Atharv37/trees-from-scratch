"""One-hot encoding for categorical columns (see DESIGN.md §10.2). NumPy only."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

OTHER = "other"


class OneHotEncoder:
    """One-hot encoder with a per-column category cap and an ``"other"`` bucket.

    For each column the categories are counted on the **training** data. A column with more than
    ``max_categories`` categories keeps the ``max_categories - 1`` most frequent (ties broken by
    name, for determinism) and sends the rest to an ``"other"`` bucket, so it produces exactly
    ``max_categories`` output columns. Categories seen fewer than ``min_frequency`` times also
    go to ``"other"``. Output columns of a feature are sorted by name with ``"other"`` last.

    At transform time, unseen values go to ``"other"`` when the column has that bucket,
    otherwise they produce an all-zero block (no error). A literal category called ``"other"``
    is merged into the bucket whenever the bucket exists.

    Parameters
    ----------
    max_categories : int or None
        Cap on output columns per feature (``>= 2``), or ``None`` for no cap.
    min_frequency : int
        Minimum training count for a category to get its own column (``>= 1``).

    Attributes
    ----------
    categories_ : dict[str, list[str]]
        Output categories per column (including a trailing ``"other"`` if present).
    """

    def __init__(self, max_categories: int | None = None, min_frequency: int = 1) -> None:
        if max_categories is not None and max_categories < 2:
            raise ValueError(f"max_categories must be None or >= 2, got {max_categories}")
        if min_frequency < 1:
            raise ValueError(f"min_frequency must be >= 1, got {min_frequency}")
        self.max_categories = max_categories
        self.min_frequency = min_frequency

    def fit(self, columns: Mapping[str, np.ndarray]) -> OneHotEncoder:
        """Learn the categories of each column.

        Parameters
        ----------
        columns : mapping of name -> ndarray of str, each shape (n,)
        """
        self.categories_: dict[str, list[str]] = {}
        self._has_other: dict[str, bool] = {}
        # Loops below are over columns and over *categories*, never over samples.
        for name, col in columns.items():
            values, counts = np.unique(np.asarray(col, dtype=str), return_counts=True)
            ranked = sorted(
                zip(values.tolist(), counts.tolist(), strict=True), key=lambda vc: (-vc[1], vc[0])
            )
            kept = [v for v, c in ranked if c >= self.min_frequency]
            has_other = len(kept) < len(ranked)
            if self.max_categories is not None and len(kept) + has_other > self.max_categories:
                kept = kept[: self.max_categories - 1]
                has_other = True
            if has_other:
                kept = [v for v in kept if v != OTHER]
            self.categories_[name] = sorted(kept) + ([OTHER] if has_other else [])
            self._has_other[name] = has_other
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "categories_"):
            raise RuntimeError("OneHotEncoder is not fitted; call fit() first")

    def transform(self, columns: Mapping[str, np.ndarray]) -> np.ndarray:
        """Encode columns as a 0/1 matrix.

        Parameters
        ----------
        columns : mapping of name -> ndarray of str, each shape (n,)
            Must contain every column seen in ``fit``.

        Returns
        -------
        ndarray, shape (n, sum of per-column category counts), float64
        """
        self._check_fitted()
        blocks = []
        for name, cats in self.categories_.items():
            if name not in columns:
                raise KeyError(f"column {name!r} missing from input")
            col = np.asarray(columns[name], dtype=str)
            has_other = self._has_other[name]
            named = cats[:-1] if has_other else cats
            block = np.zeros((col.shape[0], len(cats)), dtype=np.float64)
            for j, cat in enumerate(named):  # loop over categories
                block[:, j] = col == cat
            if has_other:
                block[:, -1] = ~np.isin(col, named)
            blocks.append(block)
        if not blocks:
            return np.zeros((0, 0))
        return np.hstack(blocks)

    def fit_transform(self, columns: Mapping[str, np.ndarray]) -> np.ndarray:
        """Equivalent to ``fit(columns).transform(columns)``."""
        return self.fit(columns).transform(columns)

    def get_feature_names_out(self) -> list[str]:
        """Output column names in matrix order, formatted ``"column=category"``."""
        self._check_fitted()
        return [f"{name}={cat}" for name, cats in self.categories_.items() for cat in cats]


def build_design_matrix(
    numeric: Mapping[str, np.ndarray],
    categorical: Mapping[str, np.ndarray],
    encoder: OneHotEncoder,
) -> tuple[np.ndarray, list[str]]:
    """Stack numeric columns (untouched) and one-hot categorical columns.

    Trees need no scaling, so numeric columns pass through as float64.

    Parameters
    ----------
    numeric : mapping of name -> ndarray (n,)
    categorical : mapping of name -> ndarray of str (n,)
    encoder : OneHotEncoder
        Already fitted (on the training split only).

    Returns
    -------
    X : ndarray, shape (n, n_numeric + n_one_hot), float64
    feature_names : list[str]
    """
    cat_block = encoder.transform(categorical)
    n = cat_block.shape[0] if cat_block.size else len(next(iter(numeric.values())))
    num_block = (
        np.column_stack([np.asarray(v, dtype=np.float64) for v in numeric.values()])
        if numeric
        else np.zeros((n, 0))
    )
    X = np.hstack([num_block, cat_block]) if cat_block.size else num_block
    return X, [*numeric.keys(), *encoder.get_feature_names_out()]
