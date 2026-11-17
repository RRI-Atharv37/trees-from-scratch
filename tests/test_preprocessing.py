"""OneHotEncoder and design-matrix assembly (DESIGN.md §10.2)."""

from __future__ import annotations

import numpy as np
import pytest

from treesfs.data import load_adult
from treesfs.preprocessing import OneHotEncoder, build_design_matrix


def _col(*values: str) -> np.ndarray:
    return np.array(values, dtype=str)


def test_basic_encoding_and_names():
    enc = OneHotEncoder().fit({"color": _col("red", "blue", "red", "green")})
    assert enc.categories_["color"] == ["blue", "green", "red"]
    assert enc.get_feature_names_out() == ["color=blue", "color=green", "color=red"]
    X = enc.transform({"color": _col("red", "blue")})
    assert X.dtype == np.float64
    assert X.tolist() == [[0, 0, 1], [1, 0, 0]]


def test_max_categories_cap_gives_exact_width_with_other_last():
    col = _col(*(["a"] * 5 + ["b"] * 4 + ["c"] * 3 + ["d"] * 2 + ["e"]))
    enc = OneHotEncoder(max_categories=3).fit({"x": col})
    assert enc.categories_["x"] == ["a", "b", "other"]
    X = enc.transform({"x": col})
    assert X.shape == (15, 3)
    assert (X.sum(axis=1) == 1).all()  # every row lands in exactly one column
    assert X[:, 2].sum() == 6  # c + d + e


def test_cap_tie_break_is_by_name():
    col = _col("z", "y", "x", "w")  # all count 1
    enc = OneHotEncoder(max_categories=3).fit({"c": col})
    assert enc.categories_["c"] == ["w", "x", "other"]


def test_no_cap_needed_means_no_other_bucket():
    enc = OneHotEncoder(max_categories=4).fit({"x": _col("a", "b", "c", "d")})
    assert enc.categories_["x"] == ["a", "b", "c", "d"]


def test_unseen_goes_to_other_or_zero_block():
    with_other = OneHotEncoder(max_categories=2).fit({"x": _col("a", "a", "b", "c")})
    assert with_other.categories_["x"] == ["a", "other"]
    assert with_other.transform({"x": _col("zzz")}).tolist() == [[0, 1]]
    no_other = OneHotEncoder().fit({"x": _col("a", "b")})
    assert no_other.transform({"x": _col("zzz", "a")}).tolist() == [[0, 0], [1, 0]]


def test_min_frequency_sends_rare_to_other():
    enc = OneHotEncoder(min_frequency=2).fit({"x": _col("a", "a", "b", "b", "c")})
    assert enc.categories_["x"] == ["a", "b", "other"]
    assert enc.transform({"x": _col("c")}).tolist() == [[0, 0, 1]]


def test_literal_other_category_is_merged_into_bucket():
    enc = OneHotEncoder(max_categories=2).fit({"x": _col("a", "a", "a", "other", "q")})
    assert enc.categories_["x"] == ["a", "other"]
    assert enc.transform({"x": _col("other", "q")}).tolist() == [[0, 1], [0, 1]]


def test_multiple_columns_and_fit_transform():
    cols = {"a": _col("x", "y", "x"), "b": _col("p", "p", "q")}
    X = OneHotEncoder().fit_transform(cols)
    assert X.shape == (3, 4)
    names = OneHotEncoder().fit(cols).get_feature_names_out()
    assert names == ["a=x", "a=y", "b=p", "b=q"]


def test_errors():
    with pytest.raises(ValueError):
        OneHotEncoder(max_categories=1)
    with pytest.raises(ValueError):
        OneHotEncoder(min_frequency=0)
    with pytest.raises(RuntimeError, match="not fitted"):
        OneHotEncoder().transform({"a": _col("x")})
    enc = OneHotEncoder().fit({"a": _col("x")})
    with pytest.raises(KeyError):
        enc.transform({"b": _col("x")})


def test_build_design_matrix_layout():
    enc = OneHotEncoder().fit({"c": _col("u", "v")})
    X, names = build_design_matrix(
        {"n1": np.array([1.0, 2.0]), "n2": np.array([3.0, 4.0])}, {"c": _col("v", "u")}, enc
    )
    assert names == ["n1", "n2", "c=u", "c=v"]
    assert X.tolist() == [[1.0, 3.0, 0.0, 1.0], [2.0, 4.0, 1.0, 0.0]]


def test_encoder_fit_on_train_only(fixture_config):
    data = load_adult(fixture_config)
    # Fixture: "Holand-Netherlands" is train-only, "Mexico" and "Local-gov" are test-only.
    assert "native-country=Holand-Netherlands" in data.feature_names
    assert "native-country=Mexico" not in data.feature_names
    assert "workclass=Local-gov" not in data.feature_names
    assert data.X_train.shape[1] == data.X_test.shape[1] == len(data.feature_names)
    j = data.feature_names.index("workclass=Private")
    assert data.X_test[0, j] == 1.0  # known category encoded
    # Local-gov row (index 3) has all-zero workclass block: unseen, and no "other" bucket
    wc = [i for i, n in enumerate(data.feature_names) if n.startswith("workclass=")]
    assert data.X_test[3, wc].sum() == 0.0
    # each row's one-hot block for a fully-seen column sums to 1
    sex = [i for i, n in enumerate(data.feature_names) if n.startswith("sex=")]
    assert (data.X_train[:, sex].sum(axis=1) == 1).all()
