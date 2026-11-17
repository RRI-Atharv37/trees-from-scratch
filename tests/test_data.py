"""Data loading: Adult parser gotchas (offline fixture), splits, and the real files (slow)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from treesfs.data import (
    ADULT_CATEGORICAL,
    ADULT_COLUMNS,
    ADULT_NUMERIC,
    load_adult,
    load_dataset,
    parse_adult,
    task_for,
    train_test_split_indices,
)

ROOT = Path(__file__).resolve().parents[1]


def test_column_constants():
    assert len(ADULT_COLUMNS) == 15
    assert len(ADULT_NUMERIC) + len(ADULT_CATEGORICAL) == 14
    assert set(ADULT_NUMERIC + ADULT_CATEGORICAL + ["income"]) == set(ADULT_COLUMNS)


def test_parse_train_fixture(adult_fixture_dir):
    cols, y = parse_adult(adult_fixture_dir / "adult.data", is_test=False)
    assert len(y) == 6 and y.dtype == np.int64
    assert y.tolist() == [0, 0, 0, 0, 1, 1]
    assert set(cols) == set(ADULT_NUMERIC + ADULT_CATEGORICAL)
    for name in ADULT_NUMERIC:
        assert cols[name].dtype == np.float64 and cols[name].shape == (6,)
    assert cols["age"].tolist() == [39, 50, 38, 53, 28, 37]
    assert cols["workclass"][0] == "State-gov"  # whitespace stripped
    assert cols["workclass"][3] == "missing"  # "?" -> "missing"
    assert cols["occupation"][3] == "missing" and cols["native-country"][3] == "missing"
    assert "?" not in np.concatenate([cols[c] for c in ADULT_CATEGORICAL])


def test_parse_test_fixture_skips_junk_line_and_dots(adult_fixture_dir):
    cols, y = parse_adult(adult_fixture_dir / "adult.test", is_test=True)
    assert len(y) == 4  # junk line and trailing blank line dropped
    assert y.tolist() == [0, 0, 1, 1]  # ">50K." and "<=50K." parsed
    assert cols["age"][0] == 25.0
    assert cols["workclass"][2] == "missing"


def test_parse_rejects_malformed_row(tmp_path):
    bad = tmp_path / "adult.data"
    bad.write_text("39, State-gov, 77516\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1 has 3 fields"):
        parse_adult(bad, is_test=False)
    odd = tmp_path / "odd.data"
    odd.write_text("a, b, c, d, e, f, g, h, i, j, k, l, m, n, maybe\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_adult(odd, is_test=False)


def test_load_adult_fixture(fixture_config):
    data = load_adult(fixture_config)
    assert data.X_train.shape[0] == 6 and data.X_test.shape[0] == 4
    assert data.X_train.dtype == np.float64
    assert data.feature_names[: len(ADULT_NUMERIC)] == ADULT_NUMERIC
    assert data.y_train.tolist() == [0, 0, 0, 0, 1, 1]


def test_load_adult_missing_files_message(default_config, tmp_path):
    from dataclasses import replace

    cfg = replace(default_config, paths=replace(default_config.paths, raw=tmp_path))
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_adult(cfg)


def test_split_indices_disjoint_cover_deterministic():
    tr, te = train_test_split_indices(100, 0.2, np.random.default_rng(3))
    assert len(te) == 20 and len(tr) == 80
    assert set(tr).isdisjoint(te)
    assert sorted(np.concatenate([tr, te]).tolist()) == list(range(100))
    tr2, te2 = train_test_split_indices(100, 0.2, np.random.default_rng(3))
    assert np.array_equal(tr, tr2) and np.array_equal(te, te2)
    tr3, _ = train_test_split_indices(100, 0.2, np.random.default_rng(4))
    assert not np.array_equal(tr, tr3)
    with pytest.raises(ValueError):
        train_test_split_indices(10, 1.5, np.random.default_rng(0))


def test_dataset_dispatch(default_config):
    assert task_for("adult") == "classification" and task_for("california") == "regression"
    with pytest.raises(ValueError):
        task_for("iris")
    with pytest.raises(ValueError):
        load_dataset("iris", default_config)


@pytest.mark.slow
@pytest.mark.skipif(
    not (ROOT / "data" / "raw" / "adult.data").exists(), reason="Adult not downloaded"
)
def test_real_adult_files():
    raw = ROOT / "data" / "raw"
    cols_tr, y_tr = parse_adult(raw / "adult.data", is_test=False)
    cols_te, y_te = parse_adult(raw / "adult.test", is_test=True)
    assert len(y_tr) == 32561 and len(y_te) == 16281
    assert y_tr.mean() == pytest.approx(0.2408, abs=5e-5)
    assert y_te.mean() == pytest.approx(0.2362, abs=5e-5)
    for cols, expected in (
        (cols_tr, {"workclass": 1836, "occupation": 1843, "native-country": 583}),
        (cols_te, {"workclass": 963, "occupation": 966, "native-country": 274}),
    ):
        for name, count in expected.items():
            assert int((cols[name] == "missing").sum()) == count
        for name in set(ADULT_CATEGORICAL) - set(expected):
            assert int((cols[name] == "missing").sum()) == 0
    assert (cols_tr["native-country"] == "Holand-Netherlands").sum() == 1
    assert (cols_te["native-country"] == "Holand-Netherlands").sum() == 0
