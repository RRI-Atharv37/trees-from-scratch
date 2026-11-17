"""Config loading, utils helpers and metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import r2_score as sk_r2

from treesfs import accuracy, load_config, mae, mse, r2_score
from treesfs.utils import (
    PROJECT_ROOT,
    ensure_dir,
    get_logger,
    set_seed,
    timer,
    update_marked_section,
)


def test_default_config_loads(default_config):
    assert default_config.seed == 42
    assert default_config.paths.raw == PROJECT_ROOT / "data" / "raw"
    assert default_config.data.max_categories == 16
    assert default_config.tree.max_depth is None
    assert default_config.depth_sweep.depths[-1] is None
    assert isinstance(default_config.depth_sweep.depths, tuple)
    assert default_config.export.small_tree_depth == 3


def test_unknown_keys_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 1\ntree:\n  max_dept: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_dept"):
        load_config(p)
    p.write_text("sede: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sede"):
        load_config(p)
    p.write_text("tree: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(p)


def test_partial_config_uses_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 7\ntree:\n  max_depth: 4\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.seed == 7 and cfg.tree.max_depth == 4 and cfg.tree.min_samples_leaf == 1
    assert cfg.paths.figures.is_absolute()


def test_set_seed_reproducible():
    a = set_seed(5).random(3)
    x = np.random.rand(2)
    b = set_seed(5).random(3)
    y = np.random.rand(2)
    assert np.array_equal(a, b) and np.array_equal(x, y)


def test_timer_logger_dirs(tmp_path):
    with timer() as t:
        pass
    assert t.elapsed >= 0
    assert get_logger("x").name == "treesfs.x" and get_logger("treesfs.y").name == "treesfs.y"
    d = ensure_dir(tmp_path / "a" / "b")
    assert d.is_dir()


def test_update_marked_section_idempotent(tmp_path):
    p = tmp_path / "results.md"
    update_marked_section(p, "adult", "A v1")
    update_marked_section(p, "california", "C v1")
    once = p.read_text(encoding="utf-8")
    update_marked_section(p, "adult", "A v1")
    assert p.read_text(encoding="utf-8") == once
    update_marked_section(p, "adult", "A v2")
    text = p.read_text(encoding="utf-8")
    assert "A v2" in text and "A v1" not in text and "C v1" in text
    assert text.count("<!-- treesfs:adult:start -->") == 1
    assert text.startswith("# Results")


def test_metrics_against_sklearn(rng):
    y = rng.normal(size=50)
    p = y + rng.normal(scale=0.3, size=50)
    assert r2_score(y, p) == pytest.approx(sk_r2(y, p))
    assert mse(y, p) == pytest.approx(np.mean((y - p) ** 2))
    assert mae(y, p) == pytest.approx(np.mean(np.abs(y - p)))
    assert accuracy([1, 0, 1, 1], [1, 1, 1, 0]) == 0.5


def test_r2_degenerate_matches_sklearn():
    const = np.full(5, 3.0)
    # constant targets: 1.0 if the prediction is perfect, else 0.0 (sklearn force_finite)
    assert r2_score(const, const) == 1.0 == sk_r2(const, const)
    assert r2_score(const, const + 1) == 0.0 == sk_r2(const, const + 1)


def test_metrics_validate_shapes():
    with pytest.raises(ValueError):
        accuracy([1, 2], [1])
    with pytest.raises(ValueError):
        mse([], [])


def test_forest_and_study_sections_load(default_config, tmp_path):
    assert default_config.forest.n_estimators == 100 and default_config.forest.oob_score
    assert default_config.forest.max_features is None  # task default
    assert default_config.n_trees_sweep.grid[-1] == 200
    assert isinstance(default_config.n_trees_sweep.grid, tuple)
    assert default_config.importance.max_rows == 5000
    assert default_config.benchmark.tree_depth == 10
    p = tmp_path / "c.yaml"
    p.write_text("forest:\n  n_estimator: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="n_estimator"):
        load_config(p)
