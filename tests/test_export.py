"""Exporters (text, DOT) and the matplotlib plots."""

from __future__ import annotations

import re

import numpy as np
import pytest

from treesfs import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, export_text
from treesfs.export import render_dot
from treesfs.models.tree import TreeArrays
from treesfs.viz import plot_depth_sweep, plot_tree

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _hand_built_classifier() -> DecisionTreeClassifier:
    """Depth-2 tree, preorder ids: 0 root, 1 left subtree (2, 3), 4 right leaf."""
    t = TreeArrays(
        children_left=np.array([1, 2, -1, -1, -1]),
        children_right=np.array([4, 3, -1, -1, -1]),
        feature=np.array([0, 1, -2, -2, -2]),
        threshold=np.array([12.5, 5095.5, -2.0, -2.0, -2.0]),
        value=np.array([[0.75, 0.25], [0.9, 0.1], [1.0, 0.0], [0.2, 0.8], [0.3, 0.7]]),
        n_node_samples=np.array([100, 60, 50, 10, 40]),
        impurity=np.array([0.375, 0.18, 0.0, 0.32, 0.42]),
    )
    clf = DecisionTreeClassifier()
    clf.tree_, clf.classes_, clf.n_classes_, clf.n_features_in_ = t, np.array([0, 1]), 2, 2
    return clf


EXPECTED_TEXT = """\
|--- education-num <= 12.500
|   |--- capital-gain <= 5095.500
|   |   |--- class: 0
|   |--- capital-gain >  5095.500
|   |   |--- class: 1
|--- education-num >  12.500
|   |--- class: 1
"""


def test_export_text_matches_expected_exactly():
    out = export_text(_hand_built_classifier(), ["education-num", "capital-gain"])
    assert out == EXPECTED_TEXT


def test_export_text_class_names_and_default_feature_names():
    out = export_text(_hand_built_classifier(), class_names=["low", "high"], decimals=1)
    assert "|--- feature_0 <= 12.5" in out and "|--- class: high" in out


def test_export_text_truncation_line():
    out = export_text(_hand_built_classifier(), ["education-num", "capital-gain"], max_depth=0)
    assert out == (
        "|--- education-num <= 12.500\n"
        "|   |--- truncated branch of depth 2\n"
        "|--- education-num >  12.500\n"
        "|   |--- truncated branch of depth 1\n"
    )


def test_export_text_regression_leaf(reg_data):
    X, y = reg_data
    reg = DecisionTreeRegressor(max_depth=2).fit(X, y)
    out = export_text(reg, decimals=2)
    leaves = [line for line in out.splitlines() if "value:" in line]
    assert len(leaves) == reg.get_n_leaves()
    assert re.fullmatch(r"(\|   )*\|--- value: \[-?\d+\.\d\d\]", leaves[0])


def test_export_text_unfitted_and_bad_names():
    with pytest.raises(RuntimeError):
        export_text(DecisionTreeClassifier())
    with pytest.raises(ValueError, match="feature_names"):
        export_text(_hand_built_classifier(), ["only_one"])


def test_export_text_deep_tree_is_iterative(blobs_clf):
    X, y = blobs_clf
    clf = DecisionTreeClassifier().fit(X, y)
    out = export_text(clf)
    assert out.count("class:") == clf.get_n_leaves()


def test_export_graphviz_structure(blobs_clf):
    X, y = blobs_clf
    clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
    dot = export_graphviz(clf, class_names=["a", "b", "c"])
    assert dot.startswith("digraph")
    lines = dot.splitlines()
    nodes = [ln for ln in lines if re.match(r"^\d+ \[", ln)]
    edges = [ln for ln in lines if re.match(r"^\d+ -> \d+", ln)]
    assert len(nodes) == clf.tree_.node_count
    assert len(edges) == clf.tree_.node_count - 1
    assert lines[-1] == "}" and "fillcolor" in nodes[0]
    assert "class = " in nodes[0] and "samples = 300" in nodes[0] and "impurity = " in nodes[0]
    assert dot.count('headlabel="True"') == dot.count('headlabel="False"')


def test_export_graphviz_options(reg_data):
    X, y = reg_data
    reg = DecisionTreeRegressor(max_depth=4).fit(X, y)
    plain = export_graphviz(reg, filled=False)
    assert "fillcolor" not in plain and "value = " in plain and "class = " not in plain
    truncated = export_graphviz(reg, max_depth=1)
    assert len([ln for ln in truncated.splitlines() if re.match(r"^\d+ \[", ln)]) == 3


def test_render_dot_without_graphviz_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert render_dot("digraph T {}", tmp_path / "t.png") is None


def test_plot_tree_writes_png(blobs_clf, tmp_path):
    X, y = blobs_clf
    clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
    out = plot_tree(clf, [f"f{i}" for i in range(6)], ["a", "b", "c"], tmp_path / "t.png", 3)
    data = out.read_bytes()
    assert len(data) > 5_000 and data.startswith(PNG_MAGIC)


def test_plot_tree_regressor_and_single_leaf(reg_data, tmp_path):
    X, y = reg_data
    reg = DecisionTreeRegressor(max_depth=2).fit(X, y)
    assert plot_tree(reg, None, None, tmp_path / "r.png", 2).stat().st_size > 0
    leaf = DecisionTreeClassifier().fit(X, np.zeros(len(y), dtype=int))
    assert plot_tree(leaf, None, None, tmp_path / "l.png", 2).stat().st_size > 0


def test_plot_depth_sweep_writes_png(tmp_path):
    depths = [1, 2, 3, 5, None]
    out = plot_depth_sweep(
        depths,
        [0.76, 0.83, 0.84, 0.87, 1.0],
        [0.76, 0.83, 0.845, 0.86, 0.81],
        tmp_path / "sweep.png",
        best_depth=5,
    )
    assert out.read_bytes().startswith(PNG_MAGIC)


def test_new_forest_figures_write_pngs(tmp_path):
    from treesfs.viz import plot_benchmark, plot_importance_comparison, plot_n_trees_sweep

    grid = [1, 2, 5, 10]
    p1 = plot_n_trees_sweep(
        grid,
        [0.8, 0.82, 0.84, 0.85],
        [0.8, 0.83, 0.85, 0.86],
        tmp_path / "a.png",
        single_tree_score=0.81,
    )
    names = [f"f{i}" for i in range(6)]
    p2 = plot_importance_comparison(
        names,
        [0.4, 0.3, 0.1, 0.1, 0.05, 0.05],
        [0.1, 0.0, 0.05, 0.2, 0.0, 0.01],
        [0.01] * 6,
        tmp_path / "b.png",
        top_k=4,
    )
    p3 = plot_benchmark(["a", "b"], [2.0, 30.0], [0.2, 1.5], tmp_path / "c.png")
    for p in (p1, p2, p3):
        assert p.read_bytes().startswith(PNG_MAGIC) and p.stat().st_size > 1000
