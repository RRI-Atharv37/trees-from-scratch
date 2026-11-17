"""Human-readable tree exports: sklearn-style text, Graphviz DOT, and node-label helpers.

Traversals are iterative (explicit stacks), so very deep trees cannot hit Python's recursion
limit. :func:`node_label` and :func:`node_fill` are shared with :mod:`treesfs.viz`.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from treesfs.models.tree import TREE_LEAF, BaseDecisionTree, DecisionTreeClassifier
from treesfs.utils import get_logger

log = get_logger(__name__)

#: Categorical colors for classes (slots 1-8 of the project palette); cycled beyond 8 classes.
CLASS_COLORS: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
_MIN_ALPHA = 0.12


def _check_fitted(tree: BaseDecisionTree) -> None:
    if not hasattr(tree, "tree_"):
        raise RuntimeError(f"{type(tree).__name__} is not fitted; call fit() first")


def _feature_name(tree: BaseDecisionTree, j: int, feature_names: Sequence[str] | None) -> str:
    if feature_names is None:
        return f"feature_{j}"
    if len(feature_names) != tree.n_features_in_:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries, expected {tree.n_features_in_}"
        )
    return str(feature_names[j])


def _class_name(tree: BaseDecisionTree, k: int, class_names: Sequence[str] | None) -> str:
    if class_names is not None:
        return str(class_names[k])
    return str(tree.classes_[k])  # type: ignore[attr-defined]


def subtree_levels(tree: BaseDecisionTree, node: int) -> int:
    """Number of levels in the subtree under ``node`` (a lone leaf counts as 1), iteratively."""
    t = tree.tree_
    best, stack = 0, [(node, 1)]
    while stack:
        n, lv = stack.pop()
        best = max(best, lv)
        if t.children_left[n] != TREE_LEAF:
            stack.append((int(t.children_left[n]), lv + 1))
            stack.append((int(t.children_right[n]), lv + 1))
    return best


def export_text(
    tree: BaseDecisionTree,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    decimals: int = 3,
    max_depth: int | None = None,
) -> str:
    """Render a fitted tree as indented text, in the same layout as sklearn's ``export_text``.

    ::

        |--- education-num <= 12.500
        |   |--- class: 0
        |--- education-num >  12.500
        |   |--- class: 1

    Regression leaves print ``value: [12.345]``. Nodes deeper than ``max_depth`` (root = 0) are
    replaced by ``|--- truncated branch of depth k`` where ``k`` is the number of levels in the
    omitted subtree. Built iteratively with an explicit stack.

    Parameters
    ----------
    tree : DecisionTreeClassifier or DecisionTreeRegressor
        Fitted estimator.
    feature_names : sequence of str, optional
        Defaults to ``feature_0, feature_1, ...``.
    class_names : sequence of str, optional
        Display names in ``classes_`` order; defaults to the labels themselves.
    decimals : int
        Digits after the decimal point.
    max_depth : int, optional
        Maximum node depth to expand.
    """
    _check_fitted(tree)
    t = tree.tree_
    classify = isinstance(tree, DecisionTreeClassifier)
    lines: list[str] = []
    # Work items: ("node", id, depth) or ("text", line). LIFO, so push in reverse print order.
    stack: list[tuple] = [("node", 0, 0)]
    while stack:
        item = stack.pop()
        if item[0] == "text":
            lines.append(item[1])
            continue
        _, node, depth = item
        indent = "|   " * depth
        if max_depth is not None and depth > max_depth:
            lines.append(f"{indent}|--- truncated branch of depth {subtree_levels(tree, node)}")
        elif t.children_left[node] == TREE_LEAF:
            if classify:
                label = f"class: {_class_name(tree, int(np.argmax(t.value[node])), class_names)}"
            else:
                label = f"value: [{t.value[node, 0]:.{decimals}f}]"
            lines.append(f"{indent}|--- {label}")
        else:
            name = _feature_name(tree, int(t.feature[node]), feature_names)
            thr = f"{t.threshold[node]:.{decimals}f}"
            stack.append(("node", int(t.children_right[node]), depth + 1))
            stack.append(("text", f"{indent}|--- {name} >  {thr}"))
            stack.append(("node", int(t.children_left[node]), depth + 1))
            stack.append(("text", f"{indent}|--- {name} <= {thr}"))
    return "\n".join(lines) + "\n"


def node_label(
    tree: BaseDecisionTree,
    node: int,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    decimals: int = 3,
) -> list[str]:
    """Label rows of a node: split (internal only), impurity, samples, value, class.

    Shared by :func:`export_graphviz` and :func:`treesfs.viz.plot_tree`.
    """
    t = tree.tree_
    rows: list[str] = []
    if t.children_left[node] != TREE_LEAF:
        name = _feature_name(tree, int(t.feature[node]), feature_names)
        rows.append(f"{name} <= {t.threshold[node]:.{decimals}f}")
    rows.append(f"impurity = {t.impurity[node]:.{decimals}f}")
    rows.append(f"samples = {int(t.n_node_samples[node])}")
    if isinstance(tree, DecisionTreeClassifier):
        probs = ", ".join(f"{p:.{decimals}f}" for p in t.value[node])
        rows.append(f"value = [{probs}]")
        rows.append(f"class = {_class_name(tree, int(np.argmax(t.value[node])), class_names)}")
    else:
        rows.append(f"value = {t.value[node, 0]:.{decimals}f}")
    return rows


def node_fill(tree: BaseDecisionTree, node: int) -> tuple[str, float]:
    """Base color (``#rrggbb``) and alpha in ``[0.12, 1]`` for a node.

    Classifier: the majority class's color, alpha growing with purity
    ``(p_max - 1/K) / (1 - 1/K)``. Regressor: one hue, alpha growing with the node's mean
    target relative to the range over the tree.
    """
    t = tree.tree_
    if isinstance(tree, DecisionTreeClassifier):
        probs = t.value[node]
        k = int(np.argmax(probs))
        n_cls = probs.shape[0]
        purity = 1.0 if n_cls < 2 else (probs[k] - 1.0 / n_cls) / (1.0 - 1.0 / n_cls)
        return CLASS_COLORS[k % len(CLASS_COLORS)], _MIN_ALPHA + (1 - _MIN_ALPHA) * float(purity)
    vals = t.value[:, 0]
    lo, hi = float(vals.min()), float(vals.max())
    frac = 0.5 if hi <= lo else (float(t.value[node, 0]) - lo) / (hi - lo)
    return CLASS_COLORS[0], _MIN_ALPHA + (1 - _MIN_ALPHA) * frac


def visible_nodes(tree: BaseDecisionTree, max_depth: int | None) -> list[tuple[int, int, int]]:
    """Preorder list of ``(node, depth, parent)`` that survive ``max_depth`` truncation."""
    t = tree.tree_
    out: list[tuple[int, int, int]] = []
    stack = [(0, 0, -1)]
    while stack:
        node, depth, parent = stack.pop()
        out.append((node, depth, parent))
        if t.children_left[node] != TREE_LEAF and (max_depth is None or depth < max_depth):
            stack.append((int(t.children_right[node]), depth + 1, node))
            stack.append((int(t.children_left[node]), depth + 1, node))
    return out


def export_graphviz(
    tree: BaseDecisionTree,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    filled: bool = True,
    decimals: int = 3,
    max_depth: int | None = None,
) -> str:
    """Export a fitted tree as a Graphviz DOT string.

    Each node is one line ``<id> [label="...", fillcolor="#rrggbbaa"] ;`` and each edge one line
    ``<parent> -> <child> [headlabel="True"|"False"] ;``. Internal labels start with the split
    ``feature <= threshold``; all labels show impurity, samples and value (class proportions or
    mean), plus the majority class for classifiers. With ``filled``, color = majority-class hue
    with alpha proportional to purity (see :func:`node_fill`).

    Parameters
    ----------
    tree : fitted estimator
    feature_names, class_names : sequence of str, optional
    filled : bool
    decimals : int
    max_depth : int, optional
        Nodes deeper than this are omitted (their parent is drawn as a leaf).

    Returns
    -------
    str
        DOT source; render with :func:`render_dot` or ``dot -Tpng``.
    """
    _check_fitted(tree)
    style = '"filled, rounded"' if filled else "rounded"
    out = [
        "digraph Tree {",
        f'node [shape=box, style={style}, color="black", fontname="helvetica"] ;',
        'edge [fontname="helvetica"] ;',
    ]
    t = tree.tree_
    edges: list[str] = []
    for node, _depth, parent in visible_nodes(tree, max_depth):
        rows = node_label(tree, node, feature_names, class_names, decimals)
        label = "\\n".join(r.replace('"', '\\"') for r in rows)
        attrs = f'label="{label}"'
        if filled:
            color, alpha = node_fill(tree, node)
            attrs += f', fillcolor="{color}{int(round(alpha * 255)):02x}"'
        out.append(f"{node} [{attrs}] ;")
        if parent >= 0:
            side = "True" if t.children_left[parent] == node else "False"
            edges.append(
                f'{parent} -> {node} [labeldistance=2.5, labelangle=45, headlabel="{side}"] ;'
            )
    out.extend(edges)
    out.append("}")
    return "\n".join(out) + "\n"


def render_dot(dot: str, out_png: Path | str) -> Path | None:
    """Render DOT to PNG with the ``dot`` binary; return the path, or ``None`` if unavailable.

    A missing Graphviz install logs a warning and is **not** an error (CI and Windows machines
    often lack it; use :func:`treesfs.viz.plot_tree` instead).
    """
    exe = shutil.which("dot")
    if exe is None:
        log.warning("Graphviz 'dot' not found on PATH; skipping DOT rendering")
        return None
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, "-Tpng", "-o", str(out)], input=dot.encode("utf-8"), check=True)
    return out
