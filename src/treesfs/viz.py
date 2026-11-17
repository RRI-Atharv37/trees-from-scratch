"""Figures (matplotlib, Agg backend, no pyplot global state), saved at dpi=150.

Figures are built with :class:`matplotlib.figure.Figure` attached to an ``FigureCanvasAgg``, so
they render headlessly on every OS and nothing is left open afterwards.
Colors follow the project palette: series 1 blue, series 2 orange, on a light surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from treesfs.export import node_fill, node_label, visible_nodes
from treesfs.models.tree import TREE_LEAF, BaseDecisionTree
from treesfs.utils import ensure_dir

DPI = 150
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
TRAIN_COLOR = "#2a78d6"
TEST_COLOR = "#eb6834"


def _new_figure(figsize: tuple[float, float]) -> Figure:
    fig = Figure(figsize=figsize, facecolor=SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def _save(fig: Figure, out_path: Path | str) -> Path:
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    return out


def plot_depth_sweep(
    depths: Sequence[int | None],
    train_scores: Sequence[float],
    test_scores: Sequence[float],
    out_path: Path | str,
    best_depth: int | None = None,
    metric_name: str = "accuracy",
) -> Path:
    """Hero figure: train vs test score as a function of ``max_depth``.

    ``None`` (unlimited) is drawn one step past the last finite depth and labelled "∞", where
    the train-test gap is annotated. A dashed vertical line marks ``best_depth``.

    Parameters
    ----------
    depths : sequence of int or None, length D
    train_scores, test_scores : sequence of float, length D
    out_path : Path or str
    best_depth : int, optional
        Depth with the best test score.
    metric_name : str
    """
    finite = [d for d in depths if d is not None]
    step = (finite[-1] - finite[-2]) if len(finite) >= 2 else 1
    x_inf = finite[-1] + step if finite else 1
    xs = np.array([x_inf if d is None else d for d in depths], dtype=float)
    tr, te = np.asarray(train_scores, dtype=float), np.asarray(test_scores, dtype=float)

    fig = _new_figure((10, 5.5))
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY)

    ax.plot(xs, tr, color=TRAIN_COLOR, marker="o", markersize=5, linewidth=2, label="train")
    ax.plot(xs, te, color=TEST_COLOR, marker="o", markersize=5, linewidth=2, label="test")

    if best_depth is not None:
        i = int(np.flatnonzero(xs == best_depth)[0])
        ax.axvline(best_depth, color=INK_SECONDARY, linestyle="--", linewidth=1)
        ax.annotate(
            f"best test depth = {best_depth}\n({metric_name} {te[i]:.3f})",
            xy=(best_depth, te[i]),
            xytext=(best_depth + 0.6, te[i] - 0.06),
            color=INK,
            fontsize=9,
            va="top",
        )
    if None in depths:
        j = list(depths).index(None)
        gap = tr[j] - te[j]
        xa = x_inf + step * 0.35
        ax.annotate(
            "",
            xy=(xa, te[j]),
            xytext=(xa, tr[j]),
            arrowprops={"arrowstyle": "<->", "color": INK_SECONDARY, "linewidth": 1.2},
        )
        ax.text(
            xa + step * 0.12,
            (tr[j] + te[j]) / 2,
            f"gap\n{gap:.3f}",
            color=INK,
            fontsize=9,
            va="center",
        )
        ax.set_xlim(xs.min() - 0.8, x_inf + step * 1.6)
    ticks = list(xs)
    labels = ["∞" if d is None else str(d) for d in depths]
    ax.set_xticks(ticks, labels)
    ax.set_xlabel("max_depth", color=INK_SECONDARY)
    ax.set_ylabel(metric_name, color=INK_SECONDARY)
    ax.set_title("Adult: deeper trees memorize", color=INK, fontsize=14, loc="left")
    leg = ax.legend(loc="lower right", frameon=False)
    for text in leg.get_texts():
        text.set_color(INK)
    return _save(fig, out_path)


def _blend_on_white(color: str, alpha: float) -> tuple[float, float, float]:
    """Opaque RGB equal to ``color`` at ``alpha`` over white (so edges never show through)."""
    r, g, b = to_rgb(color)
    return (1 - alpha + alpha * r, 1 - alpha + alpha * g, 1 - alpha + alpha * b)


def plot_tree(
    tree: BaseDecisionTree,
    feature_names: Sequence[str] | None,
    class_names: Sequence[str] | None,
    out_path: Path | str,
    max_depth: int = 3,
) -> Path:
    """Draw the top of a fitted tree with pure matplotlib (no Graphviz needed).

    Layout: leaves (and nodes cut at ``max_depth``) take consecutive x positions left to right;
    each parent sits at the mean x of its children; ``y = -depth``. Boxes carry the same label
    as the DOT export, filled by majority class with alpha growing with purity; edges are
    labelled True (``<=``) / False.

    Parameters
    ----------
    tree : fitted estimator
    feature_names, class_names : sequence of str or None
    out_path : Path or str
    max_depth : int
        Number of split levels to draw.
    """
    if not hasattr(tree, "tree_"):
        raise RuntimeError(f"{type(tree).__name__} is not fitted; call fit() first")
    t = tree.tree_
    visible = visible_nodes(tree, max_depth)  # preorder
    shown = {node for node, _, _ in visible}
    children = {
        node: (int(t.children_left[node]), int(t.children_right[node]))
        for node in shown
        if t.children_left[node] != TREE_LEAF and int(t.children_left[node]) in shown
    }
    pos_x: dict[int, float] = {}
    n_leaf = 0
    for node, _, _ in visible:  # preorder visits leaves left to right
        if node not in children:
            pos_x[node] = float(n_leaf)
            n_leaf += 1
    for node, _, _ in reversed(visible):  # children precede parents in the reversed order
        if node in children:
            pos_x[node] = float(np.mean([pos_x[c] for c in children[node]]))
    depth_of = {node: depth for node, depth, _ in visible}
    max_d = max(depth_of.values())

    fig = _new_figure((16, 7))
    ax = fig.add_subplot()
    ax.set_axis_off()
    ax.set_xlim(-0.6, n_leaf - 0.4)
    ax.set_ylim(-max_d - 0.55, 0.55)
    for parent, (left, right) in children.items():
        for child, side in ((left, "True"), (right, "False")):
            x0, y0 = pos_x[parent], -depth_of[parent]
            x1, y1 = pos_x[child], -depth_of[child]
            ax.plot([x0, x1], [y0, y1], color=INK_SECONDARY, linewidth=1, zorder=1)
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                side,
                fontsize=8,
                color=INK_SECONDARY,
                ha="center",
                va="center",
                zorder=2,
                bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none"},
            )
    for node, depth, _ in visible:
        color, alpha = node_fill(tree, node)
        rows = node_label(tree, node, feature_names, class_names)
        ax.text(
            pos_x[node],
            -depth,
            "\n".join(rows),
            fontsize=7.5,
            color=INK,
            ha="center",
            va="center",
            zorder=3,
            bbox={
                "boxstyle": "round,pad=0.35",
                "fc": _blend_on_white(color, alpha),
                "ec": INK_SECONDARY,
                "linewidth": 0.8,
            },
        )
    fig.suptitle("Decision tree (top levels)", color=INK, fontsize=13, x=0.01, ha="left")
    return _save(fig, out_path)


def _style_axes(ax) -> None:  # type: ignore[no-untyped-def]
    """Recessive grid and spines shared by the bar/line figures."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY)


def plot_n_trees_sweep(
    grid: Sequence[int],
    oob_scores: Sequence[float],
    test_scores: Sequence[float],
    out_path: Path | str,
    single_tree_score: float | None = None,
    metric_name: str = "accuracy",
    title: str = "Adult: more trees help, then plateau",
) -> Path:
    """Forest score vs number of trees: OOB (free, from training rows) against held-out test.

    Parameters
    ----------
    grid : sequence of int
        Number of trees ``k`` at each point (drawn on a log x axis).
    oob_scores, test_scores : sequence of float
        Score of the first-``k``-trees sub-forest on its out-of-bag rows / on the test set.
    single_tree_score : float, optional
        Test score of one unlimited-depth tree, drawn as a dashed reference line.
    """
    ks = np.asarray(grid, dtype=float)
    fig = _new_figure((10, 5.5))
    ax = fig.add_subplot()
    _style_axes(ax)
    ax.plot(ks, oob_scores, color=TRAIN_COLOR, marker="o", markersize=5, linewidth=2, label="OOB")
    ax.plot(ks, test_scores, color=TEST_COLOR, marker="o", markersize=5, linewidth=2, label="test")
    if single_tree_score is not None:
        ax.axhline(single_tree_score, color=INK_SECONDARY, linestyle="--", linewidth=1)
        ax.annotate(
            f"one unlimited tree ({single_tree_score:.3f})",
            xy=(ks[-1], single_tree_score),
            xytext=(-6, -14),
            textcoords="offset points",
            ha="right",
            color=INK,
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(int(k)) for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("number of trees", color=INK_SECONDARY)
    ax.set_ylabel(metric_name, color=INK_SECONDARY)
    ax.set_title(title, color=INK, fontsize=14, loc="left")
    # mid-right is empty: the curves climb to the top right and the baseline label sits at the
    # bottom right
    leg = ax.legend(loc="center right", frameon=False)
    for text in leg.get_texts():
        text.set_color(INK)
    return _save(fig, out_path)


def plot_importance_comparison(
    names: Sequence[str],
    impurity: Sequence[float],
    perm_mean: Sequence[float],
    perm_std: Sequence[float],
    out_path: Path | str,
    top_k: int = 15,
    metric_name: str = "accuracy",
) -> Path:
    """Impurity importance (training data) next to permutation importance (held-out data).

    Features are the ``top_k`` by impurity importance, in that order on both panels, so
    disagreements between the two rankings are visible as bars that do not shrink together.
    The scales differ (a share of total impurity decrease vs a drop in ``metric_name``), so
    each panel has its own axis.
    """
    imp = np.asarray(impurity, dtype=float)
    pm = np.asarray(perm_mean, dtype=float)
    ps = np.asarray(perm_std, dtype=float)
    order = np.argsort(-imp, kind="stable")[:top_k]
    labels = [names[i] for i in order]
    y = np.arange(len(order))[::-1]  # largest at the top

    fig = _new_figure((13, 6.5))
    ax1, ax2 = fig.subplots(1, 2, sharey=True)
    for ax in (ax1, ax2):
        _style_axes(ax)
        ax.grid(False, axis="y")
    ax1.barh(y, imp[order], color=TRAIN_COLOR, height=0.62)
    ax2.barh(
        y,
        pm[order],
        xerr=ps[order],
        color=TEST_COLOR,
        height=0.62,
        error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 2},
    )
    ax1.set_yticks(y, labels)
    ax1.tick_params(axis="y", colors=INK)
    ax1.set_xlabel("share of total impurity decrease (training data)", color=INK_SECONDARY)
    ax2.set_xlabel(
        f"drop in held-out {metric_name} when shuffled (mean ± std)", color=INK_SECONDARY
    )
    ax1.set_title("Impurity importance", color=INK, fontsize=12, loc="left")
    ax2.set_title("Permutation importance", color=INK, fontsize=12, loc="left")
    fig.suptitle(
        "Adult: what the forest used vs what it needs", color=INK, fontsize=14, x=0.01, ha="left"
    )
    return _save(fig, out_path)


def plot_benchmark(
    labels: Sequence[str],
    mine: Sequence[float],
    reference: Sequence[float],
    out_path: Path | str,
    ylabel: str = "fit time (s, log scale)",
    title: str = "treesfs vs scikit-learn: fit time",
) -> Path:
    """Grouped bars (treesfs vs scikit-learn) on a log axis, with the value above each bar."""
    x = np.arange(len(labels), dtype=float)
    w = 0.38
    fig = _new_figure((11, 5.5))
    ax = fig.add_subplot()
    _style_axes(ax)
    ax.grid(False, axis="x")
    for xs, vals, color, name in (
        (x - w / 2, mine, TRAIN_COLOR, "treesfs"),
        (x + w / 2, reference, TEST_COLOR, "scikit-learn"),
    ):
        ax.bar(xs, vals, width=w - 0.04, color=color, label=name)
        for xi, v in zip(xs, vals, strict=True):
            label = f"{v:.0f}" if v >= 10 else f"{v:.2g}"  # never scientific notation
            ax.text(xi, v * 1.12, label, ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.set_title(title, color=INK, fontsize=14, loc="left")
    top = max(max(mine), max(reference))
    ax.set_ylim(min(min(mine), min(reference)) / 2, top * 4)
    leg = ax.legend(loc="upper left", frameon=False)
    for text in leg.get_texts():
        text.set_color(INK)
    return _save(fig, out_path)
