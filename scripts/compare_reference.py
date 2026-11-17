"""Benchmark treesfs against scikit-learn: test metric + fit time + predict time.

Writes the ``compare`` section of ``reports/results.md``, ``reports/benchmark.csv`` and
``reports/figures/fit_time_vs_sklearn.png``. scikit-learn is imported here, as a *reference
only*: nothing under ``src/treesfs`` depends on it for modelling.

Timing protocol: best-of-N wall time (``time.perf_counter``) per fit on the same float64 arrays
(scikit-learn casts to float32 internally). Forests use the same ``n_jobs`` on both sides.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as SkRandomForestRegressor
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor as SkDecisionTreeRegressor

from treesfs.config import Config, load_config
from treesfs.data import load_dataset, task_for
from treesfs.models.forest import forest_from_config
from treesfs.models.tree import tree_from_config
from treesfs.utils import ensure_dir, get_logger, set_seed, update_marked_section
from treesfs.viz import plot_benchmark

log = get_logger("compare_reference")


@dataclass(frozen=True)
class BenchRow:
    """One measurement: an implementation on one (dataset, model) case."""

    dataset: str
    model: str
    impl: str
    metric_name: str
    metric: float
    fit_s: float
    predict_s: float
    oob: float | None = None


def best_of(fn: Any, repeats: int) -> tuple[Any, float]:
    """Run ``fn()`` ``repeats`` times; return the last result and the fastest wall time."""
    best, result = float("inf"), None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return result, best


def bench_model(
    dataset: str,
    model_label: str,
    impl: str,
    make: Any,
    data: Any,
    task: str,
    repeats: int,
    predict_repeats: int = 3,
) -> BenchRow:
    """Fit ``make()`` on the train split, time fit and predict, score on the test split.

    Parameters
    ----------
    make : callable
        Returns a fresh unfitted estimator exposing ``fit``, ``predict`` and ``score``.
    data : object with ``X_train, y_train, X_test, y_test``
    task : {"classification", "regression"}
        Only selects the metric name.
    repeats : int
        Number of timed fits (best-of).
    """
    model, fit_s = best_of(lambda: make().fit(data.X_train, data.y_train), repeats)
    _, predict_s = best_of(lambda: model.predict(data.X_test), predict_repeats)
    oob = getattr(model, "oob_score_", None)
    return BenchRow(
        dataset=dataset,
        model=model_label,
        impl=impl,
        metric_name="accuracy" if task == "classification" else "R²",
        metric=float(model.score(data.X_test, data.y_test)),
        fit_s=float(fit_s),
        predict_s=float(predict_s),
        oob=None if oob is None else float(oob),
    )


def sklearn_tree(task: str, cfg: Config, depth: int | None):
    if task == "classification":
        return SkDecisionTreeClassifier(
            criterion=cfg.tree.criterion_classification, max_depth=depth, random_state=cfg.seed
        )
    return SkDecisionTreeRegressor(
        criterion=cfg.tree.criterion_regression, max_depth=depth, random_state=cfg.seed
    )


def sklearn_forest(task: str, cfg: Config, n_estimators: int):
    common = {
        "n_estimators": n_estimators,
        "max_depth": cfg.tree.max_depth,
        "oob_score": True,
        "n_jobs": cfg.forest.n_jobs,
        "random_state": cfg.seed,
    }
    if task == "classification":
        return SkRandomForestClassifier(criterion=cfg.tree.criterion_classification, **common)
    return SkRandomForestRegressor(criterion=cfg.tree.criterion_regression, **common)


def run_dataset(
    name: str, cfg: Config, data: Any, n_estimators: int, with_forest: bool, quick: bool
) -> list[BenchRow]:
    """All benchmark cases for one dataset (trees at two depths, and optionally the forest)."""
    task = task_for(name)
    tree_reps = 1 if quick else cfg.benchmark.tree_repeats
    forest_reps = 1 if quick else cfg.benchmark.forest_repeats
    rows: list[BenchRow] = []
    for depth in (cfg.benchmark.tree_depth, None):
        label = f"tree d={depth}" if depth is not None else "tree unlimited"
        for impl, make in (
            ("treesfs", lambda d=depth: tree_from_config(task, cfg.tree, cfg.seed, max_depth=d)),
            ("scikit-learn", lambda d=depth: sklearn_tree(task, cfg, d)),
        ):
            row = bench_model(name, label, impl, make, data, task, tree_reps)
            log.info(
                "%s %s %s: %s %.4f fit %.2fs",
                name,
                label,
                impl,
                row.metric_name,
                row.metric,
                row.fit_s,
            )
            rows.append(row)
    if with_forest:
        label = f"forest x{n_estimators}"
        for impl, make in (
            (
                "treesfs",
                lambda: forest_from_config(
                    task,
                    cfg.forest,
                    cfg.tree,
                    cfg.seed,
                    n_estimators=n_estimators,
                    oob_score=True,
                ),
            ),
            ("scikit-learn", lambda: sklearn_forest(task, cfg, n_estimators)),
        ):
            row = bench_model(name, label, impl, make, data, task, forest_reps)
            log.info(
                "%s %s %s: %s %.4f fit %.1fs",
                name,
                label,
                impl,
                row.metric_name,
                row.metric,
                row.fit_s,
            )
            rows.append(row)
    return rows


def markdown_table(rows: list[BenchRow]) -> str:
    """Pretty table with one line per (case, implementation) and a fit-time ratio column."""
    lines = [
        "| dataset | model | implementation | test metric | OOB | fit (s) | predict (s) | "
        "fit vs sklearn |",
        "|---|---|---|---|---|---|---|---|",
    ]
    ref_fit = {(r.dataset, r.model): r.fit_s for r in rows if r.impl == "scikit-learn"}
    for r in rows:
        ratio = (
            f"{r.fit_s / ref_fit[(r.dataset, r.model)]:.1f}x slower"
            if r.impl == "treesfs" and (r.dataset, r.model) in ref_fit
            else "-"
        )
        oob = "-" if r.oob is None else f"{r.oob:.4f}"
        lines.append(
            f"| {r.dataset} | {r.model} | {r.impl} | {r.metric_name} {r.metric:.4f} | {oob} | "
            f"{r.fit_s:.2f} | {r.predict_s:.4f} | {ratio} |"
        )
    return "\n".join(lines)


def environment_note(n_jobs: int | None) -> str:
    """Hardware/software context a reader needs to interpret the timings."""
    return (
        # platform.release() says "10" on Windows 11, so report the build number instead
        f"Measured on {platform.system()} (build {platform.version()}), {os.cpu_count()} "
        f"logical cores, "
        f"Python {platform.python_version()}, NumPy {np.__version__}, scikit-learn "
        f"{sklearn.__version__}. Same float64 train/test arrays for both implementations "
        f"(scikit-learn casts to float32 internally). Best-of-N wall time; forests use "
        f"`n_jobs={n_jobs}` on both sides. Timings vary with machine load; the ratios, not the "
        f"seconds, are the portable result."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--datasets", nargs="+", default=["adult", "california"], choices=["adult", "california"]
    )
    ap.add_argument("--n-estimators", type=int, default=None, help="override forest.n_estimators")
    ap.add_argument("--no-forest", action="store_true", help="benchmark single trees only")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="smoke test: 2000 training rows, 10 trees, one repeat, writes nothing",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    n_estimators = 10 if args.quick else (args.n_estimators or cfg.forest.n_estimators)

    rows: list[BenchRow] = []
    for name in args.datasets:
        data = load_dataset(name, cfg)
        if args.quick:
            data = type(data)(
                **{**data.__dict__, "X_train": data.X_train[:2000], "y_train": data.y_train[:2000]}
            )
        rows += run_dataset(name, cfg, data, n_estimators, not args.no_forest, args.quick)

    table = markdown_table(rows)
    print(table)
    if args.quick:
        print("quick mode: nothing written")
        return

    results_dir = ensure_dir(cfg.paths.results.parent)
    with (results_dir / "benchmark.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(
            ["dataset", "model", "impl", "metric_name", "metric", "oob", "fit_s", "predict_s"]
        )
        for r in rows:
            writer.writerow(
                [
                    r.dataset,
                    r.model,
                    r.impl,
                    r.metric_name,
                    f"{r.metric:.6f}",
                    "" if r.oob is None else f"{r.oob:.6f}",
                    f"{r.fit_s:.4f}",
                    f"{r.predict_s:.5f}",
                ]
            )

    body = (
        "## treesfs vs scikit-learn\n\n"
        + table
        + "\n\n"
        + environment_note(cfg.forest.n_jobs)
        + "\n\nNotes: treesfs `predict` is single-threaded while scikit-learn's forest predicts "
        "with `n_jobs` threads, which explains most of the forest predict-time gap. Metric "
        "differences between the two implementations on single trees come from tie-breaking "
        "(DESIGN.md §9.3)."
    )
    update_marked_section(cfg.paths.results, "compare", body)

    cases = sorted(
        {(r.dataset, r.model) for r in rows},
        key=lambda c: [(x.dataset, x.model) for x in rows].index(c),
    )
    by = {(r.dataset, r.model, r.impl): r.fit_s for r in rows}
    fig = plot_benchmark(
        [f"{d}\n{m}" for d, m in cases],
        [by[(d, m, "treesfs")] for d, m in cases],
        [by[(d, m, "scikit-learn")] for d, m in cases],
        ensure_dir(cfg.paths.figures) / "fit_time_vs_sklearn.png",
    )
    print(f"updated {cfg.paths.results}, {results_dir / 'benchmark.csv'} and {fig}")


if __name__ == "__main__":
    main()
