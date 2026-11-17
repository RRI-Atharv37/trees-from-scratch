"""Score vs number of trees: one forest, staged scoring -> CSV + figure (OOB vs test)."""

from __future__ import annotations

import argparse
import csv

from treesfs.config import load_config
from treesfs.data import load_dataset, task_for
from treesfs.models.forest import forest_from_config
from treesfs.models.tree import tree_from_config
from treesfs.utils import ensure_dir, get_logger, set_seed, timer
from treesfs.viz import plot_n_trees_sweep

log = get_logger("n_trees_sweep")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None, choices=["adult", "california"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    dataset = args.dataset or cfg.n_trees_sweep.dataset
    task = task_for(dataset)
    metric = "accuracy" if task == "classification" else "R²"
    data = load_dataset(dataset, cfg)
    grid = list(cfg.n_trees_sweep.grid)

    forest = forest_from_config(
        task, cfg.forest, cfg.tree, cfg.seed, n_estimators=max(grid), oob_score=False
    )
    log.info("Fitting %d trees on %s", max(grid), dataset)
    with timer() as t:
        forest.fit(data.X_train, data.y_train)
    log.info("forest fit %.1fs", t.elapsed)

    # Staged scoring: one pass over the trees yields the whole curve (see forest.staged_scores).
    oob = forest.staged_scores(data.X_train, data.y_train, grid, oob=True)
    test = forest.staged_scores(data.X_test, data.y_test, grid)
    for k, o, s in zip(grid, oob, test, strict=True):
        log.info("trees=%3d  OOB %s=%.4f  test %s=%.4f", k, metric, o, metric, s)

    log.info("Fitting one unlimited-depth tree as the baseline")
    single = tree_from_config(task, cfg.tree, cfg.seed).fit(data.X_train, data.y_train)
    single_score = single.score(data.X_test, data.y_test)

    out_dir = ensure_dir(cfg.paths.results.parent)
    csv_path = out_dir / f"n_trees_sweep_{dataset}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["n_trees", f"oob_{metric}", f"test_{metric}"])
        for k, o, s in zip(grid, oob, test, strict=True):
            writer.writerow([k, f"{o:.6f}", f"{s:.6f}"])
    fig = plot_n_trees_sweep(
        grid,
        oob,
        test,
        ensure_dir(cfg.paths.figures) / f"n_trees_vs_score_{dataset}.png",
        single_tree_score=single_score,
        metric_name=metric,
        title=(
            "Adult: more trees help, then plateau"
            if dataset == "adult"
            else "California: more trees help, then plateau"
        ),
    )
    print(f"single unlimited tree test {metric}: {single_score:.4f}")
    print(f"{max(grid)}-tree forest test {metric}: {test[-1]:.4f} (OOB {oob[-1]:.4f})")
    print(f"wrote {csv_path} and {fig}")


if __name__ == "__main__":
    main()
