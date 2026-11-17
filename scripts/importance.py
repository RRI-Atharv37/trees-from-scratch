"""Impurity vs permutation importance for a forest: CSV + figure + section of results.md."""

from __future__ import annotations

import argparse
import csv

import numpy as np

from treesfs.config import load_config
from treesfs.data import load_dataset, task_for
from treesfs.models.forest import forest_from_config
from treesfs.permutation_importance import permutation_importance
from treesfs.utils import ensure_dir, get_logger, set_seed, timer, update_marked_section
from treesfs.viz import plot_importance_comparison

log = get_logger("importance")


def _ranks(a: np.ndarray) -> np.ndarray:
    """Rank of each entry, 0 = largest (ties broken by position; fine for continuous scores)."""
    order = np.argsort(-a, kind="stable")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a))
    return ranks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None, choices=["adult", "california"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    dataset = args.dataset or cfg.importance.dataset
    task = task_for(dataset)
    metric = "accuracy" if task == "classification" else "R²"
    data = load_dataset(dataset, cfg)

    forest = forest_from_config(task, cfg.forest, cfg.tree, cfg.seed, oob_score=False)
    log.info("Fitting %d trees on %s", forest.n_estimators, dataset)
    with timer() as t_fit:
        forest.fit(data.X_train, data.y_train)
    log.info(
        "forest fit %.1fs; test %s %.4f",
        t_fit.elapsed,
        metric,
        forest.score(data.X_test, data.y_test),
    )

    log.info(
        "Permutation importance: %d features x %d repeats on up to %s held-out rows",
        data.X_test.shape[1],
        cfg.importance.n_repeats,
        cfg.importance.max_rows or "all",
    )
    with timer() as t_perm:
        res = permutation_importance(
            forest,
            data.X_test,
            data.y_test,
            n_repeats=cfg.importance.n_repeats,
            random_state=cfg.seed,
            max_rows=cfg.importance.max_rows,
            n_jobs=cfg.importance.n_jobs,
        )
    log.info(
        "permutation importance %.1fs (baseline %s %.4f)",
        t_perm.elapsed,
        metric,
        res.baseline_score,
    )

    names = data.feature_names
    impurity = forest.feature_importances_
    rank_imp, rank_perm = _ranks(impurity), _ranks(res.importances_mean)
    spearman = float(np.corrcoef(rank_imp, rank_perm)[0, 1])

    out_dir = ensure_dir(cfg.paths.results.parent)
    csv_path = out_dir / f"importance_{dataset}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(
            ["feature", "impurity", "perm_mean", "perm_std", "impurity_rank", "perm_rank"]
        )
        for j in np.argsort(-impurity, kind="stable"):
            writer.writerow(
                [
                    names[j],
                    f"{impurity[j]:.6f}",
                    f"{res.importances_mean[j]:.6f}",
                    f"{res.importances_std[j]:.6f}",
                    int(rank_imp[j]) + 1,
                    int(rank_perm[j]) + 1,
                ]
            )
    fig = plot_importance_comparison(
        names,
        impurity,
        res.importances_mean,
        res.importances_std,
        ensure_dir(cfg.paths.figures) / f"importance_{dataset}.png",
        top_k=cfg.importance.top_k,
        metric_name=metric,
    )

    top = np.argsort(-impurity, kind="stable")[: cfg.importance.top_k]
    body = (
        f"## Feature importance, {dataset} forest ({forest.n_estimators} trees)\n\n"
        f"Permutation importance on {res.n_rows} held-out rows, {cfg.importance.n_repeats} "
        f"repeats; baseline test {metric} {res.baseline_score:.4f}. "
        f"Spearman correlation of the two rankings over all {len(names)} features: "
        f"{spearman:.2f}.\n\n"
        f"| feature | impurity (rank) | permutation drop (rank) |\n|---|---|---|\n"
        + "\n".join(
            f"| {names[j]} | {impurity[j]:.4f} ({int(rank_imp[j]) + 1}) | "
            f"{res.importances_mean[j]:.4f} ± {res.importances_std[j]:.4f} "
            f"({int(rank_perm[j]) + 1}) |"
            for j in top
        )
    )
    update_marked_section(cfg.paths.results, f"importance_{dataset}", body)
    print(body)
    print(f"wrote {csv_path} and {fig}")


if __name__ == "__main__":
    main()
