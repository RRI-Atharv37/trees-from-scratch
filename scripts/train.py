"""Fit a treesfs decision tree or random forest on Adult or California housing and pickle it."""

from __future__ import annotations

import argparse
import pickle

from treesfs.config import load_config
from treesfs.data import load_dataset, task_for
from treesfs.models.forest import forest_from_config
from treesfs.models.tree import tree_from_config
from treesfs.utils import ensure_dir, get_logger, set_seed, timer

log = get_logger("train")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", choices=["adult", "california"], required=True)
    ap.add_argument("--model", choices=["tree", "forest"], default="tree")
    ap.add_argument("--max-depth", type=int, default=None, help="override tree.max_depth")
    ap.add_argument("--n-estimators", type=int, default=None, help="override forest.n_estimators")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    data = load_dataset(args.dataset, cfg)
    task = task_for(args.dataset)
    overrides = {} if args.max_depth is None else {"max_depth": args.max_depth}
    if args.model == "forest":
        if args.n_estimators is not None:
            overrides["n_estimators"] = args.n_estimators
        model = forest_from_config(task, cfg.forest, cfg.tree, cfg.seed, **overrides)
    else:
        model = tree_from_config(task, cfg.tree, cfg.seed, **overrides)
    log.info("Fitting %s on %s (%d x %d)", model, args.dataset, *data.X_train.shape)
    with timer() as t:
        model.fit(data.X_train, data.y_train)
    if args.model == "forest":
        depths = [tree.get_depth() for tree in model.estimators_]
        log.info(
            "fit %.2fs | %d trees | depth mean %.1f max %d | OOB score %s",
            t.elapsed,
            len(model.estimators_),
            sum(depths) / len(depths),
            max(depths),
            f"{model.oob_score_:.4f}" if hasattr(model, "oob_score_") else "n/a",
        )
    else:
        log.info(
            "fit %.2fs | depth %d | leaves %d | nodes %d",
            t.elapsed,
            model.get_depth(),
            model.get_n_leaves(),
            model.tree_.node_count,
        )
    out = ensure_dir(cfg.paths.artifacts) / f"{args.dataset}_{args.model}.pkl"
    with out.open("wb") as fh:
        pickle.dump(
            {"model": model, "fit_seconds": t.elapsed, "feature_names": data.feature_names}, fh
        )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
