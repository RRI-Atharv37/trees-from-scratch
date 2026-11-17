"""Evaluate a pickled tree/forest and rewrite its section of reports/results.md (idempotent)."""

from __future__ import annotations

import argparse
import pickle

from treesfs.config import load_config
from treesfs.data import load_dataset
from treesfs.metrics import accuracy, mae, mse, r2_score
from treesfs.models.forest import BaseForest
from treesfs.utils import get_logger, timer, update_marked_section

log = get_logger("evaluate")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", choices=["adult", "california"], required=True)
    ap.add_argument("--model", choices=["tree", "forest"], default="tree")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pkl = cfg.paths.artifacts / f"{args.dataset}_{args.model}.pkl"
    if not pkl.exists():
        raise SystemExit(
            f"{pkl} not found; run scripts/train.py --dataset {args.dataset} --model {args.model}"
        )
    with pkl.open("rb") as fh:
        # Our own artifact from train.py. NEVER unpickle a file you did not create: a pickle can
        # execute arbitrary code on load (see SECURITY.md).
        blob = pickle.load(fh)
    model = blob["model"]
    data = load_dataset(args.dataset, cfg)
    is_forest = isinstance(model, BaseForest)
    kind = "RandomForest" if is_forest else "DecisionTree"

    with timer() as t_pred:
        pred_te = model.predict(data.X_test)
    pred_tr = model.predict(data.X_train)
    if args.dataset == "adult":
        rows = [
            ("train accuracy", f"{accuracy(data.y_train, pred_tr):.4f}"),
            ("test accuracy", f"{accuracy(data.y_test, pred_te):.4f}"),
        ]
        title = f"Adult (income > 50K), treesfs {kind}Classifier"
    else:
        rows = [
            ("train R²", f"{r2_score(data.y_train, pred_tr):.4f}"),
            ("test R²", f"{r2_score(data.y_test, pred_te):.4f}"),
            ("test RMSE (x$100k)", f"{mse(data.y_test, pred_te) ** 0.5:.4f}"),
            ("test MAE (x$100k)", f"{mae(data.y_test, pred_te):.4f}"),
        ]
        title = f"California housing, treesfs {kind}Regressor"
    if is_forest:
        depths = [tree.get_depth() for tree in model.estimators_]
        leaves = [tree.get_n_leaves() for tree in model.estimators_]
        if hasattr(model, "oob_score_"):
            rows.append(
                (
                    "OOB " + ("accuracy" if args.dataset == "adult" else "R²"),
                    f"{model.oob_score_:.4f}",
                )
            )
        rows += [
            ("trees", str(len(model.estimators_))),
            ("max_depth (param)", str(model.max_depth)),
            ("depth (mean / max)", f"{sum(depths) / len(depths):.1f} / {max(depths)}"),
            ("leaves per tree (mean)", f"{sum(leaves) / len(leaves):.0f}"),
        ]
    else:
        rows += [
            ("max_depth (param)", str(model.max_depth)),
            ("depth (actual)", str(model.get_depth())),
            ("leaves", str(model.get_n_leaves())),
            ("nodes", str(model.tree_.node_count)),
        ]
    rows += [
        ("features", str(model.n_features_in_)),
        ("fit time (s)", f"{blob['fit_seconds']:.2f}"),
        (f"predict time, {len(data.X_test)} test rows (s)", f"{t_pred.elapsed:.4f}"),
    ]
    body = f"## {title}\n\n| metric | value |\n|---|---|\n"
    body += "\n".join(f"| {k} | {v} |" for k, v in rows)
    key = args.dataset if args.model == "tree" else f"{args.dataset}_forest"
    update_marked_section(cfg.paths.results, key, body)
    print("\n".join(f"{k}: {v}" for k, v in rows))
    print(f"updated {cfg.paths.results}")


if __name__ == "__main__":
    main()
