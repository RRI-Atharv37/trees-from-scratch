"""Sweep max_depth on Adult: CSV + hero figure + the depth-3 tree as PNG/DOT/text."""

from __future__ import annotations

import argparse
import csv

from treesfs.config import load_config
from treesfs.data import load_dataset, task_for
from treesfs.export import export_graphviz, export_text
from treesfs.metrics import accuracy
from treesfs.models.tree import tree_from_config
from treesfs.utils import ensure_dir, get_logger, set_seed, timer, write_text_lf
from treesfs.viz import plot_depth_sweep, plot_tree

log = get_logger("depth_sweep")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None, choices=["adult"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    dataset = args.dataset or cfg.depth_sweep.dataset
    data = load_dataset(dataset, cfg)
    task = task_for(dataset)
    class_names = ["<=50K", ">50K"]

    records = []
    for depth in cfg.depth_sweep.depths:
        model = tree_from_config(task, cfg.tree, cfg.seed, max_depth=depth)
        with timer() as t:
            model.fit(data.X_train, data.y_train)
        rec = {
            "max_depth": "None" if depth is None else depth,
            "train_acc": accuracy(data.y_train, model.predict(data.X_train)),
            "test_acc": accuracy(data.y_test, model.predict(data.X_test)),
            "fit_seconds": t.elapsed,
            "actual_depth": model.get_depth(),
            "leaves": model.get_n_leaves(),
        }
        records.append(rec)
        log.info(
            "max_depth=%s train=%.4f test=%.4f depth=%d leaves=%d fit=%.1fs",
            rec["max_depth"],
            rec["train_acc"],
            rec["test_acc"],
            rec["actual_depth"],
            rec["leaves"],
            t.elapsed,
        )

    csv_path = cfg.paths.results.parent / "depth_sweep.csv"
    ensure_dir(csv_path.parent)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in rec.items()}
            )

    finite = [r for r in records if r["max_depth"] != "None"]
    best = max(finite, key=lambda r: r["test_acc"])  # first best on ties = shallowest
    fig_dir = ensure_dir(cfg.paths.figures)
    plot_depth_sweep(
        [None if r["max_depth"] == "None" else int(r["max_depth"]) for r in records],
        [r["train_acc"] for r in records],
        [r["test_acc"] for r in records],
        fig_dir / "depth_vs_accuracy.png",
        best_depth=int(best["max_depth"]),
    )
    log.info("best finite depth %s test acc %.4f", best["max_depth"], best["test_acc"])

    k = cfg.export.small_tree_depth
    small = tree_from_config(task, cfg.tree, cfg.seed, max_depth=k).fit(data.X_train, data.y_train)
    plot_tree(small, data.feature_names, class_names, fig_dir / "small_tree.png", max_depth=k)
    write_text_lf(
        fig_dir / "small_tree.dot", export_graphviz(small, data.feature_names, class_names)
    )
    write_text_lf(fig_dir / "small_tree.txt", export_text(small, data.feature_names, class_names))
    print(f"wrote {csv_path} and figures in {fig_dir}")


if __name__ == "__main__":
    main()
