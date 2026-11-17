# Learning log

## 2026-11-17, Day 1: data, preprocessing, criteria

- The Adult files have more quirks than I expected: the test file's first line is junk (`|1x3 Cross validator`), test labels end in a dot, `?` is the missing marker, and tokens are separated by `", "`. I wrote a six-row train fixture and a four-row test fixture that contain every one of these so the parser is tested without the network.
- Missing values: I kept `?` as an explicit `"missing"` category instead of dropping rows. 2,399 train rows have at least one `?`, and dropping them would bias the data. A tree can now split on missingness directly.
- With one-hot columns capped at 16 (`max_categories`), only `native-country` collapses to 15 + `other`, so Adult has 82 features (6 numeric + 76 one-hot) instead of 108.
- Entropy: `0 * log2(0)` is NaN, so I mask instead of adding an epsilon. Gini/entropy tests use hand-computed values (`[2,3,5]` gives Gini 0.62).
- Gotcha: `ruff format .` in recent versions also reformats code blocks inside Markdown files (it rewrote the code in `docs/DESIGN.md`). I restored them and excluded `*.md` in `pyproject.toml`.
- Windows blocked `pytest.exe` (application control), so I run `uv run python -m pytest`.

## 2026-11-17, Day 2: the splitter and the tree

- Why the O(n) scan works: sort one column once, and every threshold is a prefix. Class counts (or target sums) of the left child are prefix sums, the right child is total minus prefix, and the Gini/entropy/variance cost is a function of those sums only. One `cumsum` and one vectorized formula give all `m-1` costs.
- The fast scan matched a brute-force repartitioning oracle on 900 random cases (300 per criterion) with half the features heavily tied. I compare *costs*, never thresholds, because equal-cost thresholds are legitimate ties.
- Regression centers the node's targets before the cumsum (`y - mean`), which makes `Q_t = sum(ys**2)` a per-node scalar and avoids cancellation when targets sit at 1e4.
- Ties are everywhere in classification trees, because the cost is a function of small integer counts. sklearn breaks cross-feature ties with a random permutation; I scan features in order. So a structural comparison has to be tie-aware: at each node either the split is identical or it has exactly the same cost, and I stop descending at the first tie. Regression on continuous data matched sklearn array-for-array.
- Even sklearn disagrees with itself: with `min_impurity_decrease=0.005` its node count is 49 or 53 depending on `random_state`. My test therefore checks that my count is one sklearn can produce, and that every kept split really clears the threshold.
- sklearn casts `X` to float32. Without the same cast in the tests, thresholds differ by about 1e-8 and ties shift.
- Leaf detection uses `impurity <= EPS`, not `== 0`, and the `min_impurity_decrease` check uses sklearn's `+ EPS` tolerance.
- Handoff slips I hit: `FEATURE_THRESHOLD = 1e-7` means two adjacent floats near 1.0 are *not* splittable, so the rounding-guard test uses values near 4e9 where adjacent floats are far enough apart. And sklearn's `r2_score` returns 1.0 (not 0.0) for a perfect prediction on constant targets; I follow sklearn.

## 2026-11-17, Day 3: export, plots and the depth sweep

- Adult, `max_depth` sweep (82 features, my implementation): test accuracy rises to 0.8601 at depth 10 and then falls to 0.8117 at unlimited depth (49 levels, 4,679 leaves) while train accuracy goes to about 1.0. That is a 0.19 train-test gap: the right half of the overfitting U, which is the guaranteed consequence of impurity never increasing under a split.
- California: an unlimited regressor gets test R² 0.582 (depth 34, 15.8k leaves), but `max_depth=10` gets 0.659. Same lesson on a different task.
- `export_text` follows sklearn's layout so the output is familiar. `plot_tree` is pure matplotlib (leaves left to right, parents at the mean x of their children); I blend the class color with white so the edges do not show through the translucent boxes.
- Fit time is dominated by Python/NumPy call overhead: roughly one `argsort` and one `cumsum` per (node, feature). Adult unlimited takes 10 to 28 s on this machine depending on load, versus about 1.6 s for sklearn's Cython.

## 2026-11-17, Day 4: random forest and out-of-bag scores

- A forest is mostly bookkeeping on top of the tree: bootstrap the rows, call the tree's private `_fit` with the forest's class count (so a bootstrap that misses a class still gives aligned `(n_nodes, K)` values and a plain mean works), average the leaf values.
- Reproducibility under parallelism: one `SeedSequence`, spawned once per tree, and each child spawned again into a bootstrap stream and a feature-subsampling stream. A tree depends only on its own seed, so `n_jobs=1` and `n_jobs=2` give bit-identical trees (there is a test). The OOB rows of tree `i` are recomputed from its bootstrap seed instead of stored.
- Gotcha: `SeedSequence.spawn()` mutates its parent's counter, so refitting with the same `SeedSequence` object silently gave a different forest until I copied it before spawning.
- OOB works: Adult OOB accuracy 0.8553 vs test 0.8528; California OOB R² 0.8063 vs test 0.8001. A row is out-of-bag for about 36.8% of trees, but with only 5 trees about 7% of rows have no OOB vote at all; I exclude them and warn, as sklearn does.
- My own OOB test is a slow per-row Python loop that averages only the trees that never saw the row. It matches the vectorized aggregation exactly, which is a better check than comparing against sklearn's different random stream.
- Test slip: I asserted that a forest of 3 identical trees equals the single tree *exactly*. `(p + p + p) / 3` differs from `p` by one ulp, so that is an `allclose`, not an `array_equal`. The same test caught a real bug: `predict()` on an unfitted forest raised `AttributeError` (it touched `classes_` first) instead of the intended `RuntimeError`.
- Speed: 24 trees take 237 s serial and 28 s with 24 workers, but 8 workers already get 7.7x; it saturates around 8.5x on this 24-thread machine. Timings are noisy even on a quiet machine: the same 100-tree Adult forest took 161 s (tests running), 85 s (benchmark) and 151 s (a later plain `train.py` run). I could not pin the 2x spread on one cause, so I quote ratios from one best-of-N benchmark run and say the seconds vary.

## 2026-11-17, Day 5: permutation importance, sweeps, benchmarks and the legal pass

- **More trees help, then plateau.** One staged pass over a 200-tree forest gives the whole OOB and test curve (running sums of leaf values), so the sweep costs one fit, not fourteen. Test accuracy goes 0.810 (1 tree) to 0.855 (200 trees), flat after about 50.
- **Surprise: the forest does not beat a tuned tree on Adult.** The depth-10 tree gets 0.860; the 100-tree unlimited forest 0.853, and sklearn's forest 0.852. I had assumed otherwise and wrote it into a first README draft; re-reading the numbers caught it. On California the forest wins clearly (R² 0.800 vs 0.659 for the best depth-limited tree). Forests fix variance, which only helps when variance was the problem.
- **Impurity importance misleads.** `fnlwgt` (a census sampling weight) is #1 by impurity importance (17.4%) but shuffling it on held-out data changes accuracy by 0.0004 ± 0.0017. `capital-gain` costs 3.8 points. The Spearman correlation between the two rankings over all 82 features is only 0.01, but that is mostly noise from ~60 near-zero features, so I lead with the `fnlwgt` contrast, not that number. Permutation importance is also not clean: it splits credit between correlated features, and shuffling one one-hot column makes invalid rows.
- **Cost of permutation importance:** `d x repeats` full predictions. 82 features x 5 repeats x 5,000 rows over a 100-tree forest took 194 s on threads. That is why the study permutes a subsample of test rows.
- **Benchmark vs scikit-learn:** same accuracy up to ties, 3.5x to 47x slower to fit (forest predict is 5x to 19x slower, mostly because mine is single-threaded). Per-tree work is Python-call overhead, not arithmetic, exactly as DESIGN §8 predicted.
- Housekeeping slips: `platform.release()` reports "10" on Windows 11, so the benchmark note now prints the build number; a `1.9e+02` bar label and a legend sitting on top of the baseline label were only visible by opening the PNGs.
- **Legal pass before publishing.** Checked licenses from package metadata, not memory: all dependencies are permissive. Adult is CC BY 4.0 (attribution, link and a "changes made" statement are in `THIRD_PARTY_NOTICES.md`). For California housing I could find no stated license at all, so the repo never contains or redistributes any data and says so plainly. Added `DISCLAIMER.md` (no warranty, not for decisions about people because Adult includes sex and race, no affiliation), `SECURITY.md` (pickle files can execute code when loaded) and a README section linking them. None of this is legal advice.
