# Results

<!-- treesfs:adult:start -->
## Adult (income > 50K), treesfs DecisionTreeClassifier

| metric | value |
|---|---|
| train accuracy | 0.8703 |
| test accuracy | 0.8601 |
| max_depth (param) | 10 |
| depth (actual) | 10 |
| leaves | 260 |
| nodes | 519 |
| features | 82 |
| fit time (s) | 4.88 |
| predict time, 16281 test rows (s) | 0.0060 |
<!-- treesfs:adult:end -->

<!-- treesfs:california:start -->
## California housing, treesfs DecisionTreeRegressor

| metric | value |
|---|---|
| train R² | 0.8228 |
| test R² | 0.6590 |
| test RMSE (x$100k) | 0.6654 |
| test MAE (x$100k) | 0.4434 |
| max_depth (param) | 10 |
| depth (actual) | 10 |
| leaves | 780 |
| nodes | 1559 |
| features | 8 |
| fit time (s) | 0.35 |
| predict time, 4128 test rows (s) | 0.0011 |
<!-- treesfs:california:end -->

<!-- treesfs:adult_forest:start -->
## Adult (income > 50K), treesfs RandomForestClassifier

| metric | value |
|---|---|
| train accuracy | 1.0000 |
| test accuracy | 0.8528 |
| OOB accuracy | 0.8553 |
| trees | 100 |
| max_depth (param) | None |
| depth (mean / max) | 49.0 / 65 |
| leaves per tree (mean) | 3556 |
| features | 82 |
| fit time (s) | 150.76 |
| predict time, 16281 test rows (s) | 0.8711 |
<!-- treesfs:adult_forest:end -->

<!-- treesfs:california_forest:start -->
## California housing, treesfs RandomForestRegressor

| metric | value |
|---|---|
| train R² | 0.9733 |
| test R² | 0.8001 |
| test RMSE (x$100k) | 0.5094 |
| test MAE (x$100k) | 0.3333 |
| OOB R² | 0.8063 |
| trees | 100 |
| max_depth (param) | None |
| depth (mean / max) | 33.1 / 39 |
| leaves per tree (mean) | 10001 |
| features | 8 |
| fit time (s) | 146.23 |
| predict time, 4128 test rows (s) | 0.1893 |
<!-- treesfs:california_forest:end -->

<!-- treesfs:importance_adult:start -->
## Feature importance, adult forest (100 trees)

Permutation importance on 5000 held-out rows, 5 repeats; baseline test accuracy 0.8508. Spearman correlation of the two rankings over all 82 features: 0.01.

| feature | impurity (rank) | permutation drop (rank) |
|---|---|---|
| fnlwgt | 0.1742 (1) | 0.0004 ± 0.0017 (15) |
| age | 0.1357 (2) | 0.0123 ± 0.0014 (2) |
| capital-gain | 0.1064 (3) | 0.0378 ± 0.0003 (1) |
| hours-per-week | 0.0743 (4) | 0.0015 ± 0.0029 (8) |
| marital-status=Married-civ-spouse | 0.0739 (5) | 0.0117 ± 0.0029 (3) |
| education-num | 0.0715 (6) | 0.0107 ± 0.0027 (4) |
| relationship=Husband | 0.0419 (7) | -0.0015 ± 0.0011 (78) |
| capital-loss | 0.0323 (8) | 0.0080 ± 0.0010 (5) |
| marital-status=Never-married | 0.0240 (9) | 0.0007 ± 0.0010 (12) |
| occupation=Exec-managerial | 0.0191 (10) | 0.0048 ± 0.0019 (6) |
| occupation=Prof-specialty | 0.0153 (11) | 0.0008 ± 0.0017 (11) |
| workclass=Private | 0.0101 (12) | -0.0037 ± 0.0004 (81) |
| education=Bachelors | 0.0094 (13) | -0.0004 ± 0.0009 (61) |
| workclass=Self-emp-not-inc | 0.0088 (14) | -0.0002 ± 0.0006 (52) |
| relationship=Not-in-family | 0.0087 (15) | -0.0012 ± 0.0007 (75) |
<!-- treesfs:importance_adult:end -->

<!-- treesfs:compare:start -->
## treesfs vs scikit-learn

| dataset | model | implementation | test metric | OOB | fit (s) | predict (s) | fit vs sklearn |
|---|---|---|---|---|---|---|---|
| adult | tree d=10 | treesfs | accuracy 0.8601 | - | 2.18 | 0.0055 | 8.4x slower |
| adult | tree d=10 | scikit-learn | accuracy 0.8603 | - | 0.26 | 0.0051 | - |
| adult | tree unlimited | treesfs | accuracy 0.8117 | - | 6.99 | 0.0099 | 17.1x slower |
| adult | tree unlimited | scikit-learn | accuracy 0.8098 | - | 0.41 | 0.0070 | - |
| adult | forest x100 | treesfs | accuracy 0.8528 | 0.8553 | 85.16 | 2.5962 | 31.6x slower |
| adult | forest x100 | scikit-learn | accuracy 0.8517 | 0.8539 | 2.69 | 0.1346 | - |
| california | tree d=10 | treesfs | R² 0.6590 | - | 0.91 | 0.0028 | 3.5x slower |
| california | tree d=10 | scikit-learn | R² 0.6677 | - | 0.26 | 0.0017 | - |
| california | tree unlimited | treesfs | R² 0.5820 | - | 20.94 | 0.0058 | 46.6x slower |
| california | tree unlimited | scikit-learn | R² 0.6016 | - | 0.45 | 0.0034 | - |
| california | forest x100 | treesfs | R² 0.8001 | 0.8063 | 192.19 | 0.6636 | 38.9x slower |
| california | forest x100 | scikit-learn | R² 0.8014 | 0.8063 | 4.94 | 0.1254 | - |

Measured on Windows (build 10.0.26200), 24 logical cores, Python 3.11.17, NumPy 2.4.6, scikit-learn 1.9.1. Same float64 train/test arrays for both implementations (scikit-learn casts to float32 internally). Best-of-N wall time; forests use `n_jobs=-1` on both sides. Timings vary with machine load; the ratios, not the seconds, are the portable result.

Notes: treesfs `predict` is single-threaded while scikit-learn's forest predicts with `n_jobs` threads, which explains most of the forest predict-time gap. Metric differences between the two implementations on single trees come from tie-breaking (DESIGN.md §9.3).
<!-- treesfs:compare:end -->
