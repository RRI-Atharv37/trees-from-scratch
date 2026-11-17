# Third-party notices

`treesfs` is released under the [MIT License](LICENSE) (Copyright (c) 2026 Atharv Dewangan).
That licence covers **this repository's code and documentation only**. It does not cover the
software or datasets listed below, which keep their own licences.

Nothing listed here is copied into this repository or redistributed with it. Dependencies are
installed from PyPI by `uv sync`, and datasets are downloaded by the scripts from their
original sources.

## Software dependencies

Licences below were read from the package metadata of the versions pinned in `uv.lock` at the
time of writing; check the installed package for the authoritative text.

| Package | Used for | Version checked | Licence |
|---|---|---|---|
| NumPy | all numerics | 2.4.6 | BSD-3-Clause (plus bundled components under 0BSD, MIT, Zlib, CC0-1.0) |
| PyYAML | config loading | 6.0.3 | MIT |
| Matplotlib | figures | 3.11.2 | Matplotlib licence (PSF-based, permissive) |
| joblib | parallel tree growing, threads for permutation importance | 1.6.0 | BSD-3-Clause |
| scikit-learn | **downloader for California housing; reference implementation in tests, benchmarks and notebooks only** | 1.9.1 | BSD-3-Clause |
| pytest, Ruff, pre-commit | development tools | 9.2.0, 0.17.5, 4.7.0 | MIT |

scikit-learn pulls in SciPy (BSD-3-Clause) and threadpoolctl (BSD-3-Clause) transitively.

### scikit-learn and compatibility

No scikit-learn source code is copied into this project. The models are written from the
algorithm descriptions (see References in the README and `docs/DESIGN.md`). For comparability
the project deliberately follows scikit-learn's *conventions*: estimator method names, the
parallel-array layout of a fitted tree (`children_left`, `threshold`, ...), the plain-text
layout of `export_text`, and numerical tolerances. Interfaces, data layouts and output formats
of this kind are what makes the head-to-head tests in `tests/` possible.

scikit-learn: Copyright (c) 2007-2026 The scikit-learn developers. BSD 3-Clause License,
<https://github.com/scikit-learn/scikit-learn/blob/main/COPYING>.

## Datasets (not included in this repository)

### UCI Adult (classification)

* Creators: Barry Becker and Ronny Kohavi (1996). *Adult*. UCI Machine Learning Repository.
  <https://doi.org/10.24432/C5XW20>
* Source: <https://archive.ics.uci.edu/static/public/2/adult.zip> (SHA-256 verified on download).
* Licence: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
  as stated on the dataset's UCI page when checked.
* **Changes made** (CC BY 4.0 asks that modifications be indicated): the data is parsed (header
  and trailing-dot quirks handled), `?` is recoded to the category `missing`, categorical
  columns are one-hot encoded (capped at 16 values per column), and the files are split as
  provided. Nothing is redistributed; the transformed arrays live only on your disk, in
  gitignored folders.
* This is not an endorsement of this project by the creators or by the UCI repository.

### California housing (regression)

* Origin as cited by scikit-learn's documentation of the loader: Pace, R. Kelley and Ronald
  Barry (1997). "Sparse spatial autoregressions." *Statistics & Probability Letters* 33(3),
  291-297. Rows are 1990 US census block groups.
* Fetched with `sklearn.datasets.fetch_california_housing`, used as a downloader only.
* **Licence: no licence or terms of use were found** on scikit-learn's documentation page for
  the loader when checked. To stay on the safe side the data is **not** redistributed here: the
  repository's `.gitignore` excludes `data/raw/` and `data/processed/`, and each user downloads
  it from the original distribution. If you are the rights holder and want an attribution
  changed or a reference removed, open an issue and it will be handled promptly.

## Research references

* L. Breiman, J. Friedman, R. Olshen, C. Stone (1984). *Classification and Regression Trees*.
* L. Breiman (2001). "Random Forests." *Machine Learning* 45(1), 5-32.

Ideas and algorithms are not covered by copyright, but their authors deserve credit, and
this project cites them wherever it uses them.
