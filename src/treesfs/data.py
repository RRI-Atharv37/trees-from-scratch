"""Datasets: UCI Adult (classification) and California housing (regression).

Everything here is deterministic given the config seed. Network access happens only in
:func:`download_adult` and :func:`download_california`; all parsing/loading works offline from
``data/raw``. See ``data/README.md`` and DESIGN.md §10.
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from treesfs.config import Config
from treesfs.preprocessing import OneHotEncoder, build_design_matrix
from treesfs.utils import ensure_dir, get_logger

log = get_logger(__name__)

ADULT_COLUMNS: list[str] = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
ADULT_NUMERIC: list[str] = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
ADULT_CATEGORICAL: list[str] = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
ADULT_FILES: tuple[str, ...] = ("adult.data", "adult.test", "adult.names")
MISSING_TOKEN = "missing"
_CHUNK = 1 << 16


@dataclass(frozen=True)
class AdultData:
    """Encoded Adult train/test split (``y`` is 1 for ``>50K``)."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    encoder: OneHotEncoder


@dataclass(frozen=True)
class RegressionData:
    """Train/test split of a regression dataset."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


# --------------------------------------------------------------------------- Adult
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download_adult(raw_dir: Path | str, url: str, sha256: str) -> Path:
    """Download the UCI Adult zip, verify its SHA-256 and extract the data files.

    Streams to ``adult.zip.part``, checks the hash, renames to ``adult.zip`` and extracts
    ``adult.data``, ``adult.test`` and ``adult.names`` into ``raw_dir``. Does nothing if the
    zip is already present with a matching hash and the files are extracted.

    Parameters
    ----------
    raw_dir : Path or str
    url : str
    sha256 : str
        Expected hex digest of the zip.

    Returns
    -------
    Path
        ``raw_dir`` (containing the extracted files).

    Raises
    ------
    RuntimeError
        If the downloaded file's hash does not match.
    """
    raw = ensure_dir(raw_dir)
    zip_path = raw / "adult.zip"
    if zip_path.exists() and _sha256_file(zip_path) == sha256:
        log.info("adult.zip already present with matching sha256; skipping download")
    else:
        part = raw / "adult.zip.part"
        log.info("Downloading %s", url)
        h = hashlib.sha256()
        req = urllib.request.Request(url, headers={"User-Agent": "treesfs-downloader"})
        with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as out:
            while chunk := resp.read(_CHUNK):
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != sha256:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"sha256 mismatch for {url}: expected {sha256}, got {h.hexdigest()}")
        part.replace(zip_path)
        log.info("Downloaded and verified adult.zip (sha256 ok)")
    if all((raw / name).exists() for name in ADULT_FILES):
        return raw
    with zipfile.ZipFile(zip_path) as zf:
        for name in ADULT_FILES:  # read by member name; never extract attacker-chosen paths
            (raw / name).write_bytes(zf.read(name))
    log.info("Extracted %s into %s", ", ".join(ADULT_FILES), raw)
    return raw


def parse_adult(path: Path | str, is_test: bool) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Parse ``adult.data`` / ``adult.test`` (DESIGN.md §10.1).

    Handles every UCI quirk: the ``|1x3 Cross validator`` junk first line of the test file,
    blank lines, the ``", "`` separator (split on ``,`` and strip each token), the trailing
    ``.`` on test labels, and ``?`` meaning missing (mapped to the explicit category
    ``"missing"``, so a tree can split on missingness).

    Parameters
    ----------
    path : Path or str
    is_test : bool
        ``True`` for ``adult.test`` (skips ``|``-comment lines, strips the label's dot).

    Returns
    -------
    columns : dict[str, ndarray]
        The 14 feature columns, shape ``(n,)``: numeric as float64, categorical as ``str``.
    y : ndarray, shape (n,), int64
        1 for ``>50K`` else 0.

    Raises
    ------
    ValueError
        If a row does not have 15 fields (message includes the 1-based line number) or the
        label is unknown.
    """
    text = Path(path).read_text(encoding="utf-8")
    rows: list[list[str]] = []
    # Parsing text lines is I/O-bound line handling, not a numerical hot path.
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or (is_test and line.startswith("|")):
            continue
        fields = [tok.strip() for tok in line.split(",")]
        if len(fields) != len(ADULT_COLUMNS):
            raise ValueError(
                f"{path}: line {lineno} has {len(fields)} fields, expected {len(ADULT_COLUMNS)}"
            )
        rows.append(fields)

    cols = list(zip(*rows, strict=True)) if rows else [()] * len(ADULT_COLUMNS)
    by_name = dict(zip(ADULT_COLUMNS, cols, strict=True))
    columns: dict[str, np.ndarray] = {}
    for name in ADULT_NUMERIC:
        columns[name] = np.asarray(by_name[name], dtype=np.float64)
    for name in ADULT_CATEGORICAL:
        arr = np.asarray(by_name[name], dtype=str)
        columns[name] = np.where(arr == "?", MISSING_TOKEN, arr)
    labels = np.asarray([lab.rstrip(".") for lab in by_name["income"]], dtype=str)
    bad = sorted(set(labels.tolist()) - {"<=50K", ">50K"})
    if bad:
        raise ValueError(f"{path}: unexpected income labels {bad}")
    return columns, (labels == ">50K").astype(np.int64)


def load_adult(cfg: Config) -> AdultData:
    """Load, parse and one-hot encode Adult. The encoder is fit on the **train** split only.

    Parameters
    ----------
    cfg : Config
        Uses ``paths.raw``, ``data.max_categories`` and ``data.min_frequency``.

    Raises
    ------
    FileNotFoundError
        If the raw files are missing (run ``scripts/download_data.py``).
    """
    raw = Path(cfg.paths.raw)
    train_path, test_path = raw / "adult.data", raw / "adult.test"
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run: uv run python scripts/download_data.py --dataset adult"
            )
    cols_tr, y_tr = parse_adult(train_path, is_test=False)
    cols_te, y_te = parse_adult(test_path, is_test=True)

    def split(cols: dict[str, np.ndarray]) -> tuple[dict, dict]:
        return (
            {k: cols[k] for k in ADULT_NUMERIC},
            {k: cols[k] for k in ADULT_CATEGORICAL},
        )

    num_tr, cat_tr = split(cols_tr)
    num_te, cat_te = split(cols_te)
    encoder = OneHotEncoder(cfg.data.max_categories, cfg.data.min_frequency).fit(cat_tr)
    X_tr, names = build_design_matrix(num_tr, cat_tr, encoder)
    X_te, _ = build_design_matrix(num_te, cat_te, encoder)
    return AdultData(X_tr, y_tr, X_te, y_te, names, encoder)


# --------------------------------------------------------------------------- California
def download_california(raw_dir: Path | str) -> Path:
    """Fetch California housing via scikit-learn (used as a **downloader only**).

    The lazy import keeps scikit-learn out of every other module. The arrays are also written to
    ``raw_dir/california_housing.npz`` so later loads do not need scikit-learn at all.

    Returns
    -------
    Path
        The ``california_housing.npz`` file (20,640 x 8 features, 20,640 targets).
    """
    raw = ensure_dir(raw_dir)
    out = raw / "california_housing.npz"
    if out.exists():
        log.info("%s already present; skipping download", out.name)
        return out
    from sklearn.datasets import fetch_california_housing  # lazy: downloader only

    log.info("Fetching California housing via scikit-learn into %s", raw)
    bunch = fetch_california_housing(data_home=str(raw), as_frame=False)
    np.savez_compressed(
        out,
        X=bunch.data.astype(np.float64),
        y=bunch.target.astype(np.float64),
        feature_names=np.asarray(bunch.feature_names, dtype=str),
    )
    return out


def train_test_split_indices(
    n: int, test_size: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Seeded random split into disjoint train/test index arrays (NumPy only).

    A single ``rng.permutation(n)``; the first ``round(n * test_size)`` indices are the test set.
    Both outputs are returned sorted so they are easy to inspect.

    Parameters
    ----------
    n : int
    test_size : float
        Fraction in ``(0, 1)``.
    rng : numpy.random.Generator

    Returns
    -------
    train_idx, test_idx : ndarray of int64
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    perm = rng.permutation(n)
    n_test = int(round(n * test_size))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def load_california(cfg: Config) -> RegressionData:
    """Load California housing with a seeded 80/20 split; cache to ``california.npz``.

    The processed file ``data/processed/california.npz`` holds the exact split, so the result is
    identical on every run and machine.

    Raises
    ------
    FileNotFoundError
        If the raw data has not been downloaded.
    """
    processed = Path(cfg.paths.processed) / "california.npz"
    if processed.exists():
        z = np.load(processed)
        return RegressionData(
            z["X_train"], z["y_train"], z["X_test"], z["y_test"], z["feature_names"].tolist()
        )
    raw_file = Path(cfg.paths.raw) / "california_housing.npz"
    if not raw_file.exists():
        raise FileNotFoundError(
            f"{raw_file} not found. "
            "Run: uv run python scripts/download_data.py --dataset california"
        )
    z = np.load(raw_file)
    X, y, names = z["X"], z["y"], z["feature_names"].tolist()
    rng = np.random.default_rng(cfg.seed)
    tr, te = train_test_split_indices(len(y), cfg.data.california_test_size, rng)
    ensure_dir(processed.parent)
    np.savez_compressed(
        processed,
        X_train=X[tr],
        y_train=y[tr],
        X_test=X[te],
        y_test=y[te],
        feature_names=np.asarray(names, dtype=str),
    )
    return RegressionData(X[tr], y[tr], X[te], y[te], names)


def load_dataset(name: str, cfg: Config) -> AdultData | RegressionData:
    """Load ``"adult"`` or ``"california"`` by name."""
    if name == "adult":
        return load_adult(cfg)
    if name == "california":
        return load_california(cfg)
    raise ValueError(f"unknown dataset {name!r}; choose 'adult' or 'california'")


def task_for(name: str) -> str:
    """``"classification"`` for Adult, ``"regression"`` for California."""
    if name == "adult":
        return "classification"
    if name == "california":
        return "regression"
    raise ValueError(f"unknown dataset {name!r}; choose 'adult' or 'california'")
