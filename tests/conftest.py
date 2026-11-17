"""Shared fixtures: seeded synthetic datasets and a tiny Adult-format fixture on disk."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from treesfs.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]

ADULT_TRAIN = """\
39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K
38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K
53, ?, 234721, 11th, 7, Married-civ-spouse, ?, Husband, Black, Male, 0, 0, 40, ?, <=50K
28, Private, 338409, Bachelors, 13, Married-civ-spouse, Prof-specialty, Wife, Black, Female, 0, 0, 40, Holand-Netherlands, >50K
37, Private, 284582, Masters, 14, Married-civ-spouse, Exec-managerial, Wife, White, Female, 0, 0, 40, United-States, >50K

"""

ADULT_TEST = """\
|1x3 Cross validator
25, Private, 226802, 11th, 7, Never-married, Machine-op-inspct, Own-child, Black, Male, 0, 0, 40, United-States, <=50K.
38, Private, 89814, HS-grad, 9, Married-civ-spouse, Farming-fishing, Husband, White, Male, 0, 0, 50, United-States, <=50K.
44, ?, 160323, Some-college, 10, Married-civ-spouse, ?, Husband, Black, Male, 7688, 0, 40, ?, >50K.
28, Local-gov, 336951, Assoc-acdm, 12, Married-civ-spouse, Protective-serv, Husband, White, Male, 0, 0, 40, Mexico, >50K.

"""


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def tiny_clf() -> tuple[np.ndarray, np.ndarray]:
    """DESIGN.md §2.3 vector: best split x <= 3.5, children cost 4/3."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), np.array([0, 0, 0, 1, 1, 0])


@pytest.fixture
def blobs_clf() -> tuple[np.ndarray, np.ndarray]:
    """(300, 6) Gaussian 3-class data, cast through float32 as scikit-learn would see it."""
    g = np.random.default_rng(1)
    y = g.integers(0, 3, size=300)
    centers = np.array([[0, 0, 0], [1.5, 1.0, -1.0], [-1.0, 1.5, 1.5]])
    X = g.normal(size=(300, 6))
    X[:, :3] += centers[y]
    return X.astype(np.float32).astype(np.float64), y


@pytest.fixture
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    """(300, 5) continuous regression data; X cast through float32."""
    g = np.random.default_rng(2)
    X = g.normal(size=(300, 5))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 + 0.5 * X[:, 2] + g.normal(scale=0.1, size=300)
    return X.astype(np.float32).astype(np.float64), y


@pytest.fixture
def adult_fixture_dir(tmp_path: Path) -> Path:
    """Directory holding a 6-row ``adult.data`` and a 4-row ``adult.test`` with every gotcha."""
    (tmp_path / "adult.data").write_text(ADULT_TRAIN, encoding="utf-8", newline="\n")
    (tmp_path / "adult.test").write_text(ADULT_TEST, encoding="utf-8", newline="\n")
    return tmp_path


@pytest.fixture
def default_config() -> Config:
    return load_config(ROOT / "configs" / "default.yaml")


@pytest.fixture
def fixture_config(default_config: Config, adult_fixture_dir: Path) -> Config:
    """Default config whose raw dir points at the Adult fixture."""
    return replace(default_config, paths=replace(default_config.paths, raw=adult_fixture_dir))
