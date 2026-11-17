"""Typed, frozen configuration loaded from YAML (``configs/default.yaml``).

Every tunable lives in the YAML; code never hides magic numbers. Unknown keys are rejected
with a clear error so typos do not silently fall back to defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from treesfs.utils import PROJECT_ROOT


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations; relative paths are resolved against ``PROJECT_ROOT``."""

    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    figures: Path = Path("reports/figures")
    results: Path = Path("reports/results.md")
    artifacts: Path = Path("artifacts")


@dataclass(frozen=True)
class DataConfig:
    """Dataset sources and preprocessing knobs (see DESIGN.md §10)."""

    adult_url: str = "https://archive.ics.uci.edu/static/public/2/adult.zip"
    adult_sha256: str = "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb"
    max_categories: int | None = 16
    min_frequency: int = 1
    california_test_size: float = 0.2


@dataclass(frozen=True)
class TreeConfig:
    """Decision-tree hyper-parameters shared by both tasks."""

    criterion_classification: str = "gini"
    criterion_regression: str = "squared_error"
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_impurity_decrease: float = 0.0
    max_features: int | float | str | None = None


@dataclass(frozen=True)
class DepthSweepConfig:
    """Depths for the overfitting experiment; ``None`` means unlimited."""

    dataset: str = "adult"
    depths: tuple[int | None, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, None)


@dataclass(frozen=True)
class ExportConfig:
    """Settings for the small exported tree (``small_tree.png`` / ``.txt``)."""

    small_tree_depth: int = 3


@dataclass(frozen=True)
class ForestConfig:
    """Random-forest hyper-parameters (DESIGN.md §11).

    ``max_features=None`` means "the task default": ``"sqrt"`` for classification and ``1.0``
    for regression (as in scikit-learn >= 1.1). Depth and leaf-size limits are inherited from
    the ``tree`` section unless set here.
    """

    n_estimators: int = 100
    max_features: int | float | str | None = None
    bootstrap: bool = True
    oob_score: bool = True
    n_jobs: int | None = -1


@dataclass(frozen=True)
class NTreesSweepConfig:
    """OOB / test score as a function of the number of trees (one forest, staged scoring)."""

    dataset: str = "adult"
    grid: tuple[int, ...] = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200)


@dataclass(frozen=True)
class ImportanceConfig:
    """Impurity vs permutation importance study (README figure 3)."""

    dataset: str = "adult"
    n_repeats: int = 5
    max_rows: int | None = 5000  # held-out rows used for permuting; None = all test rows
    top_k: int = 15
    n_jobs: int | None = -1  # threads: the model is shared, never pickled


@dataclass(frozen=True)
class BenchmarkConfig:
    """Settings for ``scripts/compare_reference.py`` (treesfs vs scikit-learn)."""

    tree_depth: int = 10
    tree_repeats: int = 3  # best-of-N wall time for single trees
    forest_repeats: int = 1  # forests are slow; one run averages many trees already


@dataclass(frozen=True)
class Config:
    """Top-level configuration."""

    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tree: TreeConfig = field(default_factory=TreeConfig)
    depth_sweep: DepthSweepConfig = field(default_factory=DepthSweepConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    forest: ForestConfig = field(default_factory=ForestConfig)
    n_trees_sweep: NTreesSweepConfig = field(default_factory=NTreesSweepConfig)
    importance: ImportanceConfig = field(default_factory=ImportanceConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


_SECTIONS: dict[str, type] = {
    "paths": PathsConfig,
    "data": DataConfig,
    "tree": TreeConfig,
    "depth_sweep": DepthSweepConfig,
    "export": ExportConfig,
    "forest": ForestConfig,
    "n_trees_sweep": NTreesSweepConfig,
    "importance": ImportanceConfig,
    "benchmark": BenchmarkConfig,
}


def _build(cls: type, raw: dict[str, Any] | None, where: str) -> Any:
    """Instantiate dataclass ``cls`` from the mapping ``raw``, rejecting unknown keys."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"config section '{where}' must be a mapping, got {type(raw).__name__}")
    valid = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) {unknown} in section '{where}'. Valid keys: {sorted(valid)}"
        )
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        if cls is Config and name in _SECTIONS:
            kwargs[name] = _build(_SECTIONS[name], value, name)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: Path | str) -> Config:
    """Load a YAML config into a :class:`Config`.

    Relative ``paths.*`` entries are resolved against :data:`treesfs.utils.PROJECT_ROOT`, and
    ``depth_sweep.depths`` is converted to a tuple so the config stays hashable/frozen.

    Parameters
    ----------
    path : Path or str
        YAML file. A relative path is tried as given, then against ``PROJECT_ROOT``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        On unknown keys or malformed sections.
    """
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    with p.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _build(Config, raw, "<root>")

    paths = PathsConfig(
        **{f.name: _resolve(getattr(cfg.paths, f.name)) for f in fields(PathsConfig)}
    )
    sweep = DepthSweepConfig(dataset=cfg.depth_sweep.dataset, depths=tuple(cfg.depth_sweep.depths))
    n_trees = NTreesSweepConfig(
        dataset=cfg.n_trees_sweep.dataset, grid=tuple(int(k) for k in cfg.n_trees_sweep.grid)
    )
    return Config(
        seed=int(cfg.seed),
        paths=paths,
        data=cfg.data,
        tree=cfg.tree,
        depth_sweep=sweep,
        export=cfg.export,
        forest=cfg.forest,
        n_trees_sweep=n_trees,
        importance=cfg.importance,
        benchmark=cfg.benchmark,
    )


def _resolve(p: Path | str) -> Path:
    """Resolve ``p`` against the project root unless it is already absolute."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
