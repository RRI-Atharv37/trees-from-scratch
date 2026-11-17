"""Smoke tests for scripts/compare_reference.py on tiny synthetic data (no real datasets)."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_reference.py"


@pytest.fixture(scope="module")
def cr():
    spec = importlib.util.spec_from_file_location("compare_reference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve string annotations via sys.modules
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@dataclass
class _Data:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def _tiny_data() -> _Data:
    g = np.random.default_rng(0)
    X = g.normal(size=(240, 4)).astype(np.float32).astype(np.float64)
    y = (X[:, 0] + 0.3 * g.normal(size=240) > 0).astype(int)
    return _Data(X[:160], y[:160], X[160:], y[160:])


def test_run_dataset_pairs_every_case_and_metrics_are_close(cr, default_config, monkeypatch):
    monkeypatch.setattr(cr, "task_for", lambda name: "classification")
    cfg = default_config
    rows = cr.run_dataset("toy", cfg, _tiny_data(), n_estimators=8, with_forest=True, quick=True)
    # 3 cases (tree d=10, tree unlimited, forest) x 2 implementations
    assert len(rows) == 6
    for case in {r.model for r in rows}:
        mine, ref = (
            next(r for r in rows if r.model == case and r.impl == i)
            for i in ("treesfs", "scikit-learn")
        )
        assert abs(mine.metric - ref.metric) < 0.12  # same task, same data
        assert mine.fit_s > 0 and ref.predict_s > 0
    forests = [r for r in rows if r.model.startswith("forest")]
    assert all(r.oob is not None for r in forests)


def test_markdown_table_and_environment_note(cr):
    rows = [
        cr.BenchRow("toy", "tree d=3", "treesfs", "accuracy", 0.9, 2.0, 0.01),
        cr.BenchRow("toy", "tree d=3", "scikit-learn", "accuracy", 0.9, 0.5, 0.001),
    ]
    table = cr.markdown_table(rows)
    assert "4.0x slower" in table and table.count("\n") == 3
    assert "scikit-learn" in cr.environment_note(-1) and "n_jobs=-1" in cr.environment_note(-1)


def test_best_of_returns_fastest_time_and_last_result(cr):
    calls = []
    result, best = cr.best_of(lambda: calls.append(1) or len(calls), repeats=3)
    assert result == 3 and len(calls) == 3 and best >= 0
