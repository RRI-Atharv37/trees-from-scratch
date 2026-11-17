"""Small shared helpers: project paths, seeding, logging, timing, report sections."""

from __future__ import annotations

import logging
import random
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: this file) to the directory with ``pyproject.toml``.

    Falls back to the current working directory when the package is installed somewhere
    that is not inside a checkout (for example a wheel in ``site-packages``).
    """
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


PROJECT_ROOT: Path = _find_project_root()

_LOGGER_ROOT = "treesfs"


def set_seed(seed: int) -> np.random.Generator:
    """Seed every global RNG we might touch and return a fresh ``Generator``.

    Seeds Python's ``random`` and NumPy's *legacy* global state (so stray ``np.random.*``
    calls are reproducible too) and returns ``np.random.default_rng(seed)``, which is what the
    library itself uses.

    Parameters
    ----------
    seed : int
        The single seed from the config.

    Returns
    -------
    numpy.random.Generator
        A new generator seeded with ``seed``.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    return np.random.default_rng(seed)


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``treesfs`` namespace with one stderr handler configured.

    The handler is attached once, to the ``treesfs`` logger, so repeated calls never print
    duplicate lines.
    """
    root = logging.getLogger(_LOGGER_ROOT)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    if name != _LOGGER_ROOT and not name.startswith(_LOGGER_ROOT + "."):
        name = f"{_LOGGER_ROOT}.{name}"
    return logging.getLogger(name)


def ensure_dir(path: Path | str) -> Path:
    """Create ``path`` (and parents) if missing and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class Timer:
    """Holds the elapsed wall-clock seconds filled in by :func:`timer`."""

    elapsed: float = 0.0


@contextmanager
def timer() -> Iterator[Timer]:
    """Context manager measuring wall-clock time with ``time.perf_counter``.

    Examples
    --------
    >>> with timer() as t:
    ...     _ = sum(range(1000))
    >>> t.elapsed >= 0.0
    True
    """
    t = Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - start


def write_text_lf(path: Path | str, text: str) -> Path:
    """Write UTF-8 text with ``\\n`` line endings on every OS (stable diffs on Windows)."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return p


def update_marked_section(path: Path | str, key: str, body: str, title: str = "Results") -> Path:
    """Idempotently (re)write one named section of a Markdown file.

    The section lives between ``<!-- treesfs:KEY:start -->`` and ``<!-- treesfs:KEY:end -->``.
    If both markers exist the text between them is replaced; otherwise the block is appended
    (creating the file with a ``# title`` header if needed). Running it twice with the same
    ``body`` leaves the file byte-identical, and sections for other keys are untouched.

    Parameters
    ----------
    path : Path or str
        Markdown file to update.
    key : str
        Section key, e.g. ``"adult"``.
    body : str
        Markdown for the section (without the markers).
    title : str
        Header used only when the file does not exist yet.
    """
    p = Path(path)
    start = f"<!-- treesfs:{key}:start -->"
    end = f"<!-- treesfs:{key}:end -->"
    block = f"{start}\n{body.strip()}\n{end}"
    text = p.read_text(encoding="utf-8") if p.exists() else f"# {title}\n"
    i, j = text.find(start), text.find(end)
    if i != -1 and j > i:
        text = text[:i] + block + text[j + len(end) :]
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    return write_text_lf(p, text)
