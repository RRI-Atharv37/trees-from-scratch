"""Download datasets into data/raw (Adult with sha256 verification, California via sklearn)."""

from __future__ import annotations

import argparse

from treesfs.config import load_config
from treesfs.data import download_adult, download_california
from treesfs.utils import get_logger

log = get_logger("download_data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", choices=["adult", "california", "all"], default="all")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dataset in ("adult", "all"):
        download_adult(cfg.paths.raw, cfg.data.adult_url, cfg.data.adult_sha256)
    if args.dataset in ("california", "all"):
        download_california(cfg.paths.raw)
    log.info("Done. Raw data in %s", cfg.paths.raw)


if __name__ == "__main__":
    main()
