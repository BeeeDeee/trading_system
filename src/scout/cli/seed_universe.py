from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scout.config.hashing import config_hash
from scout.config.loader import load_config
from scout.data.ingest import load_tickers
from scout.universe.candidates import seed_candidates
from scout.utils.logging import configure_logging


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)


def run(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    digest = config_hash(cfg)
    configure_logging(f"seed-universe-{digest[:12]}", level=cfg.logging.level)
    log = logging.getLogger("scout.cli.seed_universe")
    dest = Path(cfg.universe.candidates_file)
    ohlcv_dir = Path(cfg.data.raw_dir) / "ohlcv" / cfg.data.decision_timeframe
    log.info("resolved config hash=%s dest=%s", digest, dest)
    print(f"config_hash={digest}")
    tickers = load_tickers(Path(cfg.data.raw_dir))
    n_selected, n_added, fraction = seed_candidates(tickers, ohlcv_dir, dest)
    print(f"selected={n_selected}")
    print(f"added={n_added}")
    print(f"delisted_fraction={fraction:.4f}")
    print(f"candidates_path={dest.as_posix()}")
    return 0
