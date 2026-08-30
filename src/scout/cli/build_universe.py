from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scout.config.hashing import config_hash
from scout.config.loader import load_config
from scout.data.ingest import (
    load_tickers,
    read_candidate_pairs,
    require_min_delisted_fraction,
)
from scout.utils.errors import ScoutDataError
from scout.utils.logging import configure_logging


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)


def run(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    digest = config_hash(cfg)
    configure_logging(f"build-universe-{digest[:12]}", level=cfg.logging.level)
    log = logging.getLogger("scout.cli.build_universe")
    log.info("resolved config hash=%s", digest)
    print(f"config_hash={digest}")
    candidates = read_candidate_pairs(Path(cfg.universe.candidates_file))
    tickers = load_tickers(Path(cfg.data.raw_dir))
    fraction = require_min_delisted_fraction(candidates, tickers)
    print(f"delisted_fraction={fraction:.4f}")
    raise ScoutDataError(
        "universe snapshot construction is M1.7; delisted-fraction check passed"
    )
