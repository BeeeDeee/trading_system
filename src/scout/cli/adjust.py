from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scout.config.hashing import config_hash
from scout.config.loader import load_config
from scout.data.actions import run_adjust
from scout.utils.logging import configure_logging


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)


def run(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    digest = config_hash(cfg)
    configure_logging(f"adjust-{digest[:12]}", level=cfg.logging.level)
    log = logging.getLogger("scout.cli.adjust")
    log.info("resolved config hash=%s processed_dir=%s", digest, cfg.data.processed_dir)
    print(f"config_hash={digest}")
    result = run_adjust(cfg)
    print(f"calendar_rows={result.calendar_rows}")
    print(f"panel_rows={result.panel_rows}")
    print(f"suspect_rows={result.suspect_rows}")
    print(f"suspect_rate={result.suspect_rate:.6f}")
    return 0
