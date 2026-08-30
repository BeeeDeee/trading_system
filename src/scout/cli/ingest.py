from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from scout.config.hashing import config_hash
from scout.config.loader import load_config
from scout.data.ingest import run_ingest
from scout.utils.clock import WallClock
from scout.utils.logging import configure_logging


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--symbols-file", type=Path, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)


def run(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    digest = config_hash(cfg)
    configure_logging(f"ingest-{digest[:12]}", level=cfg.logging.level)
    log = logging.getLogger("scout.cli.ingest")
    log.info(
        "resolved config hash=%s vendor=%s raw_dir=%s",
        digest,
        cfg.data.vendor.value,
        cfg.data.raw_dir,
    )
    print(f"config_hash={digest}")
    print(f"vendor={cfg.data.vendor.value}")
    if cfg.data.vendor.value == "sharadar":
        print("sharadar bulk download can take several minutes (~1 GB stocks table)")
    start = _parse_bound(args.start, cfg.period.start)
    end = _parse_bound(args.end, cfg.period.end)
    manifest = run_ingest(
        cfg,
        start=start,
        end=end,
        raw_dir=Path(cfg.data.raw_dir),
        snapshot_path=Path(cfg.data.snapshot_id_path),
        clock=WallClock(),
        symbols_file=Path(args.symbols_file) if args.symbols_file else None,
    )
    print(f"data_snapshot_id={manifest.data_snapshot_id}")
    print(f"symbols={manifest.symbols} rows={manifest.rows}")
    return 0


def _parse_bound(raw: str | None, fallback: datetime) -> datetime:
    if raw is None:
        return fallback
    parsed = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=UTC)
    return parsed
