from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scout.config.hashing import config_hash
from scout.config.loader import load_config
from scout.data.calendar import reference_calendar_path
from scout.data.ingest import (
    load_tickers,
    read_candidate_pairs,
    require_min_delisted_fraction,
)
from scout.data.parquet_source import ParquetCandleSource
from scout.data.store import read_parquet, write_parquet_atomic
from scout.universe.build import build_snapshots, format_snapshot_summary
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

    cal_path = reference_calendar_path(Path(cfg.data.processed_dir), cfg.data.calendar)
    if not cal_path.is_file():
        raise ScoutDataError(f"calendar not found: {cal_path}; run adjust first")
    calendar = read_parquet(cal_path)
    source = ParquetCandleSource(cfg.data.processed_dir)
    panel = source.load_panel(
        [str(x) for x in candidates["asset_id"].tolist()],
        cfg.data.decision_timeframe,
        cfg.period.start,
        cfg.period.end,
    )
    snapshots = build_snapshots(
        panel.frame,
        candidates=candidates,
        tickers=tickers,
        calendar=calendar,
        cfg=cfg.universe,
        start=cfg.period.start,
        end=cfg.period.end,
    )
    out_path = Path(cfg.universe.snapshots_path)
    write_parquet_atomic(out_path, snapshots)
    for line in format_snapshot_summary(snapshots):
        print(line)
    print(f"snapshots_path={out_path.as_posix()}")
    print(f"snapshot_rows={len(snapshots)}")
    return 0
