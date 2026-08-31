from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from scout.config.hashing import config_hash, read_snapshot_id
from scout.config.loader import load_config
from scout.config.schema import ScoutConfig
from scout.data.calendar import reference_calendar_path
from scout.data.store import read_parquet
from scout.scoring.edge import build_edge_table
from scout.utils.errors import ScoutDataError
from scout.utils.logging import configure_logging


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)


def run(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    digest = config_hash(cfg)
    snapshot_id = read_snapshot_id(cfg.data.snapshot_id_path)
    configure_logging(f"build-edge-{digest[:12]}", level=cfg.logging.level)
    log = logging.getLogger("scout.cli.build_edge_table")
    log.info("resolved config hash=%s", digest)
    print(f"config_hash={digest}")
    print(f"data_snapshot_id={snapshot_id}")

    resolved = _load_labels(Path(cfg.labeling.labels_dir))
    calendar = _load_calendar(cfg)
    sessions = _session_closes(calendar, cfg.period.start, cfg.period.end)
    table = build_edge_table(
        resolved,
        cfg.edge,
        sessions=sessions,
        config_hash=digest,
        data_snapshot_id=snapshot_id,
    )
    path = Path(cfg.edge.table_path)
    table.save(path)
    n_resolved = 0 if resolved.empty else int(resolved["resolution_ts"].notna().sum())
    print(f"resolved_setups={n_resolved}")
    print(f"table_path={path.as_posix()}")
    return 0


def _load_labels(labels_dir: Path) -> pd.DataFrame:
    if not labels_dir.is_dir():
        raise ScoutDataError(f"labels directory not found: {labels_dir}; run label first")
    paths = sorted(labels_dir.glob("setups_*.parquet"))
    if not paths:
        raise ScoutDataError(f"no setups_*.parquet in {labels_dir}; run label first")
    frames = [read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def _load_calendar(cfg: ScoutConfig) -> pd.DataFrame:
    path = reference_calendar_path(Path(cfg.data.processed_dir), cfg.data.calendar)
    if not path.is_file():
        raise ScoutDataError(f"calendar not found: {path}; run adjust first")
    return read_parquet(path)


def _session_closes(calendar: pd.DataFrame, start: datetime, end: datetime) -> pd.DatetimeIndex:
    if calendar.empty or "close_utc" not in calendar.columns:
        raise ScoutDataError("calendar is empty or missing close_utc")
    close = pd.DatetimeIndex(pd.to_datetime(calendar["close_utc"], utc=True))
    in_range = (close >= pd.Timestamp(start)) & (close <= pd.Timestamp(end))
    sessions = close[in_range]
    if len(sessions) == 0:
        raise ScoutDataError("no calendar sessions in [period.start, period.end]")
    return sessions
