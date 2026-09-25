from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from scout.config.hashing import config_hash
from scout.config.loader import load_config
from scout.data.benchmark import (
    VIX_SYMBOL,
    benchmark_path,
    build_benchmark_frame,
    panel_slice,
    ticker_id,
    write_benchmark,
)
from scout.data.ingest import load_tickers
from scout.data.parquet_source import ParquetCandleSource
from scout.utils.errors import ScoutDataError
from scout.utils.logging import configure_logging


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)


def run(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    digest = config_hash(cfg)
    configure_logging(f"build-benchmark-{digest[:12]}", level=cfg.logging.level)
    log = logging.getLogger("scout.cli.build_benchmark")
    out = benchmark_path(Path(cfg.data.processed_dir))
    log.info("resolved config hash=%s dest=%s", digest, out)
    print(f"config_hash={digest}")

    tickers = load_tickers(Path(cfg.data.raw_dir))
    spy_id = ticker_id(tickers, cfg.market_regime.benchmark_symbol)
    try:
        vix_id = ticker_id(tickers, VIX_SYMBOL)
    except ScoutDataError:
        vix_id = None
        log.warning("VIX symbol %s not in tickers; vix_close will be null", VIX_SYMBOL)

    source = ParquetCandleSource(cfg.data.processed_dir)
    start, end = _panel_bounds(source, spy_id, cfg.data.decision_timeframe)
    ids = [spy_id] if vix_id is None else [spy_id, vix_id]
    panel = source.load_panel(ids, cfg.data.decision_timeframe, start, end)
    spy = panel_slice(panel, spy_id)
    vix = panel_slice(panel, vix_id) if vix_id is not None else None
    frame = build_benchmark_frame(spy, vix)
    write_benchmark(out, frame)
    n_vix = 0 if vix is None or vix.empty else int(frame["vix_close"].notna().sum())
    print(f"benchmark_path={out.as_posix()}")
    print(f"rows={len(frame)}")
    print(f"vix_rows={n_vix}")
    return 0


def _panel_bounds(
    source: ParquetCandleSource, asset_id: str, timeframe: str
) -> tuple[datetime, datetime]:
    bounds = source.available_range(asset_id, timeframe)
    if bounds is None:
        raise ScoutDataError(
            f"no processed panel rows for {asset_id}; run adjust first"
        )
    start, end = bounds
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    return start, end
