from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TextIO, cast

import pandas as pd  # type: ignore[import-untyped]

from scout.backtest.engine import BacktestEngine
from scout.backtest.sim_broker import SimBroker
from scout.config.hashing import config_hash
from scout.config.loader import load_config
from scout.config.schema import ScoutConfig
from scout.data.calendar import reference_calendar_path
from scout.data.ingest import load_tickers, read_candidate_pairs
from scout.data.parquet_source import ParquetCandleSource
from scout.data.store import read_parquet
from scout.domain.edge import EdgeTable
from scout.domain.enums import ActionType
from scout.domain.market import Asset, CorporateAction, EarningsEvent
from scout.domain.ports import SentimentSource
from scout.sentiment.null_source import NullSentimentSource
from scout.storage.decision_sink import ParquetDecisionSink
from scout.storage.run_outputs import make_run_id, results_dir, write_run_outputs
from scout.strategies.registry import build_strategies
from scout.universe.build import SnapshotBook
from scout.utils.clock import WallClock
from scout.utils.errors import ScoutDataError
from scout.utils.logging import configure_logging


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--notes", default="", help="Registry notes; required for holdout")


def run(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    digest = config_hash(cfg)
    clock = WallClock()
    run_id = make_run_id(cfg.run.strategy_slug, clock)
    out_dir = results_dir(cfg, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        stream = _Tee(sys.stdout, log_file)
        configure_logging(run_id, level=cfg.logging.level, stream=cast(TextIO, stream))
        log = logging.getLogger("scout.cli.run_backtest")
        log.info("resolved config hash=%s", digest)
        print(f"config_hash={digest}")
        print(f"run_id={run_id}")

        candles = ParquetCandleSource(cfg.data.processed_dir)
        sentiment: SentimentSource = NullSentimentSource()
        sink = ParquetDecisionSink(
            out_dir / "decisions.parquet",
            flush_every=cfg.audit.flush_every_cycles,
        )
        broker = SimBroker(cfg.costs, cfg.portfolio)
        strategies = build_strategies(cfg.strategies)
        edge_path = Path(cfg.edge.table_path)
        if not edge_path.is_file():
            raise ScoutDataError(f"edge table not found: {edge_path}; run build-edge first")
        edge_table = EdgeTable.load(edge_path)
        universe = SnapshotBook(_load_snapshots(cfg))
        assets = _load_assets(cfg)
        calendar = _load_calendar(cfg)
        earnings = _load_earnings(Path(cfg.data.raw_dir))
        actions = _load_actions(Path(cfg.data.raw_dir))

        engine = BacktestEngine(
            candles=candles,
            sentiment=sentiment,
            broker=broker,
            strategies=strategies,
            edge_table=edge_table,
            sink=sink,
            universe=universe,
            assets=assets,
            cfg=cfg,
            calendar=calendar,
            earnings_by_asset=earnings,
            actions=actions,
            run_id=run_id,
            clock=clock,
        )
        result = engine.run()
        write_run_outputs(
            result,
            run_id,
            cfg,
            notes=str(args.notes),
            clock=clock,
            used_bins=engine.used_bins_frame(),
        )
        print(f"results_dir={out_dir.as_posix()}")
        print(f"n_trades={len(result.trades)}")
        print(f"n_decisions={result.n_decisions_considered}")
    return 0


class _Tee:
    def __init__(self, a: TextIO, b: TextIO) -> None:
        self._a = a
        self._b = b

    def write(self, data: str) -> int:
        self._a.write(data)
        self._b.write(data)
        return len(data)

    def flush(self) -> None:
        self._a.flush()
        self._b.flush()


def _load_snapshots(cfg: ScoutConfig) -> pd.DataFrame:
    path = Path(cfg.universe.snapshots_path)
    if not path.is_file():
        raise ScoutDataError(f"universe snapshots not found: {path}; run build-universe first")
    return read_parquet(path)


def _load_calendar(cfg: ScoutConfig) -> pd.DataFrame:
    path = reference_calendar_path(Path(cfg.data.processed_dir), cfg.data.calendar)
    if not path.is_file():
        return pd.DataFrame()
    return read_parquet(path)


def _load_assets(cfg: ScoutConfig) -> dict[str, Asset]:
    candidates = read_candidate_pairs(Path(cfg.universe.candidates_file))
    tickers = load_tickers(Path(cfg.data.raw_dir))
    before = len(candidates)
    joined = candidates.merge(tickers, on="asset_id", how="left", validate="many_to_one")
    if len(joined) != before:
        raise ScoutDataError("asset merge duplicated candidate rows")
    if "symbol" in joined.columns and "symbol_x" in joined.columns:
        pass
    symbol_col = "symbol_x" if "symbol_x" in joined.columns else "symbol"
    assets: dict[str, Asset] = {}
    for rec in joined.to_dict("records"):
        symbol = str(rec.get(symbol_col) or rec.get("symbol") or rec["asset_id"])
        sector = rec.get("sector")
        cluster = str(sector) if sector is not None and str(sector) != "nan" else "OTHER"
        is_etf = bool(rec.get("is_etf", False))
        listed = rec.get("listed_date")
        delisted = rec.get("delisted_date")
        assets[symbol] = Asset(
            asset_id=str(rec["asset_id"]),
            symbol=symbol,
            exchange=str(rec.get("exchange") or "NYSE"),
            quote_currency="USD",
            is_etf=is_etf,
            cluster=cluster,
            tick_size=Decimal("0.01"),
            step_size=Decimal("1"),
            min_notional_usd=Decimal("0"),
            listed_at=_as_aware(listed),
            delisted_at=_as_aware(delisted),
            delist_reason=(
                None if rec.get("delist_reason") is None else str(rec.get("delist_reason"))
            ),
        )
    return assets


def _load_earnings(raw_dir: Path) -> dict[str, tuple[EarningsEvent, ...]]:
    path = raw_dir / "earnings.parquet"
    if not path.is_file():
        return {}
    frame = read_parquet(path)
    if frame.empty:
        return {}
    grouped: dict[str, list[EarningsEvent]] = {}
    for rec in frame.to_dict("records"):
        aid = str(rec["asset_id"])
        available = rec.get("available_ts")
        available_ts: datetime | None
        if available is None or pd.isna(available):
            available_ts = None
        else:
            ts = pd.Timestamp(available)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            available_ts = ts.to_pydatetime()
        event = EarningsEvent(
            asset_id=aid,
            earnings_date=pd.Timestamp(rec["earnings_date"]).date(),
            is_confirmed=bool(rec.get("is_confirmed", False)),
            available_ts=available_ts,
            timing=str(rec.get("timing", "UNKNOWN")),
        )
        grouped.setdefault(aid, []).append(event)
    return {aid: tuple(events) for aid, events in grouped.items()}


def _load_actions(raw_dir: Path) -> tuple[CorporateAction, ...]:
    path = raw_dir / "actions.parquet"
    if not path.is_file():
        return ()
    frame = read_parquet(path)
    if frame.empty:
        return ()
    out: list[CorporateAction] = []
    for rec in frame.to_dict("records"):
        raw_type = str(rec.get("action_type", "DIVIDEND"))
        try:
            action_type = ActionType(raw_type)
        except ValueError:
            continue
        out.append(
            CorporateAction(
                asset_id=str(rec["asset_id"]),
                ex_date=pd.Timestamp(rec["ex_date"]).date(),
                action_type=action_type,
                split_ratio=float(rec.get("split_ratio", 1.0) or 1.0),
                cash_amount=float(rec.get("cash_amount", 0.0) or 0.0),
                new_symbol=None if rec.get("new_symbol") is None else str(rec["new_symbol"]),
            )
        )
    return tuple(out)


def _as_aware(value: object) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    out = ts.to_pydatetime()
    return out if isinstance(out, datetime) else None
