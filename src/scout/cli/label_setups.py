from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from scout.config.hashing import config_hash
from scout.config.loader import load_config
from scout.config.schema import ScoutConfig
from scout.data.calendar import reference_calendar_path
from scout.data.ingest import load_tickers, read_candidate_pairs
from scout.data.parquet_source import ParquetCandleSource
from scout.data.store import read_parquet, write_parquet_atomic
from scout.domain.market import BENCHMARK_COLUMNS, BenchmarkPanel, EarningsEvent
from scout.features.engine import compute_features
from scout.scoring.labeling import (
    LABEL_COLUMNS,
    empty_label_frame,
    label_setups,
)
from scout.strategies.registry import build_strategies
from scout.utils.errors import ScoutConfigError, ScoutDataError
from scout.utils.logging import configure_logging


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--strategy", default=None, help="Restrict to one strategy_id")


def run(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    digest = config_hash(cfg)
    configure_logging(f"label-{digest[:12]}", level=cfg.logging.level)
    log = logging.getLogger("scout.cli.label_setups")
    log.info("resolved config hash=%s", digest)
    print(f"config_hash={digest}")

    strategies = list(build_strategies(cfg.strategies))
    if args.strategy is not None:
        strategies = [s for s in strategies if s.strategy_id == args.strategy]
        if not strategies:
            known = sorted(sc.strategy_id for sc in cfg.strategies if sc.enabled)
            raise ScoutConfigError(
                f"unknown or disabled strategy_id {args.strategy!r}; known: {known}"
            )
    strategy_ids = [s.strategy_id for s in strategies]
    if not strategy_ids:
        raise ScoutConfigError("no enabled strategies to label")

    candidates = read_candidate_pairs(Path(cfg.universe.candidates_file))
    asset_ids = [str(x) for x in candidates["asset_id"].tolist()]
    source = ParquetCandleSource(cfg.data.processed_dir)
    panel = source.load_panel(
        asset_ids,
        cfg.data.decision_timeframe,
        cfg.period.start,
        cfg.period.end,
    )
    snapshots = _load_snapshots(cfg)
    benchmark = _load_benchmark(Path(cfg.data.processed_dir))
    features = compute_features(panel, benchmark, snapshots, cfg.features)

    calendar = _load_calendar(cfg)
    tickers = load_tickers(Path(cfg.data.raw_dir))
    is_etf = _etf_map(tickers)
    earnings_by_asset = _load_earnings(Path(cfg.data.raw_dir))

    labeled = label_setups(
        features,
        panel,
        strategies,
        cfg.labeling,
        snapshots=snapshots,
        earnings_by_asset=earnings_by_asset,
        is_etf=is_etf,
        calendar=calendar,
    )
    labels_dir = Path(cfg.labeling.labels_dir)
    _write_by_strategy(labeled, labels_dir, strategy_ids)
    for sid in strategy_ids:
        part = labeled.loc[labeled["strategy_id"] == sid] if not labeled.empty else labeled
        n = 0 if part.empty else len(part)
        n_open = 0 if part.empty else int((part["outcome"] == "OPEN").sum())
        path = labels_dir / f"setups_{sid}.parquet"
        print(f"strategy_id={sid}")
        print(f"setups={n}")
        print(f"open={n_open}")
        print(f"resolved={n - n_open}")
        print(f"labels_path={path.as_posix()}")
    return 0


def _load_snapshots(cfg: ScoutConfig) -> pd.DataFrame:
    path = Path(cfg.universe.snapshots_path)
    if not path.is_file():
        raise ScoutDataError(f"universe snapshots not found: {path}; run build-universe first")
    return read_parquet(path)


def _load_benchmark(processed_dir: Path) -> BenchmarkPanel:
    path = processed_dir.parent / "reference" / "benchmark_1d.parquet"
    if not path.is_file():
        raise ScoutDataError(
            f"benchmark not found: {path}; expected data/reference/benchmark_1d.parquet"
        )
    frame = read_parquet(path)
    missing = [c for c in BENCHMARK_COLUMNS if c not in frame.columns]
    if missing:
        raise ScoutDataError(f"benchmark missing columns: {missing}")
    return BenchmarkPanel(frame.loc[:, list(BENCHMARK_COLUMNS)])


def _load_calendar(cfg: ScoutConfig) -> pd.DataFrame:
    path = reference_calendar_path(Path(cfg.data.processed_dir), cfg.data.calendar)
    if not path.is_file():
        raise ScoutDataError(f"calendar not found: {path}; run adjust first")
    return read_parquet(path)


def _etf_map(tickers: pd.DataFrame) -> dict[str, bool]:
    if tickers.empty or "asset_id" not in tickers.columns:
        return {}
    has_etf = "is_etf" in tickers.columns
    out: dict[str, bool] = {}
    for rec in tickers.to_dict("records"):
        aid = str(rec["asset_id"])
        out[aid] = bool(rec["is_etf"]) if has_etf else False
    return out


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
            earnings_date=_as_date(rec["earnings_date"]),
            is_confirmed=bool(rec.get("is_confirmed", False)),
            available_ts=available_ts,
            timing=str(rec.get("timing", "UNKNOWN")),
        )
        grouped.setdefault(aid, []).append(event)
    return {aid: tuple(events) for aid, events in grouped.items()}


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed: date = pd.Timestamp(value).date()
    return parsed


def _write_by_strategy(
    frame: pd.DataFrame,
    labels_dir: Path,
    strategy_ids: Sequence[str],
) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    for sid in strategy_ids:
        if frame.empty:
            part = empty_label_frame()
        else:
            part = frame.loc[frame["strategy_id"] == sid].reset_index(drop=True)
            part = empty_label_frame() if part.empty else part.loc[:, list(LABEL_COLUMNS)]
        write_parquet_atomic(labels_dir / f"setups_{sid}.parquet", part)
