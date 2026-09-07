"""Engine corner cases from 11-BACKTEST_ENGINE.md §7."""

from __future__ import annotations

import pandas as pd
import pytest

from scout.backtest.engine import BacktestEngine
from scout.backtest.sim_broker import SimBroker
from scout.config.hashing import config_hash
from scout.domain.edge import EdgeTable
from scout.domain.enums import RejectionReason
from scout.sentiment.null_source import NullSentimentSource
from scout.storage.decision_sink import NullDecisionSink
from scout.strategies.registry import build_strategies
from scout.utils.errors import ScoutConfigError, ScoutError
from tests.backtest_fixtures import (
    MemoryCandleSource,
    assets,
    benchmark_panel,
    engine_config,
    market_panel,
    snapshot_book,
    snapshot_frame,
    timestamps,
    usable_edge_table,
)


def test_config_hash_mismatch_fails_at_startup(tmp_path) -> None:
    stamps = timestamps(20)
    cfg = engine_config(tmp_path, stamps, warmup_index=5)
    symbols = ("AAA", "BBB")
    panel = market_panel(symbols, stamps)
    engine = BacktestEngine(
        candles=MemoryCandleSource(panel),
        sentiment=NullSentimentSource(),
        broker=SimBroker(cfg.costs, cfg.portfolio, panel=panel),
        strategies=build_strategies(cfg.strategies),
        edge_table=EdgeTable((), config_hash="deadbeef"),
        sink=NullDecisionSink(),
        universe=snapshot_book(symbols, stamps[0]),
        assets=assets(symbols),
        cfg=cfg,
        calendar=pd.DataFrame(),
        benchmark=benchmark_panel(stamps),
        run_id="test0001-engine",
    )
    with pytest.raises(ScoutConfigError, match="config_hash"):
        engine.run()


def test_universe_snapshot_missing_records_no_entries(tmp_path) -> None:
    from scout.universe.build import SnapshotBook

    stamps = timestamps(20)
    cfg = engine_config(tmp_path, stamps, warmup_index=5)
    symbols = ("AAA", "BBB")
    panel = market_panel(symbols, stamps)
    sink = NullDecisionSink()
    engine = BacktestEngine(
        candles=MemoryCandleSource(panel),
        sentiment=NullSentimentSource(),
        broker=SimBroker(cfg.costs, cfg.portfolio, panel=panel),
        strategies=build_strategies(cfg.strategies),
        edge_table=usable_edge_table(cfg, stamps[0]),
        sink=sink,
        universe=SnapshotBook(pd.DataFrame()),
        assets=assets(symbols),
        cfg=cfg,
        calendar=pd.DataFrame(),
        benchmark=benchmark_panel(stamps),
        run_id="test0002-engine",
    )
    result = engine.run()
    sink.flush()
    reasons = [r.rejection_reason for r in sink.records]
    assert RejectionReason.NOT_IN_UNIVERSE in reasons
    assert result.trades == ()


def test_edge_table_missing_bin_is_insufficient_samples(tmp_path) -> None:
    n = 500
    stamps = timestamps(n)
    cfg = engine_config(tmp_path, stamps, warmup_index=400)
    symbols = ("AAA", "BBB", "CCC")
    panel = market_panel(symbols, stamps)
    sink = NullDecisionSink()
    engine = BacktestEngine(
        candles=MemoryCandleSource(panel),
        sentiment=NullSentimentSource(),
        broker=SimBroker(cfg.costs, cfg.portfolio, panel=panel),
        strategies=build_strategies(cfg.strategies),
        edge_table=EdgeTable((), config_hash=config_hash(cfg)),
        sink=sink,
        universe=snapshot_book(symbols, stamps[0]),
        assets=assets(symbols),
        cfg=cfg,
        calendar=pd.DataFrame(),
        benchmark=benchmark_panel(stamps),
        run_id="test0003-engine",
    )
    engine.run()
    sink.flush()
    assert any(
        r.rejection_reason is RejectionReason.INSUFFICIENT_BIN_SAMPLES for r in sink.records
    )


def test_zero_equity_halts(tmp_path) -> None:
    stamps = timestamps(10)
    cfg = engine_config(tmp_path, stamps, warmup_index=1)
    cfg = cfg.model_copy(
        update={"portfolio": cfg.portfolio.model_copy(update={"initial_equity_usd": 0})}
    )
    symbols = ("AAA",)
    panel = market_panel(symbols, stamps)
    engine = BacktestEngine(
        candles=MemoryCandleSource(panel),
        sentiment=NullSentimentSource(),
        broker=SimBroker(cfg.costs, cfg.portfolio, panel=panel),
        strategies=build_strategies(cfg.strategies),
        edge_table=usable_edge_table(cfg, stamps[0]),
        sink=NullDecisionSink(),
        universe=snapshot_book(symbols, stamps[0]),
        assets=assets(symbols),
        cfg=cfg,
        calendar=pd.DataFrame(),
        benchmark=benchmark_panel(stamps),
        run_id="test0004-engine",
    )
    with pytest.raises(ScoutError, match="blew up"):
        engine.run()


def test_two_strategies_same_symbol_second_already_in_position(tmp_path) -> None:
    n = 500
    stamps = timestamps(n)
    cfg = engine_config(tmp_path, stamps, warmup_index=400)
    symbols = ("AAA", "BBB", "CCC")
    panel = market_panel(symbols, stamps)
    sink = NullDecisionSink()
    engine = BacktestEngine(
        candles=MemoryCandleSource(panel),
        sentiment=NullSentimentSource(),
        broker=SimBroker(cfg.costs, cfg.portfolio, panel=panel),
        strategies=build_strategies(cfg.strategies),
        edge_table=usable_edge_table(cfg, stamps[0]),
        sink=sink,
        universe=snapshot_book(symbols, stamps[0]),
        assets=assets(symbols),
        cfg=cfg,
        calendar=pd.DataFrame(),
        benchmark=benchmark_panel(stamps),
        run_id="test0005-engine",
    )
    engine.run()
    sink.flush()
    assert any(r.rejection_reason is RejectionReason.ALREADY_IN_POSITION for r in sink.records)


def test_data_gap_when_no_next_bar(tmp_path) -> None:
    n = 450
    stamps = timestamps(n)
    cfg = engine_config(tmp_path, stamps, warmup_index=400)
    cfg = cfg.model_copy(
        update={"period": cfg.period.model_copy(update={"warmup_end": stamps[-1]})}
    )
    symbols = ("AAA", "BBB", "CCC")
    panel = market_panel(symbols, stamps)
    sink = NullDecisionSink()
    engine = BacktestEngine(
        candles=MemoryCandleSource(panel),
        sentiment=NullSentimentSource(),
        broker=SimBroker(cfg.costs, cfg.portfolio, panel=panel),
        strategies=build_strategies(cfg.strategies),
        edge_table=usable_edge_table(cfg, stamps[0]),
        sink=sink,
        universe=snapshot_book(symbols, stamps[0]),
        assets=assets(symbols),
        cfg=cfg,
        calendar=pd.DataFrame(),
        benchmark=benchmark_panel(stamps),
        run_id="test0006-engine",
    )
    engine.run()
    sink.flush()
    assert any(r.rejection_reason is RejectionReason.DATA_GAP for r in sink.records)


def test_atr_zero_or_nan_is_insufficient_history(tmp_path) -> None:
    stamps = timestamps(20)
    cfg = engine_config(tmp_path, stamps, warmup_index=2)
    symbols = ("AAA",)
    panel = market_panel(symbols, stamps)
    sink = NullDecisionSink()
    engine = BacktestEngine(
        candles=MemoryCandleSource(panel),
        sentiment=NullSentimentSource(),
        broker=SimBroker(cfg.costs, cfg.portfolio, panel=panel),
        strategies=build_strategies(cfg.strategies),
        edge_table=usable_edge_table(cfg, stamps[0]),
        sink=sink,
        universe=snapshot_book(symbols, stamps[0]),
        assets=assets(symbols),
        cfg=cfg,
        calendar=pd.DataFrame(),
        benchmark=benchmark_panel(stamps),
        run_id="test0007-engine",
    )
    engine.run()
    sink.flush()
    assert any(
        r.rejection_reason is RejectionReason.INSUFFICIENT_HISTORY for r in sink.records
    )


def test_exits_applied_before_entries(tmp_path) -> None:
    n = 500
    stamps = timestamps(n)
    cfg = engine_config(tmp_path, stamps, warmup_index=400)
    symbols = ("AAA", "BBB", "CCC")

    def close_fn(symbol: str, i: int, _ts: object) -> float:
        base = 20.0 + 0.08 * i + 5.0 * (0 if symbol == "AAA" else 1)
        if i > 430:
            return base * 0.5
        return base

    panel = market_panel(symbols, stamps, close_fn=close_fn)
    sink = NullDecisionSink()
    engine = BacktestEngine(
        candles=MemoryCandleSource(panel),
        sentiment=NullSentimentSource(),
        broker=SimBroker(cfg.costs, cfg.portfolio, panel=panel),
        strategies=build_strategies(cfg.strategies),
        edge_table=usable_edge_table(cfg, stamps[0]),
        sink=sink,
        universe=snapshot_book(symbols, stamps[0]),
        assets=assets(symbols),
        cfg=cfg,
        calendar=pd.DataFrame(),
        benchmark=benchmark_panel(stamps),
        run_id="test0008-engine",
    )
    result = engine.run()
    assert result.n_decisions_considered > 0


def test_ineligible_names_count_in_funnel_not_parquet(tmp_path) -> None:
    from scout.universe.build import SnapshotBook

    stamps = timestamps(20)
    cfg = engine_config(tmp_path, stamps, warmup_index=5)
    symbols = ("AAA", "BBB")
    frame = snapshot_frame(symbols, stamps[0])
    frame.loc[frame["symbol"] == "BBB", "eligible"] = False
    frame.loc[frame["symbol"] == "BBB", "reason"] = RejectionReason.LOW_LIQUIDITY.value
    panel = market_panel(symbols, stamps)
    sink = NullDecisionSink()
    engine = BacktestEngine(
        candles=MemoryCandleSource(panel),
        sentiment=NullSentimentSource(),
        broker=SimBroker(cfg.costs, cfg.portfolio, panel=panel),
        strategies=build_strategies(cfg.strategies),
        edge_table=usable_edge_table(cfg, stamps[0]),
        sink=sink,
        universe=SnapshotBook(frame),
        assets=assets(symbols),
        cfg=cfg,
        calendar=pd.DataFrame(),
        benchmark=benchmark_panel(stamps),
        run_id="test0009-engine",
    )
    result = engine.run()
    sink.flush()
    n_trade = sum(1 for ts in stamps if ts >= cfg.period.warmup_end)
    n_strategies = len(build_strategies(cfg.strategies))
    expected = n_trade * n_strategies
    assert result.n_rejections_by_reason.get(RejectionReason.LOW_LIQUIDITY, 0) == expected
    assert all(r.symbol != "BBB" for r in sink.records)
    assert result.n_decisions_considered >= expected
