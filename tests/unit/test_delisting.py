"""Delisting: force-close at last close; no re-entry after data ends."""

from __future__ import annotations

import pandas as pd

from scout.backtest.engine import BacktestEngine
from scout.backtest.sim_broker import SimBroker
from scout.domain.market import MarketPanel
from scout.sentiment.null_source import NullSentimentSource
from scout.storage.decision_sink import NullDecisionSink
from scout.strategies.registry import build_strategies
from tests.backtest_fixtures import (
    MemoryCandleSource,
    assets,
    benchmark_panel,
    engine_config,
    market_panel,
    snapshot_book,
    timestamps,
    usable_edge_table,
)


def test_delisted_position_force_closed(tmp_path) -> None:
    n = 480
    stamps = timestamps(n)
    cfg = engine_config(tmp_path, stamps, warmup_index=400)
    symbols = ("AAA", "BBB", "CCC")
    panel = market_panel(symbols, stamps)
    cutoff = stamps[450]
    frame = panel.frame
    keep = ~((frame["symbol"].astype(str) == "AAA") & (frame["ts"] > pd.Timestamp(cutoff)))
    panel = MarketPanel(frame.loc[keep].reset_index(drop=True))
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
        run_id="delist001",
    )
    result = engine.run()
    aaa = [t for t in result.trades if t.symbol == "AAA"]
    assert aaa, "expected AAA trades including a delisting close"
    assert any(t.exit_ts >= cutoff for t in aaa)


def test_delisted_symbol_not_reentered(tmp_path) -> None:
    n = 480
    stamps = timestamps(n)
    cfg = engine_config(tmp_path, stamps, warmup_index=400)
    symbols = ("AAA", "BBB")
    panel = market_panel(symbols, stamps)
    cutoff = stamps[420]
    frame = panel.frame
    keep = ~((frame["symbol"].astype(str) == "AAA") & (frame["ts"] > pd.Timestamp(cutoff)))
    panel = MarketPanel(frame.loc[keep].reset_index(drop=True))
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
        run_id="delist002",
    )
    result = engine.run()
    after = [t for t in result.trades if t.symbol == "AAA" and t.entry_ts > cutoff]
    assert after == []
