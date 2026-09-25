"""Same config twice → byte-identical metrics.json and trades.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scout.backtest.engine import BacktestEngine
from scout.backtest.sim_broker import SimBroker
from scout.sentiment.null_source import NullSentimentSource
from scout.storage.decision_sink import NullDecisionSink
from scout.storage.run_outputs import write_run_outputs
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

N_BARS = 600
SYMBOLS = ("AAA", "BBB", "CCC")


def _one(tmp: Path, run_id: str) -> Path:
    stamps = timestamps(N_BARS)
    cfg = engine_config(tmp, stamps, warmup_index=400)
    panel = market_panel(SYMBOLS, stamps)
    engine = BacktestEngine(
        candles=MemoryCandleSource(panel),
        sentiment=NullSentimentSource(),
        broker=SimBroker(cfg.costs, cfg.portfolio, panel=panel),
        strategies=build_strategies(cfg.strategies),
        edge_table=usable_edge_table(cfg, stamps[0]),
        sink=NullDecisionSink(),
        universe=snapshot_book(SYMBOLS, stamps[0]),
        assets=assets(SYMBOLS),
        cfg=cfg,
        calendar=pd.DataFrame(),
        benchmark=benchmark_panel(stamps),
        run_id=run_id,
    )
    result = engine.run()
    return write_run_outputs(result, run_id, cfg, used_bins=engine.used_bins_frame())


def test_determinism(tmp_path: Path) -> None:
    a = _one(tmp_path / "a", "det-a")
    b = _one(tmp_path / "b", "det-b")
    assert (a / "metrics.json").read_bytes() == (b / "metrics.json").read_bytes()
    assert (a / "trades.csv").read_bytes() == (b / "trades.csv").read_bytes()
