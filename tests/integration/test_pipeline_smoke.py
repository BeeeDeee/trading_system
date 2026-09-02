"""Full-pipeline smoke: 3 symbols, 2000 bars, both strategies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scout.backtest.engine import BacktestEngine
from scout.backtest.sim_broker import SimBroker
from scout.sentiment.null_source import NullSentimentSource
from scout.storage.decision_sink import ParquetDecisionSink
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

N_BARS = 2000
SYMBOLS = ("AAA", "BBB", "CCC")


def _run(tmp_path: Path) -> tuple[object, Path]:
    stamps = timestamps(N_BARS)
    cfg = engine_config(tmp_path, stamps, warmup_index=400)
    panel = market_panel(SYMBOLS, stamps)
    out = Path(cfg.audit.results_dir) / "smoke-run"
    out.mkdir(parents=True, exist_ok=True)
    sink = ParquetDecisionSink(out / "decisions.parquet", flush_every=100)
    engine = BacktestEngine(
        candles=MemoryCandleSource(panel),
        sentiment=NullSentimentSource(),
        broker=SimBroker(cfg.costs, cfg.portfolio, panel=panel),
        strategies=build_strategies(cfg.strategies),
        edge_table=usable_edge_table(cfg, stamps[0]),
        sink=sink,
        universe=snapshot_book(SYMBOLS, stamps[0]),
        assets=assets(SYMBOLS),
        cfg=cfg,
        calendar=pd.DataFrame(),
        benchmark=benchmark_panel(stamps),
        run_id="smoke-run",
    )
    result = engine.run()
    write_run_outputs(result, "smoke-run", cfg, used_bins=engine.used_bins_frame())
    return result, Path(cfg.audit.results_dir) / "smoke-run"


def test_pipeline_smoke(tmp_path: Path) -> None:
    result, out = _run(tmp_path)
    assert result.n_decisions_considered > 0
    assert len(result.trades) > 0
    required = (
        "config.yaml",
        "config_hash.txt",
        "metrics.json",
        "trades.csv",
        "equity.csv",
        "decisions.parquet",
        "funnel.csv",
        "bins.csv",
        "run.log",
    )
    for name in required:
        assert (out / name).is_file(), name
    for plot in (
        "equity.png",
        "drawdown.png",
        "summary.png",
        "calibration.png",
        "monthly_returns.png",
        "regime_breakdown.png",
    ):
        assert (out / "plots" / plot).is_file(), plot
    funnel = pd.read_csv(out / "funnel.csv")
    assert int(funnel["n"].sum()) == result.n_decisions_considered
    eq = result.equity_curve
    assert not eq.empty
    assert float(eq.iloc[-1]) > 0.0
