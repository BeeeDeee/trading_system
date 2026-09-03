"""Run output files, plot DPI, config hash round-trip, one registry row."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pandas as pd
import yaml

from scout.backtest.engine import BacktestEngine
from scout.backtest.sim_broker import SimBroker
from scout.config.hashing import config_hash
from scout.config.schema import ScoutConfig
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

SYMBOLS = ("AAA", "BBB", "CCC")


def _png_dpi(path: Path) -> float:
    data = path.read_bytes()
    marker = data.find(b"pHYs")
    if marker < 0:
        return 0.0
    ppm = struct.unpack(">I", data[marker + 4 : marker + 8])[0]
    return ppm * 0.0254


def test_run_outputs(tmp_path: Path) -> None:
    stamps = timestamps(500)
    cfg = engine_config(tmp_path, stamps, warmup_index=400)
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
        run_id="out-run",
    )
    result = engine.run()
    out = write_run_outputs(result, "out-run", cfg, used_bins=engine.used_bins_frame())
    names = (
        "config.yaml",
        "config_hash.txt",
        "metrics.json",
        "trades.csv",
        "equity.csv",
        "funnel.csv",
        "bins.csv",
        "run.log",
    )
    for name in names:
        assert (out / name).is_file(), name
    for plot in (
        "equity.png",
        "drawdown.png",
        "summary.png",
        "calibration.png",
        "monthly_returns.png",
        "regime_breakdown.png",
    ):
        path = out / "plots" / plot
        assert path.is_file(), plot
        assert _png_dpi(path) >= 149.0, f"{plot} dpi={_png_dpi(path)}"
    written_hash = (out / "config_hash.txt").read_text(encoding="utf-8").strip()
    assert written_hash == config_hash(cfg)
    raw = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    reloaded = ScoutConfig.model_validate(raw)
    assert config_hash(reloaded) == written_hash
    registry = pd.read_csv(Path(cfg.research.registry_path))
    assert len(registry) == 1
    assert str(registry.iloc[0]["run_id"]) == "out-run"
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    for key in (
        "total_return_pct",
        "sharpe",
        "deflated_sharpe",
        "cost_drag_pct",
        "n_trials",
    ):
        assert key in metrics
