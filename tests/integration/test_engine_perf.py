"""Backtest loop budget: 6300 sessions x 1000 symbols in under 180 seconds."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from scout.backtest.engine import BacktestEngine
from scout.backtest.sim_broker import SimBroker
from scout.config.hashing import config_hash
from scout.config.schema import (
    AuditConfig,
    GatesConfig,
    PeriodConfig,
    ResearchConfig,
    ScoutConfig,
    UniverseConfig,
)
from scout.domain.edge import EdgeTable
from scout.domain.enums import RejectionReason
from scout.domain.features import FEATURE_COLUMNS, FeaturePanel
from scout.domain.market import MARKET_COLUMNS, BenchmarkPanel, MarketPanel
from scout.domain.universe import UniverseEntry, UniverseSnapshot
from scout.sentiment.null_source import NullSentimentSource
from scout.strategies.registry import build_strategies
from tests.backtest_fixtures import MemoryCandleSource, assets, write_candidates

N_SYMBOLS = 1_000
N_SESSIONS = 6_300
MAX_SECONDS = 180.0
START = datetime(2000, 1, 3, 21, 0, tzinfo=UTC)


class _DiscardSink:
    """Protocol-compatible sink that drops records. NullDecisionSink retains them."""

    def write(self, records: object) -> None:
        del records

    def flush(self) -> None:
        return


class _FixedUniverse:
    def __init__(self, snap: UniverseSnapshot, frame: pd.DataFrame) -> None:
        self._snap = snap
        self._frame = frame

    def load_all(self) -> pd.DataFrame:
        return self._frame

    def snapshot_at(self, ts: datetime) -> UniverseSnapshot | None:
        del ts
        return self._snap


def _tiny_features() -> FeaturePanel:
    ts = pd.DatetimeIndex([START], tz="UTC")
    data: dict[str, object] = {name: [] for name in FEATURE_COLUMNS}
    data["symbol"] = ["S0000"]
    data["ts"] = ts
    for name in FEATURE_COLUMNS:
        if name in {"symbol", "ts"}:
            continue
        if name in {"regime", "vol_bucket", "market_regime"}:
            data[name] = ["UNKNOWN"]
        elif name in {"spy_above_ma", "is_warm"}:
            data[name] = [False]
        elif name in {"xs_population", "bars_available", "bars_since_gap"}:
            data[name] = [0]
        else:
            data[name] = [float("nan")]
    return FeaturePanel(pd.DataFrame(data, columns=list(FEATURE_COLUMNS)))


def test_backtest_loop_6300x1000_budget(tmp_path) -> None:
    symbols = [f"S{i:04d}" for i in range(N_SYMBOLS)]
    n_ts = N_SESSIONS
    # One row per timestamp, one dummy symbol: the loop still visits 6,300 ts
    # and 1,000 snapshot names. Building a full 6.3M-row panel is the M1.3
    # budget; this test is the *loop*.
    ts_one = pd.DatetimeIndex(
        [START + timedelta(days=i) for i in range(n_ts)], tz="UTC"
    )
    close = 20.0 + 0.001 * np.arange(n_ts, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "asset_id": ["S0000"] * n_ts,
            "symbol": ["S0000"] * n_ts,
            "ts": ts_one,
            "session_index": np.arange(n_ts, dtype=np.int32),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "close_raw": close,
            "volume": np.full(n_ts, 1_000.0),
            "dollar_volume": close * 1_000.0,
            "is_suspect": np.zeros(n_ts, dtype=bool),
        },
        columns=list(MARKET_COLUMNS),
    )
    panel = MarketPanel(frame)
    bench_frame = pd.DataFrame(
        {
            "ts": ts_one,
            "session_index": np.arange(n_ts, dtype=np.int32),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "vix_close": np.full(n_ts, 18.0),
            "vix9d_close": np.full(n_ts, 17.0),
            "vix3m_close": np.full(n_ts, 19.0),
        }
    )
    from scout.domain.market import BENCHMARK_COLUMNS

    benchmark = BenchmarkPanel(bench_frame.loc[:, list(BENCHMARK_COLUMNS)])
    entries = {
        s: UniverseEntry(
            symbol=s,
            eligible=False,
            reason=RejectionReason.LOW_LIQUIDITY,
            adv_usd_30=1.0,
            spread_bps_est=50.0,
            bars_available=1,
            listed_days=1.0,
        )
        for s in symbols
    }
    snap = UniverseSnapshot(ts=START, entries=entries)
    snaps_frame = pd.DataFrame(
        {"ts": [START] * N_SYMBOLS, "symbol": symbols, "eligible": [False] * N_SYMBOLS}
    )
    cand = tmp_path / "universe_candidates.txt"
    write_candidates(cand, ["S0000"])
    warmup = START + timedelta(days=200)
    cfg = ScoutConfig(
        period=PeriodConfig(start=START, warmup_end=warmup, end=ts_one[-1].to_pydatetime()),
        universe=UniverseConfig(candidates_file=str(cand), min_bars_since_gap=0),
        gates=GatesConfig(min_xs_population=1),
        audit=AuditConfig(results_dir=str(tmp_path / "results"), flush_every_cycles=10_000),
        research=ResearchConfig(registry_path=str(tmp_path / "registry.csv")),
    )
    engine = BacktestEngine(
        candles=MemoryCandleSource(panel),
        sentiment=NullSentimentSource(),
        broker=SimBroker(cfg.costs, cfg.portfolio, panel=panel),
        strategies=build_strategies(cfg.strategies),
        edge_table=EdgeTable((), config_hash=config_hash(cfg)),
        sink=_DiscardSink(),
        universe=_FixedUniverse(snap, snaps_frame),
        assets=assets(["S0000"]),
        cfg=cfg,
        calendar=pd.DataFrame(),
        benchmark=benchmark,
        run_id="perf-loop",
        features=_tiny_features(),
    )
    t0 = time.perf_counter()
    result = engine.run()
    elapsed = time.perf_counter() - t0
    assert result.n_decisions_considered > 0
    assert elapsed < MAX_SECONDS, f"backtest loop took {elapsed:.3f}s (limit {MAX_SECONDS}s)"
