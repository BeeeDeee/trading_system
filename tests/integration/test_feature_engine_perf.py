"""M1.6 acceptance: 1000 symbols x 25 years of daily bars in under 90 seconds."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from scout.config.schema import FeaturesConfig
from scout.domain.features import FEATURE_COLUMNS
from scout.domain.market import BENCHMARK_COLUMNS, MARKET_COLUMNS, BenchmarkPanel, MarketPanel
from scout.features.engine import compute_features

N_SYMBOLS = 1_000
N_YEARS = 25
SESSIONS_PER_YEAR = 252
MAX_SECONDS = 90.0
FIRST_YEAR = 2000


def _build_panel(n_symbols: int, n_ts: int) -> tuple[MarketPanel, BenchmarkPanel, pd.DataFrame]:
    n = n_ts * n_symbols
    asset_ids = np.array([f"S{i:04d}" for i in range(n_symbols)], dtype=object)
    base = np.datetime64(f"{FIRST_YEAR}-01-03T21:00:00")
    ts_one = base + np.arange(n_ts, dtype=np.int32).astype("timedelta64[D]")
    ts = pd.DatetimeIndex(np.repeat(ts_one, n_symbols)).tz_localize("UTC")
    ids = np.tile(asset_ids, n_ts)
    session_index = np.repeat(np.arange(n_ts, dtype=np.int32), n_symbols)
    sym_num = np.tile(np.arange(n_symbols, dtype=np.float64), n_ts)
    sess = np.repeat(np.arange(n_ts, dtype=np.float64), n_symbols)
    close = 20.0 + 0.01 * sym_num + 0.001 * sess
    frame = pd.DataFrame(
        {
            "asset_id": ids,
            "symbol": ids,
            "ts": ts,
            "session_index": session_index,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "close_raw": close,
            "volume": np.full(n, 1_000.0, dtype=np.float64),
            "dollar_volume": close * 1_000.0,
            "is_suspect": np.zeros(n, dtype=bool),
        },
        columns=list(MARKET_COLUMNS),
    )
    panel = MarketPanel(frame)

    spy = 200.0 + 0.02 * np.arange(n_ts, dtype=np.float64)
    bench_ts = pd.DatetimeIndex(ts_one).tz_localize("UTC")
    bench_frame = pd.DataFrame(
        {
            "ts": bench_ts,
            "session_index": np.arange(n_ts, dtype=np.int32),
            "open": spy,
            "high": spy + 0.5,
            "low": spy - 0.5,
            "close": spy,
            "vix_close": np.full(n_ts, 18.0, dtype=np.float64),
            "vix9d_close": np.full(n_ts, 17.0, dtype=np.float64),
            "vix3m_close": np.full(n_ts, 19.0, dtype=np.float64),
        },
        columns=list(BENCHMARK_COLUMNS),
    )
    benchmark = BenchmarkPanel(bench_frame)
    snaps = pd.DataFrame(
        {
            "ts": [bench_ts[0]] * n_symbols,
            "symbol": asset_ids.tolist(),
            "eligible": [True] * n_symbols,
        }
    )
    return panel, benchmark, snaps


def test_compute_features_1000_symbols_25_years_budget() -> None:
    n_ts = SESSIONS_PER_YEAR * N_YEARS
    panel, benchmark, snaps = _build_panel(N_SYMBOLS, n_ts)
    t0 = time.perf_counter()
    out = compute_features(panel, benchmark, snaps, FeaturesConfig())
    elapsed = time.perf_counter() - t0
    assert tuple(out.frame.columns) == FEATURE_COLUMNS
    assert len(out.frame) == N_SYMBOLS * n_ts
    assert elapsed < MAX_SECONDS, f"compute_features took {elapsed:.3f}s (limit {MAX_SECONDS}s)"
