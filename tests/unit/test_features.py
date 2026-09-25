"""Golden-value tests for per-symbol indicators and cross-sectional ranks."""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from scout.config.schema import FeaturesConfig
from scout.domain.enums import VolBucket
from scout.domain.features import FEATURE_COLUMNS, FeatureRow
from scout.domain.market import BENCHMARK_COLUMNS, MARKET_COLUMNS, BenchmarkPanel, MarketPanel
from scout.features.cross_sectional import cross_sectional_ranks, eligible_mask
from scout.features.engine import compute_features
from scout.features.indicators import (
    atr_percentile,
    atr_wilder,
    classify_vol_bucket,
    donchian_high,
    donchian_low,
    efficiency_ratio,
    ema,
    required_warmup_bars,
    rolling_beta,
    slope_atr,
    true_range,
)
from scout.utils.errors import ScoutLookaheadError

REL = 1e-9


def _python_true_range(
    high: list[float], low: list[float], close: list[float]
) -> list[float]:
    out: list[float] = []
    for i in range(len(close)):
        hl = high[i] - low[i]
        if i == 0:
            out.append(hl)
            continue
        prev = close[i - 1]
        out.append(max(hl, abs(high[i] - prev), abs(low[i] - prev)))
    return out


def _python_wilder(tr: list[float], n: int) -> list[float | None]:
    alpha = 1.0 / n
    acc = tr[0]
    out: list[float | None] = [None] * len(tr)
    for i, x in enumerate(tr):
        acc = x if i == 0 else alpha * x + (1.0 - alpha) * acc
        if i >= n - 1:
            out[i] = acc
    return out


def test_true_range_golden() -> None:
    # Five bars. First bar has no previous close, so TR = high - low.
    # idx  high  low  close  TR
    # 0    12    9    11     12-9 = 3
    # 1    13    10   12     max(3, |13-11|=2, |10-11|=1) = 3
    # 2    10    7    9      max(3, |10-12|=2, |7-12|=5)  = 5   gap down
    # 3    15    9    14     max(6, |15-9|=6,  |9-9|=0)   = 6
    # 4    16    13   15     max(3, |16-14|=2, |13-14|=1) = 3
    high = pd.Series([12.0, 13.0, 10.0, 15.0, 16.0])
    low = pd.Series([9.0, 10.0, 7.0, 9.0, 13.0])
    close = pd.Series([11.0, 12.0, 9.0, 14.0, 15.0])
    expected = [3.0, 3.0, 5.0, 6.0, 3.0]
    got = true_range(high, low, close)
    for i, value in enumerate(expected):
        assert got.iloc[i] == pytest.approx(value, rel=REL)


def test_atr_wilder_golden() -> None:
    high = [
        10.0,
        12.0,
        11.0,
        13.0,
        12.0,
        14.0,
        16.0,
        15.0,
        17.0,
        19.0,
        18.0,
        20.0,
        22.0,
        21.0,
        23.0,
        25.0,
        24.0,
        26.0,
        28.0,
        27.0,
    ]
    low = [
        8.0,
        9.0,
        9.0,
        10.0,
        10.0,
        11.0,
        13.0,
        12.0,
        14.0,
        15.0,
        15.0,
        16.0,
        18.0,
        17.0,
        19.0,
        20.0,
        20.0,
        22.0,
        23.0,
        22.0,
    ]
    close = [
        9.0,
        11.0,
        10.0,
        12.0,
        11.0,
        13.0,
        15.0,
        14.0,
        16.0,
        18.0,
        17.0,
        19.0,
        21.0,
        20.0,
        22.0,
        24.0,
        23.0,
        25.0,
        27.0,
        26.0,
    ]
    n = 14
    tr = _python_true_range(high, low, close)
    expected = _python_wilder(tr, n)
    got = atr_wilder(pd.Series(high), pd.Series(low), pd.Series(close), n=n)
    assert got.iloc[: n - 1].isna().all()
    for i, value in enumerate(expected):
        if value is None:
            assert pd.isna(got.iloc[i])
        else:
            assert got.iloc[i] == pytest.approx(value, rel=REL)
    # adjust=True is a different (non-recursive) series; the two must not match.
    tr_s = pd.Series(tr)
    adjusted = tr_s.ewm(alpha=1.0 / n, adjust=True, min_periods=n).mean()
    assert not np.allclose(got.dropna().to_numpy(), adjusted.dropna().to_numpy())


def test_ema_golden() -> None:
    # span=3 => alpha = 2/(3+1) = 0.5. Closes 10, 12, ..., 28.
    # y0 = 10
    # y1 = 0.5*12 + 0.5*10 = 11
    # y2 = 0.5*14 + 0.5*11 = 12.5
    # y3 = 0.5*16 + 0.5*12.5 = 14.25
    # y4 = 0.5*18 + 0.5*14.25 = 16.125
    close = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0])
    got = ema(close, 3)
    expected = [
        None,
        None,
        12.5,
        14.25,
        16.125,
        18.0625,
        20.03125,
        22.015625,
        24.0078125,
        26.00390625,
    ]
    assert got.iloc[:2].isna().all()
    for i, value in enumerate(expected):
        if value is None:
            assert pd.isna(got.iloc[i])
        else:
            assert got.iloc[i] == pytest.approx(value, rel=REL)


def test_donchian_excludes_current_bar() -> None:
    n = 20
    high = pd.Series(np.arange(1.0, 41.0))
    close = high.copy()
    dc_high = donchian_high(high, n)
    dc_low = donchian_low(high, n)
    # rolling(n)+shift(1) first becomes valid at index n.
    assert dc_high.iloc[:n].isna().all()
    assert dc_low.iloc[:n].isna().all()
    warm = dc_high.notna()
    assert (close[warm] > dc_high[warm]).all()
    # On a strictly rising series the shifted channel equals the previous high.
    assert np.array_equal(dc_high[warm].to_numpy(), high.shift(1)[warm].to_numpy())


def test_efficiency_ratio_monotone_series() -> None:
    n = 20
    close = pd.Series(np.arange(n + 1, dtype="float64"))
    got = efficiency_ratio(close, n)
    assert got.iloc[:n].isna().all()
    assert got.iloc[n] == pytest.approx(1.0, rel=REL)


def test_efficiency_ratio_sawtooth() -> None:
    n = 20
    # 0,1,0,1,...,0 — 21 points; after 20 steps price is back at the start.
    close = pd.Series(([0.0, 1.0] * (n // 2)) + [0.0])
    got = efficiency_ratio(close, n)
    assert got.iloc[n] == pytest.approx(0.0, rel=REL)


def test_atr_percentile_bounds() -> None:
    min_periods = 5
    window = 10
    values = pd.Series([1.0, 2.0, 3.0, 1.5, 4.0, 2.5, 0.5, 5.0, 3.5, 2.0, 6.0, 1.0])
    got = atr_percentile(values, window, min_periods)
    assert got.iloc[: min_periods - 1].isna().all()
    finite = got.dropna()
    assert (finite >= 0.0).all()
    assert (finite <= 1.0).all()


def test_warmup_produces_nan_not_zero() -> None:
    n_atr = 14
    n_ema = 20
    n_dc = 20
    length = 40
    high = pd.Series(np.linspace(10.0, 20.0, length))
    low = high - 1.0
    close = high - 0.5
    atr = atr_wilder(high, low, close, n=n_atr)
    ema_s = ema(close, n_ema)
    dc = donchian_high(high, n_dc)
    assert atr.iloc[: n_atr - 1].isna().all()
    assert ema_s.iloc[: n_ema - 1].isna().all()
    assert dc.iloc[:n_dc].isna().all()
    assert not (atr.iloc[: n_atr - 1] == 0).any()
    assert not (ema_s.iloc[: n_ema - 1] == 0).any()
    assert not (dc.iloc[:n_dc] == 0).any()
    cfg = FeaturesConfig()
    assert required_warmup_bars(cfg) == 273
    assert cfg.vol_window_bars == 504


def test_slope_normalises_by_sqrt_n() -> None:
    # Close rises by 4 over n=4 bars; atr=2. slope = 4 / (2 * sqrt(4)) = 1.0.
    # Dividing by atr*n instead would give 0.5.
    n = 4
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    atr = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
    got = slope_atr(close, atr, n)
    assert pd.isna(got.iloc[n - 1])
    assert got.iloc[n] == pytest.approx(1.0, rel=REL)


def test_vol_bucket_thresholds() -> None:
    percentile = pd.Series([np.nan, 0.0, 0.329, 0.33, 0.66, 0.669, 0.67, 1.0])
    got = classify_vol_bucket(percentile)
    assert list(got) == [
        VolBucket.UNKNOWN,
        VolBucket.LOW,
        VolBucket.LOW,
        VolBucket.MID,
        VolBucket.MID,
        VolBucket.MID,
        VolBucket.HIGH,
        VolBucket.HIGH,
    ]


def test_beta_identical_series_is_one() -> None:
    close = pd.Series(np.linspace(100.0, 120.0, 40))
    beta = rolling_beta(close, close, window=10)
    finite = beta.dropna()
    assert finite.notna().any()
    assert finite.to_numpy() == pytest.approx(1.0, rel=1e-9, abs=1e-9)


def _session_index(n: int) -> pd.DatetimeIndex:
    start = datetime(2020, 1, 2, 21, 0, tzinfo=UTC)
    return pd.DatetimeIndex([start + timedelta(days=i) for i in range(n)], tz="UTC")


def _panel(n_ts: int, symbols: list[str]) -> pd.DataFrame:
    timestamps = _session_index(n_ts)
    rows: list[dict[str, object]] = []
    for i, ts in enumerate(timestamps):
        for j, symbol in enumerate(symbols):
            rows.append(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "mom_252_skip21": 0.01 * i + 0.1 * j,
                    "vol_60": 0.2 + 0.01 * j + 0.0001 * i,
                }
            )
    return pd.DataFrame(rows)


def _snapshots(
    ts: datetime,
    symbols: list[str],
    *,
    ineligible: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": [ts] * len(symbols),
            "symbol": symbols,
            "eligible": [symbol not in ineligible for symbol in symbols],
        }
    )


def test_cross_sectional_truncation() -> None:
    symbols = ["A", "B", "C", "D"]
    n = 150
    drop = 100
    panel = _panel(n, symbols)
    snaps = _snapshots(panel["ts"].iloc[0], symbols)
    full = cross_sectional_ranks(panel, snaps)
    cutoff = panel["ts"].drop_duplicates().iloc[n - drop - 1]
    truncated = cross_sectional_ranks(panel.loc[panel["ts"] <= cutoff].copy(), snaps)
    overlap = full.loc[full["ts"].isin(truncated["ts"])].sort_values(["ts", "symbol"])
    trunc_sorted = truncated.sort_values(["ts", "symbol"])
    for col in ("mom_252_xs_pct", "vol_xs_pct", "xs_population"):
        assert np.array_equal(
            overlap[col].to_numpy(),
            trunc_sorted[col].to_numpy(),
            equal_nan=True,
        )


def test_cross_sectional_excludes_ineligible() -> None:
    ts = datetime(2020, 6, 1, 20, 0, tzinfo=UTC)
    features = pd.DataFrame(
        {
            "ts": [ts] * 4,
            "symbol": ["A", "B", "C", "D"],
            "mom_252_skip21": [0.1, 0.2, 0.3, 0.9],
            "vol_60": [0.2, 0.3, 0.4, 0.9],
        }
    )
    snaps = _snapshots(ts, ["A", "B", "C", "D"], ineligible=frozenset({"D"}))
    ranked = cross_sectional_ranks(features, snaps)
    assert pd.isna(ranked.loc[ranked["symbol"] == "D", "mom_252_xs_pct"]).all()
    eligible_ranks = ranked.loc[ranked["symbol"] != "D", "mom_252_xs_pct"].to_numpy()
    assert eligible_ranks == pytest.approx(np.array([1.0 / 3.0, 2.0 / 3.0, 1.0]), rel=REL)
    assert (ranked["xs_population"] == 3).all()


def test_eligible_mask_rounds_backward() -> None:
    timestamps = _session_index(5)
    symbols = ["A", "B"]
    features = _panel(5, symbols)
    first = pd.DataFrame(
        {
            "ts": [timestamps[0], timestamps[0]],
            "symbol": symbols,
            "eligible": [True, True],
        }
    )
    later = pd.DataFrame(
        {
            "ts": [timestamps[3], timestamps[3]],
            "symbol": symbols,
            "eligible": [True, False],
        }
    )
    snaps = pd.concat([first, later], ignore_index=True)
    mask = eligible_mask(features, snaps)
    by_ts = features.assign(eligible=mask)
    before = by_ts.loc[by_ts["ts"] < timestamps[3]]
    assert bool(before["eligible"].all())
    at = by_ts.loc[by_ts["ts"] >= timestamps[3]]
    assert bool(at.loc[at["symbol"] == "A", "eligible"].all())
    assert not bool(at.loc[at["symbol"] == "B", "eligible"].any())


def _engine_timestamps(n: int) -> pd.DatetimeIndex:
    start = datetime(2015, 1, 5, 21, 0, tzinfo=UTC)
    return pd.DatetimeIndex([start + timedelta(days=i) for i in range(n)], tz="UTC")


def _trending_close(n: int, *, start: float, step: float) -> np.ndarray:
    return start + step * np.arange(n, dtype="float64")


def _market_panel_from_closes(
    timestamps: pd.DatetimeIndex,
    closes: dict[str, np.ndarray],
    *,
    symbol_order: list[str] | None = None,
    session_index: np.ndarray | None = None,
) -> MarketPanel:
    symbols = symbol_order if symbol_order is not None else list(closes)
    n = len(timestamps)
    if session_index is None:
        session_index = np.arange(n, dtype=np.int32)
    rows: list[dict[str, object]] = []
    for i, ts in enumerate(timestamps):
        for symbol in symbols:
            close = float(closes[symbol][i])
            rows.append(
                {
                    "asset_id": symbol,
                    "symbol": symbol,
                    "ts": ts,
                    "session_index": int(session_index[i]),
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "close_raw": close,
                    "volume": 1_000.0,
                    "dollar_volume": close * 1_000.0,
                    "is_suspect": False,
                }
            )
    return MarketPanel(pd.DataFrame(rows, columns=list(MARKET_COLUMNS)))


def _benchmark_panel(timestamps: pd.DatetimeIndex, close: np.ndarray) -> BenchmarkPanel:
    rows: list[dict[str, object]] = []
    for i, ts in enumerate(timestamps):
        px = float(close[i])
        rows.append(
            {
                "ts": ts,
                "session_index": i,
                "open": px,
                "high": px + 1.0,
                "low": px - 1.0,
                "close": px,
                "vix_close": 18.0,
                "vix9d_close": 17.0,
                "vix3m_close": 19.0,
            }
        )
    return BenchmarkPanel(pd.DataFrame(rows, columns=list(BENCHMARK_COLUMNS)))


def _all_eligible_snapshots(timestamps: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": [timestamps[0]] * len(symbols),
            "symbol": symbols,
            "eligible": [True] * len(symbols),
        }
    )


def _engine_bundle(
    n: int,
    symbols: list[str],
    *,
    symbol_order: list[str] | None = None,
    drop_last: int = 0,
) -> tuple[MarketPanel, BenchmarkPanel, pd.DataFrame]:
    timestamps = _engine_timestamps(n)
    closes = {
        symbol: _trending_close(n, start=50.0 + 10.0 * i, step=0.05 + 0.01 * i)
        for i, symbol in enumerate(symbols)
    }
    spy_close = (
        closes["SPY"] if "SPY" in closes else _trending_close(n, start=200.0, step=0.02)
    )
    used_ts = timestamps[: n - drop_last] if drop_last else timestamps
    used_closes = {sym: arr[: len(used_ts)] for sym, arr in closes.items()}
    panel = _market_panel_from_closes(used_ts, used_closes, symbol_order=symbol_order)
    bench = _benchmark_panel(used_ts, spy_close[: len(used_ts)])
    snaps = _all_eligible_snapshots(used_ts, symbols)
    return panel, bench, snaps


def test_feature_columns_match_dataclass() -> None:
    panel, bench, snaps = _engine_bundle(40, ["AAA", "BBB"])
    out = compute_features(panel, bench, snaps, FeaturesConfig())
    assert set(out.frame.columns) == {f.name for f in fields(FeatureRow)}
    assert tuple(out.frame.columns) == FEATURE_COLUMNS


def test_market_regime_broadcast_no_lost_rows() -> None:
    warmup = required_warmup_bars(FeaturesConfig())
    panel, bench, snaps = _engine_bundle(warmup + 10, ["AAA", "BBB"])
    out = compute_features(panel, bench, snaps, FeaturesConfig())
    assert len(out.frame) == len(panel.frame)
    warm = out.frame["is_warm"]
    assert bool(warm.any())
    assert out.frame.loc[warm, "market_regime"].notna().all()


def test_market_regime_mismatch_raises() -> None:
    panel, bench, snaps = _engine_bundle(30, ["AAA"])
    truncated = BenchmarkPanel(bench.frame.iloc[:-1].copy())
    with pytest.raises(ScoutLookaheadError, match="lost rows"):
        compute_features(panel, truncated, snaps, FeaturesConfig())


def test_beta_spy_self_is_one() -> None:
    panel, bench, snaps = _engine_bundle(80, ["SPY", "AAA"])
    out = compute_features(panel, bench, snaps, FeaturesConfig())
    spy = out.frame.loc[out.frame["symbol"].astype(str) == "SPY"]
    assert (spy["beta_bench_90"] == 1.0).all()
    assert (spy["corr_bench_90"] == 1.0).all()


def test_symbol_order_invariance() -> None:
    symbols_ab = ["AAA", "BBB"]
    symbols_ba = ["BBB", "AAA"]
    n = 60
    panel_ab, bench_ab, snaps_ab = _engine_bundle(n, symbols_ab, symbol_order=symbols_ab)
    panel_ba, bench_ba, snaps_ba = _engine_bundle(n, symbols_ab, symbol_order=symbols_ba)
    cfg = FeaturesConfig()
    a = compute_features(panel_ab, bench_ab, snaps_ab, cfg).frame
    b = compute_features(panel_ba, bench_ba, snaps_ba, cfg).frame
    a = a.sort_values(["ts", "symbol"], kind="mergesort").reset_index(drop=True)
    b = b.sort_values(["ts", "symbol"], kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_categorical=False)


def test_engine_warmup_is_warm_false() -> None:
    cfg = FeaturesConfig()
    warmup = required_warmup_bars(cfg)
    panel, bench, snaps = _engine_bundle(warmup + 5, ["AAA"])
    out = compute_features(panel, bench, snaps, cfg).frame
    cold = out["bars_available"] < warmup
    assert not out.loc[cold, "is_warm"].any()
    assert out.loc[~cold, "is_warm"].all()
    assert out.loc[out["bars_available"] < cfg.mom_lookback_bars, "mom_252_skip21"].isna().all()
    cold_atr = out.loc[cold, "atr_14"]
    assert not bool((cold_atr == 0).any())


def test_emit_from_matches_full_panel_after_cut() -> None:
    n = 80
    panel, bench, snaps = _engine_bundle(n, ["AAA", "BBB"])
    cfg = FeaturesConfig()
    full = compute_features(panel, bench, snaps, cfg).frame
    cut_ts = panel.timestamps[n // 2]
    emitted = compute_features(
        panel, bench, snaps, cfg, emit_from=cut_ts.to_pydatetime()
    ).frame
    assert emitted["ts"].min() >= pd.Timestamp(cut_ts)
    expected = (
        full.loc[full["ts"] >= cut_ts]
        .sort_values(["ts", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    got = emitted.sort_values(["ts", "symbol"], kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, got, check_categorical=False)


def test_emit_from_rejects_naive_datetime() -> None:
    panel, bench, snaps = _engine_bundle(20, ["AAA"])
    naive = datetime(2015, 1, 2)  # noqa: DTZ001  the point of this test
    with pytest.raises(ValueError, match="timezone-aware"):
        compute_features(panel, bench, snaps, FeaturesConfig(), emit_from=naive)


def test_engine_cross_sectional_truncation() -> None:
    n = 400
    drop = 100
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    panel_full, bench_full, snaps = _engine_bundle(n, symbols)
    panel_cut, bench_cut, snaps_cut = _engine_bundle(n, symbols, drop_last=drop)
    cfg = FeaturesConfig()
    full = compute_features(panel_full, bench_full, snaps, cfg).frame
    truncated = compute_features(panel_cut, bench_cut, snaps_cut, cfg).frame
    overlap_ts = truncated["ts"].unique()
    overlap = full.loc[full["ts"].isin(overlap_ts)].sort_values(["ts", "symbol"])
    trunc_sorted = truncated.sort_values(["ts", "symbol"])
    for col in ("mom_252_xs_pct", "vol_xs_pct", "xs_population"):
        assert np.array_equal(
            overlap[col].to_numpy(),
            trunc_sorted[col].to_numpy(),
            equal_nan=True,
        )


def test_bars_since_gap_resets_after_calendar_hole() -> None:
    n = 8
    timestamps = _engine_timestamps(n)
    session_index = np.array([0, 1, 2, 5, 6, 7, 8, 9], dtype=np.int32)
    closes = {"AAA": _trending_close(n, start=100.0, step=0.1)}
    panel = _market_panel_from_closes(timestamps, closes, session_index=session_index)
    bench = _benchmark_panel(timestamps, _trending_close(n, start=200.0, step=0.02))
    snaps = _all_eligible_snapshots(timestamps, ["AAA"])
    out = compute_features(panel, bench, snaps, FeaturesConfig()).frame
    got = out.sort_values("ts")["bars_since_gap"].to_numpy()
    assert list(got) == [0, 1, 2, 0, 1, 2, 3, 4]
