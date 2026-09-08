"""Feature engine: compute every FeatureRow column for a MarketPanel.

Called once before the backtest loop, for the whole history. Not per bar.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from scout.config.schema import FeaturesConfig, MarketRegimeConfig, RegimeConfig
from scout.domain.features import FEATURE_COLUMNS, FeaturePanel
from scout.domain.market import BenchmarkPanel, MarketPanel
from scout.features.cross_sectional import cross_sectional_ranks
from scout.features.indicators import (
    DONCHIAN_N_SHORT,
    MOM_MID_LOOKBACK,
    MOM_SHORT,
    VOL_N_LONG,
    VOL_N_SHORT,
    atr_pct,
    atr_percentile,
    atr_wilder,
    classify_vol_bucket,
    dist_to_high_atr,
    dist_to_low_atr,
    donchian_high,
    donchian_low,
    efficiency_ratio,
    ema,
    ema_spread_atr,
    gap_abs_mean,
    gap_atr,
    keltner_lower,
    keltner_upper,
    momentum,
    momentum_skip,
    overnight_var_share,
    realised_vol,
    required_warmup_bars,
    rolling_beta,
    rolling_corr,
    slope_atr,
)
from scout.features.market import compute_market_columns
from scout.features.regime import classify_regime_vectorized
from scout.utils.errors import ScoutLookaheadError

_INT_COLS = ("xs_population", "bars_available", "bars_since_gap")
_BOOL_COLS = ("spy_above_ma", "is_warm")


def _utc_ns(values: object) -> pd.DatetimeIndex:
    """Normalise timestamps to UTC nanoseconds so merges don't fail on unit."""
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns")


def _aware_utc(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError("emit_from must be timezone-aware UTC")
    return ts.tz_convert("UTC")


def compute_features(
    panel: MarketPanel,
    benchmark: BenchmarkPanel,
    snapshots: pd.DataFrame,
    cfg: FeaturesConfig,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    emit_from: datetime | None = None,
) -> FeaturePanel:
    """Compute all FeatureRow columns for every (ts, asset_id) in `panel`.

    Value at (ts, symbol) depends only on bars with close_time <= ts for that
    symbol, plus SPY/VIX bars with close_time <= ts for market and beta columns.
    Cross-sectional ranks use only symbols eligible at ts.

    Per-symbol indicators still see the full group. `emit_from` drops earlier
    rows after that, so a holdout run does not materialise 1998-2017 features
    it will never trade.
    """
    regime_cfg = RegimeConfig()
    market_cfg = MarketRegimeConfig()
    frame = panel.frame
    if frame.empty:
        return FeaturePanel(_empty_feature_frame())

    emit_ts = _aware_utc(emit_from) if emit_from is not None else None
    spy_close = _benchmark_close(benchmark)
    parts: list[pd.DataFrame] = []
    grouped = frame.groupby("symbol", sort=False, observed=True)
    n_symbols = int(grouped.ngroups)
    for i, (_, group) in enumerate(grouped, start=1):
        part = _symbol_features(group, spy_close, cfg, regime_cfg)
        if emit_ts is not None:
            part = part.loc[part["ts"] >= emit_ts]
        if not part.empty:
            parts.append(part)
        if on_progress is not None and (i == 1 or i == n_symbols or i % 200 == 0):
            on_progress(i, n_symbols)
    if not parts:
        return FeaturePanel(_empty_feature_frame())
    per_symbol = pd.concat(parts, ignore_index=True)

    snaps = _snapshots_for_panel(snapshots, frame["symbol"])
    ranked = cross_sectional_ranks(per_symbol, snaps)
    ranked["regime"] = classify_regime_vectorized(
        ranked["efficiency_ratio_20"],
        ranked["efficiency_ratio_60"],
        ranked["slope_atr_20"],
        ranked["atr_percentile_1y"],
        regime_cfg,
    )

    market = _market_broadcast(benchmark, spy_close, market_cfg)
    before = len(ranked)
    merged = ranked.merge(market, on="ts", how="inner", validate="many_to_one")
    if len(merged) != before:
        raise ScoutLookaheadError("market regime broadcast lost rows")
    warm_missing = merged["is_warm"] & merged["market_regime"].isna()
    if bool(warm_missing.any()):
        raise ScoutLookaheadError("warm row with no market regime")

    missing = set(FEATURE_COLUMNS) - set(merged.columns)
    extra = set(merged.columns) - set(FEATURE_COLUMNS)
    if missing or extra:
        raise ValueError(
            f"feature columns must equal FeatureRow fields; "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    assembled = merged.loc[:, list(FEATURE_COLUMNS)]
    return FeaturePanel(assembled)


def _snapshots_for_panel(snapshots: pd.DataFrame, symbols: pd.Series) -> pd.DataFrame:
    """Copy of snapshot rows for symbols in the panel. Full-file copy is too large."""
    if snapshots.empty:
        return snapshots.copy()
    if "symbol" not in snapshots.columns:
        out = snapshots.copy()
        out["ts"] = _utc_ns(out["ts"])
        return out
    wanted = pd.Index(symbols.astype(str).unique())
    mask = snapshots["symbol"].astype(str).isin(wanted)
    out = snapshots.loc[mask].copy()
    out["ts"] = _utc_ns(out["ts"])
    return out


def _benchmark_close(benchmark: BenchmarkPanel) -> pd.Series:
    close = benchmark.frame.set_index("ts")["close"]
    close.index = _utc_ns(close.index)
    close.index.name = "ts"
    return close


def _market_broadcast(
    benchmark: BenchmarkPanel,
    spy_close: pd.Series,
    market_cfg: MarketRegimeConfig,
) -> pd.DataFrame:
    cols = compute_market_columns(spy_close, market_cfg)
    vix = benchmark.frame.set_index("ts")["vix_close"]
    vix.index = _utc_ns(vix.index)
    out = cols.copy()
    out.index = _utc_ns(out.index)
    out["vix_close"] = vix.reindex(out.index)
    out = out.reset_index()
    if "ts" not in out.columns:
        out = out.rename(columns={out.columns[0]: "ts"})
    out["ts"] = _utc_ns(out["ts"])
    return out.loc[:, ["ts", "market_regime", "spy_dd_252", "spy_above_ma", "vix_close"]]


def _symbol_features(
    group: pd.DataFrame,
    spy_close: pd.Series,
    cfg: FeaturesConfig,
    regime_cfg: RegimeConfig,
) -> pd.DataFrame:
    g = group.sort_values("ts", kind="mergesort")
    idx = _utc_ns(g["ts"])
    close = pd.Series(g["close"].to_numpy(dtype="float64"), index=idx)
    high = pd.Series(g["high"].to_numpy(dtype="float64"), index=idx)
    low = pd.Series(g["low"].to_numpy(dtype="float64"), index=idx)
    open_ = pd.Series(g["open"].to_numpy(dtype="float64"), index=idx)
    session = pd.Series(g["session_index"].to_numpy(dtype="int64"), index=idx)

    atr = atr_wilder(high, low, close, n=cfg.atr_n)
    atr_frac = atr_pct(atr, close)
    ema_f = ema(close, cfg.ema_fast)
    ema_s = ema(close, cfg.ema_slow)
    dc_h_20 = donchian_high(high, DONCHIAN_N_SHORT)
    dc_l_20 = donchian_low(low, DONCHIAN_N_SHORT)
    dc_h_long = donchian_high(high, cfg.donchian_n)
    dc_l_long = donchian_low(low, cfg.donchian_n)
    atr_pctile = atr_percentile(atr_frac, cfg.vol_window_bars, cfg.vol_min_periods)
    gap = gap_atr(open_, close, atr)
    n = len(g)
    bars_available = pd.Series(np.arange(1, n + 1, dtype="int64"), index=idx)
    warmup = required_warmup_bars(cfg)
    beta = rolling_beta(close, spy_close, cfg.beta_window).reindex(idx)
    corr = rolling_corr(close, spy_close, cfg.beta_window).reindex(idx)
    symbol = str(g["symbol"].iloc[0])
    if symbol == cfg.beta_reference_symbol:
        beta = pd.Series(1.0, index=idx)
        corr = pd.Series(1.0, index=idx)

    return pd.DataFrame(
        {
            "symbol": g["symbol"].to_numpy(),
            "ts": idx,
            "close": close.to_numpy(),
            "atr_14": atr.to_numpy(),
            "atr_pct": atr_frac.to_numpy(),
            "vol_20": realised_vol(close, VOL_N_SHORT).to_numpy(),
            "vol_60": realised_vol(close, VOL_N_LONG).to_numpy(),
            "ema_fast": ema_f.to_numpy(),
            "ema_slow": ema_s.to_numpy(),
            "ema_spread_atr": ema_spread_atr(ema_f, ema_s, atr).to_numpy(),
            "slope_atr_20": slope_atr(close, atr, cfg.slope_lookback).to_numpy(),
            "efficiency_ratio_20": efficiency_ratio(close, cfg.er_n).to_numpy(),
            "efficiency_ratio_60": efficiency_ratio(close, cfg.er_long_n).to_numpy(),
            "atr_percentile_1y": atr_pctile.to_numpy(),
            "vol_bucket": classify_vol_bucket(
                atr_pctile,
                low_pct=regime_cfg.vol_low_pct,
                high_pct=regime_cfg.vol_high_pct,
            ).to_numpy(),
            "donchian_high_20": dc_h_20.to_numpy(),
            "donchian_low_20": dc_l_20.to_numpy(),
            "donchian_high_55": dc_h_long.to_numpy(),
            "donchian_low_55": dc_l_long.to_numpy(),
            "keltner_upper": keltner_upper(ema_s, atr, cfg.keltner_k).to_numpy(),
            "keltner_lower": keltner_lower(ema_s, atr, cfg.keltner_k).to_numpy(),
            "dist_to_high_atr": dist_to_high_atr(dc_h_long, close, atr).to_numpy(),
            "dist_to_low_atr": dist_to_low_atr(dc_l_long, close, atr).to_numpy(),
            "mom_252_skip21": momentum_skip(
                close, cfg.mom_lookback_bars, cfg.mom_skip_bars
            ).to_numpy(),
            "mom_126_skip21": momentum_skip(close, MOM_MID_LOOKBACK, cfg.mom_skip_bars).to_numpy(),
            "mom_21": momentum(close, MOM_SHORT).to_numpy(),
            "gap_atr": gap.to_numpy(),
            "gap_abs_mean_20": gap_abs_mean(gap).to_numpy(),
            "overnight_var_share_60": overnight_var_share(open_, close).to_numpy(),
            "beta_bench_90": beta.to_numpy(),
            "corr_bench_90": corr.to_numpy(),
            "bars_available": bars_available.to_numpy(),
            "bars_since_gap": _bars_since_gap(session).to_numpy(),
            "is_warm": (bars_available >= warmup).to_numpy(),
        }
    )


def _bars_since_gap(session_index: pd.Series) -> pd.Series:
    # Calendar holes are session_index jumps > 1. The first bar after a hole
    # is 0 (conservative); each subsequent consecutive session increments.
    delta = session_index.diff()
    is_break = delta > 1
    group_id = is_break.cumsum()
    return session_index.groupby(group_id).cumcount()


def _empty_feature_frame() -> pd.DataFrame:
    object_cols = frozenset({"symbol", "regime", "vol_bucket", "market_regime"})
    data: dict[str, pd.Series] = {}
    for name in FEATURE_COLUMNS:
        if name == "ts":
            data[name] = pd.Series(pd.DatetimeIndex([], tz="UTC"))
        elif name in _BOOL_COLS:
            data[name] = pd.Series(dtype="bool")
        elif name in _INT_COLS:
            data[name] = pd.Series(dtype="int64")
        elif name in object_cols:
            data[name] = pd.Series(dtype="object")
        else:
            data[name] = pd.Series(dtype="float64")
    return pd.DataFrame(data)
