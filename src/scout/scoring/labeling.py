"""Triple-barrier labeling. Rules in 07-EDGE_AND_SCORING.md §2."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

from scout.config.schema import LabelingConfig, TieRule
from scout.domain.enums import SetupOutcome, VolBucket
from scout.domain.features import FEATURE_COLUMNS, FeaturePanel, FeatureRow, feature_row_from_tuple
from scout.domain.market import EarningsEvent, MarketPanel
from scout.domain.ports import Strategy
from scout.domain.setup import ResolvedSetup, Setup
from scout.gates.eligibility import EARNINGS_UNCERTAINTY_SESSIONS, SessionLookup

# 07-EDGE_AND_SCORING.md §3. Column order is the schema.
LABEL_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "symbol",
    "setup_ts",
    "entry_ts",
    "resolution_ts",
    "strategy_id",
    "direction",
    "regime",
    "market_regime",
    "vol_bucket",
    "reference_price",
    "entry_price",
    "stop_price",
    "exit_price",
    "target_price",
    "risk_per_unit",
    "reward_risk_ratio",
    "entry_gap_atr",
    "outcome",
    "bars_held",
    "realised_r_gross",
    "mae_r",
    "mfe_r",
    "atr_pct",
    "efficiency_ratio_20",
    "beta_bench_90",
    "adv_usd_60",
    "mom_252_xs_pct",
    "overnight_var_share_60",
    "xs_population",
    "had_earnings_in_window",
)

_FLOAT_LABEL_COLS = frozenset(
    {
        "reference_price",
        "entry_price",
        "stop_price",
        "exit_price",
        "target_price",
        "risk_per_unit",
        "reward_risk_ratio",
        "entry_gap_atr",
        "realised_r_gross",
        "mae_r",
        "mfe_r",
        "atr_pct",
        "efficiency_ratio_20",
        "beta_bench_90",
        "adv_usd_60",
        "mom_252_xs_pct",
        "overnight_var_share_60",
    }
)


def resolve_setup(
    setup: Setup,
    forward_bars: pd.DataFrame,
    cfg: LabelingConfig,
    *,
    vol_bucket: VolBucket = VolBucket.UNKNOWN,
) -> ResolvedSetup:
    """Triple-barrier resolution. `forward_bars` are this symbol's sessions
    with ts > setup.ts, ascending. Fewer than max_hold_bars rows means OPEN
    if no barrier is hit first.
    """
    forward = _normalize_forward(forward_bars)
    if len(forward) == 0:
        return _open_unentered(setup, vol_bucket)

    entry = float(forward.iloc[0]["open"])
    entry_ts = _as_utc(forward.iloc[0]["ts"])
    if not math.isfinite(entry):
        return _open_unentered(setup, vol_bucket)

    sign = setup.direction.sign
    risk_per_unit = setup.risk_per_unit
    # Distances from the decision close, applied to the actual fill. Equities
    # gap every night; keeping the original levels would rewrite the labeled R/R.
    stop = entry - sign * risk_per_unit
    if setup.target_price is None:
        target: float | None = None
    else:
        target = entry + sign * setup.reward_per_unit

    mae_r = 0.0
    mfe_r = 0.0
    for i in range(len(forward)):
        bar = forward.iloc[i]
        mae_r, mfe_r = _update_excursions(mae_r, mfe_r, bar, entry, sign, risk_per_unit)

        open_px = float(bar["open"])
        ts = _as_utc(bar["ts"])
        gapped_through = _gapped_through_stop(open_px, stop, sign)
        if gapped_through:
            return _resolved(
                setup,
                entry_ts=entry_ts,
                entry_price=entry,
                resolution_ts=ts,
                exit_price=open_px,
                outcome=SetupOutcome.STOP,
                bars_held=i + 1,
                mae_r=mae_r,
                mfe_r=mfe_r,
                vol_bucket=vol_bucket,
            )

        hit_stop = _hit_stop(bar, stop, sign)
        hit_target = target is not None and _hit_target(bar, target, sign)
        # OHLC cannot order the low and the high. The default is stop; `target`
        # is a robustness-only override and is never the production rule.
        if hit_stop and hit_target and cfg.tie_rule is TieRule.TARGET:
            hit_stop = False
        if hit_stop:
            return _resolved(
                setup,
                entry_ts=entry_ts,
                entry_price=entry,
                resolution_ts=ts,
                exit_price=stop,
                outcome=SetupOutcome.STOP,
                bars_held=i + 1,
                mae_r=mae_r,
                mfe_r=mfe_r,
                vol_bucket=vol_bucket,
            )
        if hit_target:
            assert target is not None
            # Symmetric with the stop gap: a session opening through the target
            # fills at the open, which is better than the target. Do not clip.
            gap_fill = _gapped_through_target(open_px, target, sign)
            exit_px = open_px if gap_fill else target
            return _resolved(
                setup,
                entry_ts=entry_ts,
                entry_price=entry,
                resolution_ts=ts,
                exit_price=exit_px,
                outcome=SetupOutcome.TARGET,
                bars_held=i + 1,
                mae_r=mae_r,
                mfe_r=mfe_r,
                vol_bucket=vol_bucket,
            )
        if i + 1 >= setup.max_hold_bars:
            close_px = float(bar["close"])
            return _resolved(
                setup,
                entry_ts=entry_ts,
                entry_price=entry,
                resolution_ts=ts,
                exit_price=close_px,
                outcome=SetupOutcome.TIME,
                bars_held=i + 1,
                mae_r=mae_r,
                mfe_r=mfe_r,
                vol_bucket=vol_bucket,
            )

    return ResolvedSetup(
        setup=setup,
        entry_ts=entry_ts,
        entry_price=entry,
        resolution_ts=None,
        exit_price=None,
        outcome=SetupOutcome.OPEN,
        bars_held=len(forward),
        realised_r_gross=float("nan"),
        mae_r=mae_r,
        mfe_r=mfe_r,
        vol_bucket=vol_bucket,
    )


def regime_allows(strategy: Strategy, row: FeatureRow) -> bool:
    """Per-symbol AND market regime, matching 01-ARCHITECTURE.md §5."""
    allowed_market = getattr(strategy, "allowed_market_regimes", None)
    if allowed_market is not None and row.market_regime not in allowed_market:
        return False
    allowed = strategy.allowed_regimes
    if not allowed:
        return True
    return row.regime in allowed


@dataclass(frozen=True, slots=True)
class AssetTick:
    """One asset finished. Used for progress.json / stdout."""

    asset_id: str
    index: int
    total: int
    warm_bars: int
    setups_asset: int
    setups_total: int
    skipped: bool


@dataclass(frozen=True, slots=True)
class _OhlcPack:
    ts_ns: NDArray[np.int64]
    ts_py: NDArray[np.object_]
    open: NDArray[np.float64]
    high: NDArray[np.float64]
    low: NDArray[np.float64]
    close: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _PreparedEvent:
    session_idx: int | None
    available_ts: datetime | None


def earnings_in_window_fast(
    *,
    ts: datetime,
    max_hold_bars: int,
    events: Sequence[_PreparedEvent],
    is_etf: bool,
    lookup: SessionLookup,
) -> bool:
    """Same conservative rules as `earnings_in_window`, with a cached calendar."""
    if is_etf:
        return False
    t_idx = lookup.index_at_ts(ts)
    if t_idx is None:
        return True
    known = tuple(
        event for event in events if event.available_ts is None or event.available_ts <= ts
    )
    if not known:
        return True
    hold_lo = t_idx + 1
    hold_hi = t_idx + max_hold_bars
    for event in known:
        e_idx = event.session_idx
        if e_idx is None:
            return True
        if event.available_ts is None:
            lo = e_idx - EARNINGS_UNCERTAINTY_SESSIONS
            hi = e_idx + EARNINGS_UNCERTAINTY_SESSIONS
            if _closed_intersect(lo, hi, hold_lo, hold_hi):
                return True
        elif _closed_intersect(e_idx, e_idx, hold_lo, hold_hi):
            return True
    return False


def prepare_earnings(
    events: Sequence[EarningsEvent],
    lookup: SessionLookup,
) -> tuple[_PreparedEvent, ...]:
    return tuple(
        _PreparedEvent(
            session_idx=lookup.index_on_or_after(event.earnings_date),
            available_ts=event.available_ts,
        )
        for event in events
    )


def resolve_setup_arrays(
    setup: Setup,
    *,
    ts_py: NDArray[np.object_],
    open_: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    cfg: LabelingConfig,
    vol_bucket: VolBucket,
) -> ResolvedSetup:
    """Triple-barrier on numpy columns. Same branches as `resolve_setup`."""
    n = int(open_.shape[0])
    if n == 0:
        return _open_unentered(setup, vol_bucket)
    entry = float(open_[0])
    entry_ts = _as_utc(ts_py[0])
    if not math.isfinite(entry):
        return _open_unentered(setup, vol_bucket)

    sign = setup.direction.sign
    risk_per_unit = setup.risk_per_unit
    stop = entry - sign * risk_per_unit
    if setup.target_price is None:
        target: float | None = None
    else:
        target = entry + sign * setup.reward_per_unit

    mae_r = 0.0
    mfe_r = 0.0
    for i in range(n):
        high_px = float(high[i])
        low_px = float(low[i])
        high_r = sign * (high_px - entry) / risk_per_unit
        low_r = sign * (low_px - entry) / risk_per_unit
        mae_r = min(mae_r, min(high_r, low_r))
        mfe_r = max(mfe_r, max(high_r, low_r))

        open_px = float(open_[i])
        ts = _as_utc(ts_py[i])
        if _gapped_through_stop(open_px, stop, sign):
            return _resolved(
                setup,
                entry_ts=entry_ts,
                entry_price=entry,
                resolution_ts=ts,
                exit_price=open_px,
                outcome=SetupOutcome.STOP,
                bars_held=i + 1,
                mae_r=mae_r,
                mfe_r=mfe_r,
                vol_bucket=vol_bucket,
            )

        hit_stop = (low_px <= stop) if sign > 0 else (high_px >= stop)
        hit_target = False
        if target is not None:
            hit_target = (high_px >= target) if sign > 0 else (low_px <= target)
        if hit_stop and hit_target and cfg.tie_rule is TieRule.TARGET:
            hit_stop = False
        if hit_stop:
            return _resolved(
                setup,
                entry_ts=entry_ts,
                entry_price=entry,
                resolution_ts=ts,
                exit_price=stop,
                outcome=SetupOutcome.STOP,
                bars_held=i + 1,
                mae_r=mae_r,
                mfe_r=mfe_r,
                vol_bucket=vol_bucket,
            )
        if hit_target:
            assert target is not None
            gap_fill = _gapped_through_target(open_px, target, sign)
            exit_px = open_px if gap_fill else target
            return _resolved(
                setup,
                entry_ts=entry_ts,
                entry_price=entry,
                resolution_ts=ts,
                exit_price=exit_px,
                outcome=SetupOutcome.TARGET,
                bars_held=i + 1,
                mae_r=mae_r,
                mfe_r=mfe_r,
                vol_bucket=vol_bucket,
            )
        if i + 1 >= setup.max_hold_bars:
            close_px = float(close[i])
            return _resolved(
                setup,
                entry_ts=entry_ts,
                entry_price=entry,
                resolution_ts=ts,
                exit_price=close_px,
                outcome=SetupOutcome.TIME,
                bars_held=i + 1,
                mae_r=mae_r,
                mfe_r=mfe_r,
                vol_bucket=vol_bucket,
            )

    return ResolvedSetup(
        setup=setup,
        entry_ts=entry_ts,
        entry_price=entry,
        resolution_ts=None,
        exit_price=None,
        outcome=SetupOutcome.OPEN,
        bars_held=n,
        realised_r_gross=float("nan"),
        mae_r=mae_r,
        mfe_r=mfe_r,
        vol_bucket=vol_bucket,
    )


def label_setups(
    feature_panel: FeaturePanel,
    market_panel: MarketPanel,
    strategies: Sequence[Strategy],
    cfg: LabelingConfig,
    *,
    snapshots: pd.DataFrame,
    earnings_by_asset: Mapping[str, Sequence[EarningsEvent]],
    is_etf: Mapping[str, bool],
    calendar: pd.DataFrame,
    skip_asset_ids: Set[str] = frozenset(),
    flush_every: int = 25,
    on_asset: Callable[[AssetTick], None] | None = None,
    on_batch: Callable[[pd.DataFrame, tuple[str, ...]], None] | None = None,
) -> pd.DataFrame:
    """Detect on every warm bar, resolve each setup, return the §3 table.

    Forward bars are sliced with searchsorted to `max_hold_bars`. Batches are
    converted to a DataFrame immediately so the dict list cannot grow without
    bound.
    """
    if not strategies:
        return empty_label_frame()
    feat = feature_panel.frame
    if feat.empty:
        return empty_label_frame()
    mkt = market_panel.frame
    lookup = SessionLookup(calendar)
    earnings_prep: dict[str, tuple[_PreparedEvent, ...]] = {
        aid: prepare_earnings(events, lookup) for aid, events in earnings_by_asset.items()
    }
    feat_by_symbol: dict[str, pd.DataFrame] = {}
    for raw_sym, grp in feat.groupby("symbol", sort=False, observed=True):
        feat_by_symbol[str(raw_sym)] = grp

    grouped = mkt.groupby("asset_id", sort=False, observed=True)
    total = int(mkt["asset_id"].astype(str).nunique(dropna=True))
    pending: list[dict[str, object]] = []
    pending_ids: list[str] = []
    since_flush = 0
    setups_total = 0
    parts: list[pd.DataFrame] = []

    def _flush() -> None:
        nonlocal pending, pending_ids, since_flush
        if not pending_ids:
            return
        frame = records_to_frame(pending)
        batch_ids = tuple(pending_ids)
        if on_batch is not None:
            on_batch(frame, batch_ids)
        else:
            parts.append(frame)
        pending = []
        pending_ids = []
        since_flush = 0

    try:
        for index, (raw_id, mkt_grp) in enumerate(grouped):
            asset_id = str(raw_id)
            if asset_id in skip_asset_ids:
                if on_asset is not None:
                    on_asset(
                        AssetTick(
                            asset_id=asset_id,
                            index=index,
                            total=total,
                            warm_bars=0,
                            setups_asset=0,
                            setups_total=setups_total,
                            skipped=True,
                        )
                    )
                continue
            n_setups, n_warm = _label_one_asset(
                asset_id=asset_id,
                mkt_grp=mkt_grp,
                feat_by_symbol=feat_by_symbol,
                strategies=strategies,
                cfg=cfg,
                etf=bool(is_etf.get(asset_id, False)),
                events=earnings_prep.get(asset_id, ()),
                lookup=lookup,
                records=pending,
            )
            setups_total += n_setups
            pending_ids.append(asset_id)
            since_flush += 1
            if on_asset is not None:
                on_asset(
                    AssetTick(
                        asset_id=asset_id,
                        index=index,
                        total=total,
                        warm_bars=n_warm,
                        setups_asset=n_setups,
                        setups_total=setups_total,
                        skipped=False,
                    )
                )
            if since_flush >= flush_every:
                _flush()
        _flush()
    except KeyboardInterrupt:
        _flush()
        raise

    if not parts:
        return empty_label_frame()
    combined = pd.concat(parts, ignore_index=True)
    return _join_adv(combined, snapshots)


def _label_one_asset(
    *,
    asset_id: str,
    mkt_grp: pd.DataFrame,
    feat_by_symbol: Mapping[str, pd.DataFrame],
    strategies: Sequence[Strategy],
    cfg: LabelingConfig,
    etf: bool,
    events: Sequence[_PreparedEvent],
    lookup: SessionLookup,
    records: list[dict[str, object]],
) -> tuple[int, int]:
    if mkt_grp.empty:
        return 0, 0
    symbols = mkt_grp["symbol"].astype(str).drop_duplicates().tolist()
    feat_parts = [feat_by_symbol[sym] for sym in symbols if sym in feat_by_symbol]
    if not feat_parts:
        return 0, 0
    feat_local = feat_parts[0] if len(feat_parts) == 1 else pd.concat(feat_parts, ignore_index=True)
    keys = mkt_grp.loc[:, ["ts", "symbol", "asset_id"]].drop_duplicates(
        subset=["ts", "symbol"], keep="last"
    )
    before = len(feat_local)
    tagged = feat_local.merge(keys, on=["ts", "symbol"], how="inner")
    if len(tagged) > before:
        raise AssertionError("merge duplicated feature rows")
    if tagged.empty:
        return 0, 0
    warm = tagged.loc[tagged["is_warm"].to_numpy()]
    if warm.empty:
        return 0, 0
    ordered = mkt_grp.sort_values("ts", kind="mergesort")
    packed = _pack_ohlc(ordered)
    n_before = len(records)
    feat_view = warm.loc[:, list(FEATURE_COLUMNS)]
    for rec in feat_view.itertuples(index=False, name=None):
        row = feature_row_from_tuple(tuple(rec))
        for strategy in strategies:
            if not regime_allows(strategy, row):
                continue
            setup = strategy.detect(row)
            if setup is None:
                continue
            resolved = _resolve_packed(setup, packed, cfg, vol_bucket=row.vol_bucket)
            had = earnings_in_window_fast(
                ts=setup.ts,
                max_hold_bars=setup.max_hold_bars,
                events=events,
                is_etf=etf,
                lookup=lookup,
            )
            records.append(
                label_record(
                    resolved,
                    asset_id=asset_id,
                    feature_row=row,
                    adv_usd_60=float("nan"),
                    had_earnings_in_window=had,
                )
            )
    return len(records) - n_before, len(warm)


def _pack_ohlc(ordered: pd.DataFrame) -> _OhlcPack:
    ts_index = pd.DatetimeIndex(pd.to_datetime(ordered["ts"], utc=True))
    return _OhlcPack(
        ts_ns=ts_index.asi8.astype(np.int64, copy=False),
        ts_py=ts_index.to_pydatetime(),
        open=ordered["open"].to_numpy(dtype=np.float64, copy=False),
        high=ordered["high"].to_numpy(dtype=np.float64, copy=False),
        low=ordered["low"].to_numpy(dtype=np.float64, copy=False),
        close=ordered["close"].to_numpy(dtype=np.float64, copy=False),
    )


def _resolve_packed(
    setup: Setup,
    packed: _OhlcPack,
    cfg: LabelingConfig,
    *,
    vol_bucket: VolBucket,
) -> ResolvedSetup:
    setup_ts = pd.Timestamp(setup.ts)
    if setup_ts.tzinfo is None:
        setup_ts = setup_ts.tz_localize("UTC")
    else:
        setup_ts = setup_ts.tz_convert("UTC")
    start = int(np.searchsorted(packed.ts_ns, np.int64(setup_ts.value), side="right"))
    end = min(packed.ts_ns.size, start + setup.max_hold_bars)
    if start >= packed.ts_ns.size or end <= start:
        return resolve_setup_arrays(
            setup,
            ts_py=packed.ts_py[0:0],
            open_=packed.open[0:0],
            high=packed.high[0:0],
            low=packed.low[0:0],
            close=packed.close[0:0],
            cfg=cfg,
            vol_bucket=vol_bucket,
        )
    return resolve_setup_arrays(
        setup,
        ts_py=packed.ts_py[start:end],
        open_=packed.open[start:end],
        high=packed.high[start:end],
        low=packed.low[start:end],
        close=packed.close[start:end],
        cfg=cfg,
        vol_bucket=vol_bucket,
    )


def _closed_intersect(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 <= a1 and b0 <= b1 and a0 <= b1 and b0 <= a1


def label_record(
    resolved: ResolvedSetup,
    *,
    asset_id: str,
    feature_row: FeatureRow,
    adv_usd_60: float,
    had_earnings_in_window: bool,
) -> dict[str, object]:
    """One row of the §3 table. `stop_price` / `target_price` are re-anchored."""
    setup = resolved.setup
    entry = resolved.entry_price
    stop, target = _reanchored_levels(setup, entry)
    atr = feature_row.atr_14
    if math.isfinite(entry) and math.isfinite(setup.reference_price) and atr > 0:
        entry_gap_atr = setup.direction.sign * (entry - setup.reference_price) / atr
    else:
        entry_gap_atr = float("nan")
    reward_rr = None if setup.target_price is None else setup.reward_risk_ratio
    return {
        "asset_id": asset_id,
        "symbol": setup.symbol,
        "setup_ts": setup.ts,
        "entry_ts": resolved.entry_ts,
        "resolution_ts": resolved.resolution_ts,
        "strategy_id": setup.strategy_id,
        "direction": setup.direction.value,
        "regime": setup.regime.value,
        "market_regime": feature_row.market_regime.value,
        "vol_bucket": resolved.vol_bucket.value,
        "reference_price": setup.reference_price,
        "entry_price": entry,
        "stop_price": stop,
        "exit_price": resolved.exit_price,
        "target_price": target,
        "risk_per_unit": setup.risk_per_unit,
        "reward_risk_ratio": reward_rr,
        "entry_gap_atr": entry_gap_atr,
        "outcome": resolved.outcome.value,
        "bars_held": resolved.bars_held,
        "realised_r_gross": resolved.realised_r_gross,
        "mae_r": resolved.mae_r,
        "mfe_r": resolved.mfe_r,
        "atr_pct": feature_row.atr_pct,
        "efficiency_ratio_20": feature_row.efficiency_ratio_20,
        "beta_bench_90": feature_row.beta_bench_90,
        "adv_usd_60": adv_usd_60,
        "mom_252_xs_pct": feature_row.mom_252_xs_pct,
        "overnight_var_share_60": feature_row.overnight_var_share_60,
        "xs_population": feature_row.xs_population,
        "had_earnings_in_window": had_earnings_in_window,
    }


def records_to_frame(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    if not records:
        return empty_label_frame()
    frame = pd.DataFrame(list(records), columns=list(LABEL_COLUMNS))
    return _coerce_label_frame(frame)


def empty_label_frame() -> pd.DataFrame:
    data: dict[str, pd.Series] = {
        "asset_id": pd.Series(dtype="object"),
        "symbol": pd.Series(dtype="object"),
        "setup_ts": pd.Series(dtype="datetime64[ns, UTC]"),
        "entry_ts": pd.Series(dtype="datetime64[ns, UTC]"),
        "resolution_ts": pd.Series(dtype="datetime64[ns, UTC]"),
        "strategy_id": pd.Series(dtype="object"),
        "direction": pd.Series(dtype="object"),
        "regime": pd.Series(dtype="object"),
        "market_regime": pd.Series(dtype="object"),
        "vol_bucket": pd.Series(dtype="object"),
        "reference_price": pd.Series(dtype="float64"),
        "entry_price": pd.Series(dtype="float64"),
        "stop_price": pd.Series(dtype="float64"),
        "exit_price": pd.Series(dtype="float64"),
        "target_price": pd.Series(dtype="float64"),
        "risk_per_unit": pd.Series(dtype="float64"),
        "reward_risk_ratio": pd.Series(dtype="float64"),
        "entry_gap_atr": pd.Series(dtype="float64"),
        "outcome": pd.Series(dtype="object"),
        "bars_held": pd.Series(dtype="int32"),
        "realised_r_gross": pd.Series(dtype="float64"),
        "mae_r": pd.Series(dtype="float64"),
        "mfe_r": pd.Series(dtype="float64"),
        "atr_pct": pd.Series(dtype="float64"),
        "efficiency_ratio_20": pd.Series(dtype="float64"),
        "beta_bench_90": pd.Series(dtype="float64"),
        "adv_usd_60": pd.Series(dtype="float64"),
        "mom_252_xs_pct": pd.Series(dtype="float64"),
        "overnight_var_share_60": pd.Series(dtype="float64"),
        "xs_population": pd.Series(dtype="int32"),
        "had_earnings_in_window": pd.Series(dtype="bool"),
    }
    return pd.DataFrame(data)


def _join_adv(frame: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    needed = {"ts", "asset_id", "adv_usd_60"}
    if snapshots.empty or not needed.issubset(set(snapshots.columns)):
        return frame
    left = frame.reset_index(drop=True)
    left = left.assign(_ord=range(len(left)))
    asof_left = left.loc[:, ["_ord", "asset_id", "setup_ts"]].copy()
    asof_left["asset_id"] = asof_left["asset_id"].astype(str)
    asof_left["setup_ts"] = pd.to_datetime(asof_left["setup_ts"], utc=True)
    # pandas 2.2 merge_asof checks that `on` is globally monotonic even when
    # `by` is set. Sort by setup_ts first, not asset_id — overlapping
    # timestamps across names otherwise raise "left keys must be sorted".
    asof_left = asof_left.sort_values(["setup_ts", "asset_id"], kind="mergesort")
    right = snapshots.loc[:, ["asset_id", "ts", "adv_usd_60"]].copy()
    right["asset_id"] = right["asset_id"].astype(str)
    right["ts"] = pd.to_datetime(right["ts"], utc=True)
    right = right.rename(columns={"ts": "setup_ts", "adv_usd_60": "adv_from_snap"})
    right = right.sort_values(["setup_ts", "asset_id"], kind="mergesort")
    before = len(asof_left)
    joined = pd.merge_asof(
        asof_left,
        right,
        on="setup_ts",
        by="asset_id",
        direction="backward",
    )
    assert len(joined) == before, "adv merge_asof duplicated rows"
    adv = joined.set_index("_ord")["adv_from_snap"]
    out = left.copy()
    out["adv_usd_60"] = adv.reindex(out["_ord"]).to_numpy()
    out = out.drop(columns=["_ord"])
    return _coerce_label_frame(out)


def _coerce_label_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[:, list(LABEL_COLUMNS)].copy()
    for col in ("setup_ts", "entry_ts", "resolution_ts"):
        out[col] = pd.to_datetime(out[col], utc=True)
    for col in _FLOAT_LABEL_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out["bars_held"] = pd.to_numeric(out["bars_held"], errors="coerce").fillna(0).astype("int32")
    out["xs_population"] = (
        pd.to_numeric(out["xs_population"], errors="coerce").fillna(0).astype("int32")
    )
    out["had_earnings_in_window"] = out["had_earnings_in_window"].astype(bool)
    for col in (
        "asset_id",
        "symbol",
        "strategy_id",
        "direction",
        "regime",
        "market_regime",
        "vol_bucket",
        "outcome",
    ):
        out[col] = out[col].astype("object")
    return out


def _normalize_forward(forward_bars: pd.DataFrame) -> pd.DataFrame:
    if forward_bars.empty:
        return _empty_ohlc()
    out = forward_bars
    if "open" not in out.columns:
        raise ValueError("forward_bars must include open/high/low/close")
    if "ts" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            if "ts" not in out.columns:
                out = out.rename(columns={out.columns[0]: "ts"})
        else:
            raise ValueError("forward_bars must have a ts column or DatetimeIndex")
    needed = ("ts", "open", "high", "low", "close")
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"forward_bars missing columns: {missing}")
    out = out.loc[:, list(needed)].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out.sort_values("ts", kind="mergesort").reset_index(drop=True)


def _empty_ohlc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.Series(dtype="datetime64[ns, UTC]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
        }
    )


def _open_unentered(setup: Setup, vol_bucket: VolBucket) -> ResolvedSetup:
    return ResolvedSetup(
        setup=setup,
        entry_ts=setup.ts,
        entry_price=float("nan"),
        resolution_ts=None,
        exit_price=None,
        outcome=SetupOutcome.OPEN,
        bars_held=0,
        realised_r_gross=float("nan"),
        mae_r=0.0,
        mfe_r=0.0,
        vol_bucket=vol_bucket,
    )


def _resolved(
    setup: Setup,
    *,
    entry_ts: datetime,
    entry_price: float,
    resolution_ts: datetime,
    exit_price: float,
    outcome: SetupOutcome,
    bars_held: int,
    mae_r: float,
    mfe_r: float,
    vol_bucket: VolBucket,
) -> ResolvedSetup:
    return ResolvedSetup(
        setup=setup,
        entry_ts=entry_ts,
        entry_price=entry_price,
        resolution_ts=resolution_ts,
        exit_price=exit_price,
        outcome=outcome,
        bars_held=bars_held,
        realised_r_gross=_realised_r(
            setup.direction.sign, exit_price, entry_price, setup.risk_per_unit
        ),
        mae_r=mae_r,
        mfe_r=mfe_r,
        vol_bucket=vol_bucket,
    )


def _realised_r(sign: int, exit_price: float, entry: float, risk: float) -> float:
    return sign * (exit_price - entry) / risk


def _reanchored_levels(setup: Setup, entry: float) -> tuple[float, float | None]:
    if not math.isfinite(entry):
        return setup.stop_price, setup.target_price
    sign = setup.direction.sign
    stop = entry - sign * setup.risk_per_unit
    if setup.target_price is None:
        return stop, None
    return stop, entry + sign * setup.reward_per_unit


def _update_excursions(
    mae_r: float,
    mfe_r: float,
    bar: pd.Series,
    entry: float,
    sign: int,
    risk: float,
) -> tuple[float, float]:
    high_r = sign * (float(bar["high"]) - entry) / risk
    low_r = sign * (float(bar["low"]) - entry) / risk
    adverse = min(high_r, low_r)
    favorable = max(high_r, low_r)
    return min(mae_r, adverse), max(mfe_r, favorable)


def _gapped_through_stop(open_px: float, stop: float, sign: int) -> bool:
    if not math.isfinite(open_px):
        return False
    return (open_px <= stop) if sign > 0 else (open_px >= stop)


def _gapped_through_target(open_px: float, target: float, sign: int) -> bool:
    if not math.isfinite(open_px):
        return False
    return (open_px >= target) if sign > 0 else (open_px <= target)


def _hit_stop(bar: pd.Series, stop: float, sign: int) -> bool:
    if sign > 0:
        return float(bar["low"]) <= stop
    return float(bar["high"]) >= stop


def _hit_target(bar: pd.Series, target: float, sign: int) -> bool:
    if sign > 0:
        return float(bar["high"]) >= target
    return float(bar["low"]) <= target


def _as_utc(value: object) -> datetime:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    out = ts.to_pydatetime()
    if not isinstance(out, datetime):
        raise TypeError(f"ts is not a datetime: {type(out)!r}")
    return out
