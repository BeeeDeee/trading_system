"""Triple-barrier labeling. Rules in 07-EDGE_AND_SCORING.md §2."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime

import pandas as pd  # type: ignore[import-untyped]

from scout.config.schema import LabelingConfig, TieRule
from scout.domain.enums import MarketRegime, Regime, SetupOutcome, VolBucket
from scout.domain.features import FEATURE_COLUMNS, FeaturePanel, FeatureRow
from scout.domain.market import EarningsEvent, MarketPanel
from scout.domain.ports import Strategy
from scout.domain.setup import ResolvedSetup, Setup
from scout.gates.eligibility import earnings_in_window

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
) -> pd.DataFrame:
    """Detect on every warm bar, resolve each setup, return the §3 table."""
    if not strategies:
        return empty_label_frame()
    feat = feature_panel.frame
    if feat.empty:
        return empty_label_frame()
    mkt = market_panel.frame
    keys = mkt.loc[:, ["ts", "symbol", "asset_id"]].drop_duplicates(
        subset=["ts", "symbol"], keep="last"
    )
    before = len(feat)
    tagged = feat.merge(keys, on=["ts", "symbol"], how="left", validate="many_to_one")
    assert len(tagged) == before, "merge duplicated feature rows"

    forward_by_id: dict[str, pd.DataFrame] = {}
    grouped = mkt.groupby("asset_id", sort=False, observed=True)
    for asset_id, grp in grouped:
        forward_by_id[str(asset_id)] = grp.sort_values("ts", kind="mergesort")

    records: list[dict[str, object]] = []
    for rec in tagged.to_dict("records"):
        if not bool(rec["is_warm"]):
            continue
        raw_id = rec.get("asset_id")
        if raw_id is None or (isinstance(raw_id, float) and not math.isfinite(raw_id)):
            continue
        asset_id = str(raw_id)
        if asset_id == "" or asset_id == "nan":
            continue
        row = _feature_row_from_record({name: rec[name] for name in FEATURE_COLUMNS})
        etf = bool(is_etf.get(asset_id, False))
        events = earnings_by_asset.get(asset_id, ())
        for strategy in strategies:
            if not regime_allows(strategy, row):
                continue
            setup = strategy.detect(row)
            if setup is None:
                continue
            source = forward_by_id.get(asset_id)
            if source is None:
                resolved = resolve_setup(
                    setup, _empty_ohlc(), cfg, vol_bucket=row.vol_bucket
                )
            else:
                setup_ts = pd.Timestamp(setup.ts)
                if setup_ts.tzinfo is None:
                    setup_ts = setup_ts.tz_localize("UTC")
                else:
                    setup_ts = setup_ts.tz_convert("UTC")
                ts_index = pd.DatetimeIndex(pd.to_datetime(source["ts"], utc=True))
                forward = source.loc[ts_index > setup_ts]
                resolved = resolve_setup(
                    setup, forward, cfg, vol_bucket=row.vol_bucket
                )
            had = earnings_in_window(
                ts=setup.ts,
                max_hold_bars=setup.max_hold_bars,
                events=events,
                is_etf=etf,
                calendar=calendar,
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
    frame = records_to_frame(records)
    return _join_adv(frame, snapshots)


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
    asof_left = asof_left.sort_values(["asset_id", "setup_ts"], kind="mergesort")
    right = snapshots.loc[:, ["asset_id", "ts", "adv_usd_60"]].copy()
    right["asset_id"] = right["asset_id"].astype(str)
    right["ts"] = pd.to_datetime(right["ts"], utc=True)
    right = right.rename(columns={"ts": "setup_ts", "adv_usd_60": "adv_from_snap"})
    right = right.sort_values(["asset_id", "setup_ts"], kind="mergesort")
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


def _feature_row_from_record(record: Mapping[str, object]) -> FeatureRow:
    kwargs: dict[str, object] = {}
    for name in FEATURE_COLUMNS:
        value = record[name]
        if name == "ts":
            kwargs[name] = _as_utc(value)
        elif name == "regime":
            kwargs[name] = value if isinstance(value, Regime) else Regime(str(value))
        elif name == "vol_bucket":
            kwargs[name] = value if isinstance(value, VolBucket) else VolBucket(str(value))
        elif name == "market_regime":
            kwargs[name] = (
                value if isinstance(value, MarketRegime) else MarketRegime(str(value))
            )
        elif name in {"spy_above_ma", "is_warm"}:
            kwargs[name] = bool(value)
        elif name in {"xs_population", "bars_available", "bars_since_gap"}:
            kwargs[name] = _as_int(value)
        elif name == "symbol":
            kwargs[name] = str(value)
        else:
            kwargs[name] = _as_float(value)
    return FeatureRow(**kwargs)  # type: ignore[arg-type]


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"cannot convert {type(value).__name__} to int")


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"cannot convert {type(value).__name__} to float")
