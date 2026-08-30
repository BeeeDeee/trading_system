"""Point-in-time universe snapshot construction.

Every bar used for a snapshot at `ts` has `bar.ts <= ts`. A violation raises
ScoutLookaheadError and must not be caught.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from scout.config.schema import UniverseConfig
from scout.data.schemas import UNIVERSE_SNAPSHOT_COLUMNS
from scout.domain.enums import RejectionReason
from scout.domain.universe import UniverseEntry, UniverseSnapshot
from scout.universe.eligibility import ADV_WINDOW_BARS, apply_eligibility_rules
from scout.universe.spread import spread_bps_est
from scout.utils.errors import ScoutConfigError, ScoutDataError, ScoutLookaheadError

_FREQ_MONTHLY = "monthly"
_FREQ_WEEKLY = "weekly"


def snapshot_timestamps(
    calendar: pd.DataFrame,
    frequency: str,
    start: datetime,
    end: datetime,
) -> pd.DatetimeIndex:
    """First session of each month (equities) or ISO week. Inclusive bounds."""
    if frequency not in {_FREQ_MONTHLY, _FREQ_WEEKLY}:
        raise ScoutConfigError(
            f"universe.snapshot_frequency must be 'monthly' or 'weekly'; got {frequency!r}"
        )
    if calendar.empty:
        return pd.DatetimeIndex([], tz="UTC")
    close = _utc_ns(calendar["close_utc"])
    in_range = (close >= pd.Timestamp(start)) & (close <= pd.Timestamp(end))
    work = pd.DataFrame({"ts": close, "session_index": calendar["session_index"].to_numpy()})
    work = work.loc[in_range].sort_values("ts", kind="mergesort")
    if work.empty:
        return pd.DatetimeIndex([], tz="UTC")
    if frequency == _FREQ_MONTHLY:
        key = work["ts"].dt.strftime("%Y-%m")
    else:
        iso = work["ts"].dt.isocalendar()
        key = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    first = work.groupby(key, sort=False).head(1)
    return pd.DatetimeIndex(_utc_ns(first["ts"]))


def build_snapshots(
    panel: pd.DataFrame,
    *,
    candidates: pd.DataFrame,
    tickers: pd.DataFrame,
    calendar: pd.DataFrame,
    cfg: UniverseConfig,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """One row per (snapshot ts, candidate asset_id), eligible or not."""
    cand = _require_candidates(candidates)
    stamps = snapshot_timestamps(calendar, cfg.snapshot_frequency, start, end)
    if len(stamps) == 0:
        raise ScoutDataError("no snapshot timestamps in [start, end]; check calendar and period")
    grid = _candidate_grid(cand, stamps, calendar)
    meta = _ticker_meta(tickers)
    before = len(grid)
    grid = grid.merge(meta, on="asset_id", how="left", validate="many_to_one")
    if len(grid) != before:
        raise ScoutDataError("ticker merge duplicated snapshot rows")
    metrics = _panel_metrics(panel, cfg)
    joined = _asof_metrics(grid, metrics)
    _assert_no_lookahead(joined)
    joined = apply_delisting_grace(joined, cfg.delisting_grace_bars)
    ranked = _assign_adv_rank(joined)
    ranked = apply_eligibility_rules(ranked, cfg)
    return _finalize(ranked)


def lookup_snapshot(snapshots: pd.DataFrame, ts: datetime) -> pd.DataFrame:
    """Latest snapshot with snapshot.ts <= ts. Empty if none (conservative)."""
    if snapshots.empty:
        return snapshots.iloc[0:0].copy()
    requested = pd.Timestamp(ts)
    if requested.tzinfo is None:
        raise ValueError("lookup ts must be timezone-aware UTC")
    requested = requested.tz_convert("UTC")
    col = _utc_ns(snapshots["ts"])
    eligible = col[col <= requested]
    if len(eligible) == 0:
        return snapshots.iloc[0:0].copy()
    chosen = eligible.max()
    return snapshots.loc[col == chosen].copy()


def format_snapshot_summary(snapshots: pd.DataFrame) -> list[str]:
    """Human-readable eligible counts by year and top rejection reasons."""
    lines: list[str] = ["eligible_by_year"]
    year = pd.to_datetime(snapshots["ts"], utc=True).dt.year
    eligible = snapshots["eligible"].astype(bool)
    counts = year.loc[eligible].value_counts()
    years = sorted(int(y) for y in year.unique().tolist())
    for y in years:
        count = counts.get(y, 0)
        lines.append(f"{y} {int(count)}")
    lines.append("rejection_reasons")
    reasons = snapshots.loc[~eligible, "reason"].astype(str)
    reasons = reasons.loc[reasons.ne("")]
    if reasons.empty:
        lines.append("(none)")
        return lines
    for reason, count in reasons.value_counts().items():
        lines.append(f"{reason} {int(count)}")
    return lines


def to_universe_snapshot(frame: pd.DataFrame) -> UniverseSnapshot:
    """Typed view of one snapshot. Keyed by display symbol as the domain requires."""
    if frame.empty:
        raise ValueError("cannot build UniverseSnapshot from an empty frame")
    ts_values = _utc_ns(frame["ts"]).unique()
    if len(ts_values) != 1:
        raise ValueError(f"frame must contain exactly one snapshot ts; got {len(ts_values)}")
    ts = pd.Timestamp(ts_values[0]).to_pydatetime()
    entries: dict[str, UniverseEntry] = {}
    ordered = frame.sort_values("asset_id", kind="mergesort")
    for row in ordered.itertuples(index=False):
        symbol = str(row.symbol)
        if symbol in entries:
            raise ScoutDataError(f"duplicate display symbol {symbol!r} in one snapshot")
        reason_raw = str(row.reason) if row.reason is not None else ""
        eligible = bool(row.eligible)
        reason = None if eligible else RejectionReason(reason_raw)
        adv = _as_float(getattr(row, "adv_usd_60", None))
        spread = _as_float(getattr(row, "spread_bps_est", None))
        bars_f = _as_float(getattr(row, "bars_available", 0))
        bars = 0 if not np.isfinite(bars_f) else int(bars_f)
        listed_days = _as_float(getattr(row, "listed_days", None))
        entries[symbol] = UniverseEntry(
            symbol=symbol,
            eligible=eligible,
            reason=reason,
            adv_usd_30=adv,
            spread_bps_est=spread,
            bars_available=bars,
            listed_days=listed_days,
        )
    return UniverseSnapshot(ts=ts, entries=entries)


def _require_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ("asset_id", "symbol") if c not in candidates.columns]
    if missing:
        raise ScoutDataError(f"candidates missing columns: {missing}")
    out = candidates.loc[:, ["asset_id", "symbol"]].copy()
    out = out.assign(asset_id=out["asset_id"].astype(str), symbol=out["symbol"].astype(str))
    if out.empty:
        raise ScoutDataError("candidate list is empty")
    if out["asset_id"].duplicated().any():
        raise ScoutDataError("candidate list has duplicate asset_id values")
    return out.reset_index(drop=True)


def _candidate_grid(
    candidates: pd.DataFrame,
    stamps: pd.DatetimeIndex,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    stamp_frame = pd.DataFrame({"ts": _utc_ns(stamps)})
    cal = pd.DataFrame(
        {
            "ts": _utc_ns(calendar["close_utc"]),
            "session": calendar["session"].map(_as_date),
            "snapshot_session_index": pd.to_numeric(calendar["session_index"], errors="coerce"),
        }
    )
    before = len(stamp_frame)
    stamp_frame = stamp_frame.merge(cal, on="ts", how="left", validate="many_to_one")
    if len(stamp_frame) != before:
        raise ScoutDataError("calendar merge duplicated snapshot timestamps")
    grid = candidates.merge(stamp_frame, how="cross")
    grid["ts"] = _utc_ns(grid["ts"])
    return grid


def _ticker_meta(tickers: pd.DataFrame) -> pd.DataFrame:
    cols = ["asset_id", "symbol", "sector", "is_etf", "delisted_date", "listed_date"]
    have = [c for c in cols if c in tickers.columns]
    if "asset_id" not in have:
        raise ScoutDataError("tickers missing asset_id")
    meta = tickers.loc[:, have].copy()
    meta = meta.assign(asset_id=meta["asset_id"].astype(str))
    if "symbol" in meta.columns:
        meta = meta.drop(columns=["symbol"])
    if "sector" not in meta.columns:
        meta = meta.assign(sector="")
    else:
        meta = meta.assign(sector=meta["sector"].fillna("").astype(str))
    if "is_etf" not in meta.columns:
        meta = meta.assign(is_etf=False)
    else:
        meta = meta.assign(is_etf=meta["is_etf"].fillna(False).astype(bool))
    if "delisted_date" not in meta.columns:
        meta = meta.assign(delisted_date=pd.NaT)
    if "listed_date" not in meta.columns:
        meta = meta.assign(listed_date=pd.NaT)
    return meta.drop_duplicates(subset=["asset_id"], keep="last")


def _panel_metrics(panel: pd.DataFrame, cfg: UniverseConfig) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(
            {
                "asset_id": pd.Series(dtype="object"),
                "ts": pd.Series(dtype="datetime64[ns, UTC]"),
                "close_raw": pd.Series(dtype="float64"),
                "session_index": pd.Series(dtype="int32"),
                "adv_usd_60": pd.Series(dtype="float64"),
                "spread_bps_est": pd.Series(dtype="float64"),
                "bars_available": pd.Series(dtype="int32"),
                "bars_since_gap": pd.Series(dtype="int32"),
                "suspect_in_lookback": pd.Series(dtype=bool),
            }
        )
    required = (
        "asset_id",
        "ts",
        "high",
        "low",
        "close_raw",
        "dollar_volume",
        "session_index",
        "is_suspect",
    )
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ScoutDataError(f"panel missing columns: {missing}")
    parts: list[pd.DataFrame] = []
    working = panel.assign(asset_id=panel["asset_id"].astype(str))
    for _, group in working.groupby("asset_id", sort=False, observed=True):
        parts.append(_symbol_metrics(group, cfg))
    return pd.concat(parts, ignore_index=True)


def _symbol_metrics(group: pd.DataFrame, cfg: UniverseConfig) -> pd.DataFrame:
    g = group.sort_values("ts", kind="mergesort")
    n = len(g)
    dollar = g["dollar_volume"].astype("float64")
    adv = dollar.rolling(ADV_WINDOW_BARS, min_periods=ADV_WINDOW_BARS).median()
    spread = spread_bps_est(g["high"], g["low"], adv, cfg)
    session = pd.to_numeric(g["session_index"], errors="coerce")
    gap = _bars_since_gap(session)
    suspect = g["is_suspect"].astype(bool)
    lookback = (
        suspect.rolling(cfg.max_suspect_lookback, min_periods=1).max().fillna(False).astype(bool)
    )
    ts = _utc_ns(g["ts"])
    return pd.DataFrame(
        {
            "asset_id": g["asset_id"].astype(str).to_numpy(),
            "ts": ts,
            "close_raw": g["close_raw"].astype("float64").to_numpy(),
            "session_index": session.fillna(-1).astype("int32").to_numpy(),
            "adv_usd_60": adv.to_numpy(),
            "spread_bps_est": spread.to_numpy(),
            "bars_available": np.arange(1, n + 1, dtype=np.int32),
            "bars_since_gap": gap.astype("int32").to_numpy(),
            "suspect_in_lookback": lookback.to_numpy(),
        }
    )


def _bars_since_gap(session_index: pd.Series) -> pd.Series:
    # Calendar holes are session_index jumps > 1. The first bar after a hole
    # is 0 (conservative); each subsequent consecutive session increments.
    delta = session_index.diff()
    is_break = delta > 1
    group_id = is_break.cumsum()
    return session_index.groupby(group_id).cumcount()


def _asof_metrics(grid: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    left = grid.assign(ts=_utc_ns(grid["ts"]))
    left = left.sort_values(["ts", "asset_id"], kind="mergesort").reset_index(drop=True)
    left = left.assign(_row=np.arange(len(left), dtype=np.int64))
    if metrics.empty:
        joined = left.assign(
            close_raw=np.nan,
            session_index=np.nan,
            adv_usd_60=np.nan,
            spread_bps_est=np.nan,
            bars_available=np.int32(0),
            bars_since_gap=np.int32(0),
            suspect_in_lookback=False,
            panel_ts=pd.NaT,
        )
    else:
        right = metrics.assign(ts=_utc_ns(metrics["ts"]), panel_ts=_utc_ns(metrics["ts"]))
        right = right.sort_values(["ts", "asset_id"], kind="mergesort")
        before = len(left)
        joined = pd.merge_asof(
            left,
            right,
            on="ts",
            by="asset_id",
            direction="backward",
        )
        if len(joined) != before:
            raise ScoutDataError("metrics asof-merge duplicated snapshot rows")
    joined = joined.sort_values("_row", kind="mergesort")
    panel_ts = pd.to_datetime(joined["panel_ts"], utc=True, errors="coerce")
    snap_ts = _utc_ns(joined["ts"])
    has_session = panel_ts.notna() & (panel_ts == snap_ts)
    return joined.assign(
        has_session_at_ts=has_session.to_numpy(),
        panel_ts=panel_ts,
        bars_available=pd.to_numeric(joined["bars_available"], errors="coerce")
        .fillna(0)
        .astype("int32"),
        bars_since_gap=pd.to_numeric(joined["bars_since_gap"], errors="coerce")
        .fillna(0)
        .astype("int32"),
        suspect_in_lookback=joined["suspect_in_lookback"].fillna(False).astype(bool),
        is_delisted=_delisted_by_ticker(joined).to_numpy(),
    )


def _delisted_by_ticker(frame: pd.DataFrame) -> pd.Series:
    if "delisted_date" not in frame.columns:
        return pd.Series(False, index=frame.index)
    delisted = pd.to_datetime(frame["delisted_date"], errors="coerce")
    session = pd.to_datetime(frame["session"], errors="coerce")
    if getattr(delisted.dt, "tz", None) is not None:
        delisted = delisted.dt.tz_convert("UTC").dt.tz_localize(None)
    if getattr(session.dt, "tz", None) is not None:
        session = session.dt.tz_convert("UTC").dt.tz_localize(None)
    flag = delisted.notna() & (delisted.dt.normalize() <= session.dt.normalize())
    return pd.Series(flag.to_numpy(), index=frame.index, dtype=bool)


def apply_delisting_grace(frame: pd.DataFrame, grace_bars: int) -> pd.DataFrame:
    """Treat a last session older than `ts - grace` as delisted at `ts`."""
    last_idx = pd.to_numeric(frame["session_index"], errors="coerce")
    snap_idx = pd.to_numeric(frame["snapshot_session_index"], errors="coerce")
    by_grace = last_idx.notna() & snap_idx.notna() & (last_idx < (snap_idx - grace_bars))
    return frame.assign(is_delisted=frame["is_delisted"] | by_grace)


def _assert_no_lookahead(frame: pd.DataFrame) -> None:
    if "panel_ts" not in frame.columns:
        return
    panel_ts = pd.to_datetime(frame["panel_ts"], utc=True, errors="coerce")
    snap_ts = _utc_ns(frame["ts"])
    leaked = panel_ts.notna() & (panel_ts > snap_ts)
    if bool(leaked.any()):
        raise ScoutLookaheadError("universe snapshot used a bar with ts > snapshot ts")


def _assign_adv_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """1 = most liquid within the candidate set at this snapshot. Ties: asset_id."""
    ordered = frame.sort_values(
        ["ts", "adv_usd_60", "asset_id"],
        ascending=[True, False, True],
        kind="mergesort",
        na_position="last",
    )
    rank = ordered.groupby("ts", sort=False).cumcount() + 1
    rank_s = pd.Series(rank.to_numpy(), index=ordered.index, dtype="int32")
    rank_s = rank_s.mask(ordered["adv_usd_60"].isna())
    out = ordered.assign(adv_rank=rank_s.astype("Int32"))
    return out.sort_values(["ts", "asset_id"], kind="mergesort").reset_index(drop=True)


def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
    reason = frame["reason"].fillna("").astype(str)
    reason = reason.mask(frame["eligible"].astype(bool), "")
    result = pd.DataFrame(
        {
            "ts": _utc_ns(frame["ts"]),
            "asset_id": frame["asset_id"].astype(str),
            "symbol": frame["symbol"].astype(str),
            "eligible": frame["eligible"].astype(bool),
            "reason": reason,
            "adv_usd_60": frame["adv_usd_60"].astype("float64"),
            "adv_rank": frame["adv_rank"].astype("Int32"),
            "spread_bps_est": frame["spread_bps_est"].astype("float64"),
            "close_raw": frame["close_raw"].astype("float64"),
            "bars_available": frame["bars_available"].astype("int32"),
            "sector": frame["sector"].fillna("").astype(str),
            "is_etf": frame["is_etf"].fillna(False).astype(bool),
        }
    )
    result = result.loc[:, list(UNIVERSE_SNAPSHOT_COLUMNS)]
    return result.sort_values(["ts", "asset_id"], kind="mergesort").reset_index(drop=True)


def _utc_ns(values: object) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns")


def _as_date(value: object) -> object:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize().date()


def _as_float(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        as_float = float(str(value))
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(as_float):
        return float("nan")
    return as_float
