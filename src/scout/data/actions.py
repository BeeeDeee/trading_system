"""Causal ratio back-adjustment from unadjusted OHLCV plus an actions table.

A split or dividend on session `t` scales every session *strictly before* `t`.
Sessions on and after the ex-date keep their traded (raw) prices. `close_raw`
is the unadjusted close and is never rewritten. See ADR-017.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from scout.config.schema import ScoutConfig
from scout.data.calendar import (
    build_calendar,
    reference_calendar_path,
    write_calendar,
)
from scout.data.quality import flag_suspect, suspect_rate
from scout.data.schemas import (
    ACTIONS_COLUMNS,
    CALENDAR_COLUMNS,
    OHLCV_COLUMNS,
    PANEL_COLUMNS,
)
from scout.data.store import read_json, read_parquet, write_parquet_atomic
from scout.domain.enums import ActionType
from scout.utils.errors import ScoutDataError

_PRICE_ACTIONS = frozenset(
    {ActionType.SPLIT.value, ActionType.DIVIDEND.value, ActionType.SPINOFF.value}
)


@dataclass(frozen=True, slots=True)
class AdjustResult:
    calendar_rows: int
    panel_rows: int
    suspect_rows: int
    suspect_rate: float
    date_min: str
    date_max: str


def adjustment_factors(
    actions: pd.DataFrame,
    sessions: pd.DatetimeIndex | pd.Series,
    close_raw: pd.Series,
) -> pd.Series:
    """Cumulative multiplicative factor per session, ending at 1.0 on the
    final session. Applied to open/high/low/close; volume is divided by it.

    Per-action factor, effective from ex_date onward (applied to all sessions
    STRICTLY BEFORE ex_date when accumulating backwards):
      SPLIT:    f = 1 / split_ratio
      DIVIDEND: f = 1 - cash_amount / close_on_session_before_ex_date
      SPINOFF:  f = 1 - cash_amount / close_on_session_before_ex_date
    """
    session_ts = pd.to_datetime(pd.Series(sessions, index=close_raw.index), errors="coerce")
    if session_ts.dt.tz is not None:
        session_ts = session_ts.dt.tz_convert("UTC").dt.tz_localize(None)
    session_ts = session_ts.dt.normalize()
    n = len(session_ts)
    if n == 0:
        return pd.Series(dtype="float64", index=close_raw.index)
    factors = np.ones(n, dtype=np.float64)
    if actions.empty:
        return pd.Series(factors, index=close_raw.index)
    events = _per_action_events(actions, session_ts, close_raw)
    if not events:
        return pd.Series(factors, index=close_raw.index)
    events = sorted(events, key=lambda item: item[0])
    session_np = pd.to_datetime(session_ts)
    cum = 1.0
    a = len(events) - 1
    for i in range(n - 1, -1, -1):
        s = pd.Timestamp(session_np.iloc[i])
        while a >= 0 and events[a][0] > s:
            cum *= events[a][1]
            a -= 1
        factors[i] = cum
    return pd.Series(factors, index=close_raw.index)


def apply_adjustments(ohlcv: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Add `close_raw`, `volume_raw`, and split/dividend-adjusted OHLC/volume."""
    if ohlcv.empty:
        out = ohlcv.copy()
        return out.assign(
            close_raw=pd.Series(dtype="float64"),
            volume_raw=pd.Series(dtype="float64"),
            factor=pd.Series(dtype="float64"),
        )
    parts: list[pd.DataFrame] = []
    action_groups = {
        str(aid): grp for aid, grp in actions.groupby("asset_id", sort=False)
    }
    empty_actions = pd.DataFrame(columns=list(ACTIONS_COLUMNS))
    for asset_id, grp in ohlcv.groupby("asset_id", sort=False):
        asset_actions = action_groups.get(str(asset_id), empty_actions)
        parts.append(_adjust_one(grp, asset_actions))
    return pd.concat(parts, ignore_index=True)


def build_processed_panel(
    ohlcv: pd.DataFrame,
    actions: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Adjusted panel, long format. Missing sessions stay missing. Never filled."""
    if ohlcv.empty:
        return pd.DataFrame(columns=list(PANEL_COLUMNS))
    adjusted = apply_adjustments(ohlcv, actions)
    working = _join_calendar(adjusted, calendar)
    working = flag_suspect(working, actions=actions, calendar=calendar)
    volume_raw = working["volume_raw"].astype("float64")
    close_raw = working["close_raw"].astype("float64")
    out = working.assign(dollar_volume=close_raw * volume_raw)
    out = out.loc[:, list(PANEL_COLUMNS)]
    out = out.sort_values(["ts", "asset_id"], kind="mergesort").reset_index(drop=True)
    return out


def run_adjust(cfg: ScoutConfig) -> AdjustResult:
    raw_dir = Path(cfg.data.raw_dir)
    snapshot_path = Path(cfg.data.snapshot_id_path)
    if not snapshot_path.is_file():
        raise ScoutDataError(f"snapshot not found: {snapshot_path}; run ingest first")
    read_json(snapshot_path)
    ohlcv = _load_raw_ohlcv(raw_dir)
    actions_path = raw_dir / "actions.parquet"
    if not actions_path.is_file():
        raise ScoutDataError(f"actions parquet not found: {actions_path}")
    actions = read_parquet(actions_path)
    start, end = _session_bounds(ohlcv, cfg)
    calendar = build_calendar(start, end, calendar_code=cfg.data.calendar)
    cal_path = reference_calendar_path(Path(cfg.data.processed_dir), cfg.data.calendar)
    write_calendar(cal_path, calendar)
    panel = build_processed_panel(ohlcv, actions, calendar)
    _write_panel_by_year(Path(cfg.data.processed_dir) / "panel" / "1d", panel)
    n_suspect = int(panel["is_suspect"].sum()) if not panel.empty else 0
    if ohlcv.empty:
        sessions = pd.Series(dtype="datetime64[ns]")
    else:
        sessions = pd.to_datetime(ohlcv["session"])
    return AdjustResult(
        calendar_rows=int(len(calendar)),
        panel_rows=int(len(panel)),
        suspect_rows=n_suspect,
        suspect_rate=suspect_rate(panel),
        date_min="" if sessions.empty else sessions.min().date().isoformat(),
        date_max="" if sessions.empty else sessions.max().date().isoformat(),
    )


def _adjust_one(grp: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    ordered = grp.sort_values("session", kind="mergesort")
    close_raw = ordered["close"].astype("float64")
    volume_raw = ordered["volume"].astype("float64")
    session_ts = pd.to_datetime(ordered["session"])
    factor = adjustment_factors(actions, session_ts, close_raw)
    factor_v = factor.to_numpy()
    return ordered.assign(
        close_raw=close_raw.to_numpy(),
        volume_raw=volume_raw.to_numpy(),
        factor=factor_v,
        open=ordered["open"].astype("float64") * factor_v,
        high=ordered["high"].astype("float64") * factor_v,
        low=ordered["low"].astype("float64") * factor_v,
        close=close_raw.to_numpy() * factor_v,
        volume=volume_raw.to_numpy() / factor_v,
    )


def _per_action_events(
    actions: pd.DataFrame,
    session_ts: pd.Series,
    close_raw: pd.Series,
) -> list[tuple[pd.Timestamp, float]]:
    events: list[tuple[pd.Timestamp, float]] = []
    for row in actions.itertuples(index=False):
        action_type = str(row.action_type)
        if action_type not in _PRICE_ACTIONS:
            continue
        ex = pd.Timestamp(row.ex_date).tz_localize(None).normalize()
        if action_type == ActionType.SPLIT.value:
            ratio = float(row.split_ratio)
            if ratio <= 0.0:
                continue
            f = 1.0 / ratio
        else:
            cash = float(row.cash_amount)
            f = _cash_factor(cash, ex, session_ts, close_raw)
        if f != 1.0:
            events.append((ex, f))
    return events


def _cash_factor(
    cash: float,
    ex: pd.Timestamp,
    session_ts: pd.Series,
    close_raw: pd.Series,
) -> float:
    if cash == 0.0:
        return 1.0
    before = session_ts < ex
    if not bool(before.to_numpy().any()):
        return 1.0
    prev = float(close_raw.loc[before].iloc[-1])
    if prev <= 0.0:
        return 1.0
    f = 1.0 - cash / prev
    if f <= 0.0:
        return 1.0
    return f


def _join_calendar(adjusted: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    left = adjusted.copy()
    left = left.assign(_session_key=left["session"].map(_session_key))
    right = calendar.loc[:, list(CALENDAR_COLUMNS)].copy()
    right = right.assign(_session_key=right["session"].map(_session_key))
    right = right.drop(columns=["session"])
    before = len(left)
    merged = left.merge(
        right,
        on="_session_key",
        how="left",
        validate="many_to_one",
    )
    if len(merged) != before:
        raise ScoutDataError("calendar merge duplicated OHLCV rows")
    missing = merged["close_utc"].isna()
    fallback = _fallback_close_utc(merged["_session_key"])
    ts = merged["close_utc"].copy()
    ts = ts.mask(missing, fallback)
    ts = pd.to_datetime(ts, utc=True)
    session_index = pd.to_numeric(merged["session_index"], errors="coerce")
    session_index = session_index.fillna(-1).astype("int32")
    return merged.assign(ts=ts, session_index=session_index).drop(columns=["_session_key"])


def _fallback_close_utc(keys: pd.Series) -> pd.Series:
    naive = pd.to_datetime(keys, errors="coerce")
    localized = naive.dt.tz_localize("America/New_York")
    return (localized + pd.Timedelta(hours=16)).dt.tz_convert("UTC")


def _session_key(value: object) -> object:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_localize(None).normalize().date() if ts.tzinfo else ts.normalize().date()


def _load_raw_ohlcv(raw_dir: Path) -> pd.DataFrame:
    ohlcv_dir = raw_dir / "ohlcv" / "1d"
    files = sorted(ohlcv_dir.glob("*.parquet"))
    if not files:
        raise ScoutDataError(f"no OHLCV parquet files under {ohlcv_dir}")
    frames = [read_parquet(path) for path in files]
    out = pd.concat(frames, ignore_index=True)
    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ScoutDataError(f"ohlcv missing columns: {missing}")
    return out.sort_values(["session", "asset_id"], kind="mergesort").reset_index(drop=True)


def _session_bounds(ohlcv: pd.DataFrame, cfg: ScoutConfig) -> tuple[date, date]:
    sessions = pd.to_datetime(ohlcv["session"])
    data_min = sessions.min().date()
    data_max = sessions.max().date()
    start = min(data_min, cfg.period.start.date())
    end = max(data_max, cfg.period.end.date())
    return start, end


def _write_panel_by_year(out_dir: Path, panel: pd.DataFrame) -> None:
    if panel.empty:
        return
    years = pd.to_datetime(panel["ts"], utc=True).dt.year
    for year in sorted(int(y) for y in years.unique()):
        part = panel.loc[years == year].reset_index(drop=True)
        write_parquet_atomic(out_dir / f"{year}.parquet", part)
