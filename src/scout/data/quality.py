"""Suspect-session flags. Never fill a gap; never delete a row."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from scout.domain.enums import ActionType

_LARGE_ABS_LOG_RETURN = 1.0
_STALE_CLOSE_SESSIONS = 5


def flag_suspect(
    panel: pd.DataFrame,
    *,
    actions: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Set `is_suspect` from §6.7 checks. Row count is unchanged."""
    masks = quality_masks(panel, actions=actions, calendar=calendar)
    combined = masks.any(axis=1)
    return panel.assign(is_suspect=combined.astype(bool))


def quality_masks(
    panel: pd.DataFrame,
    *,
    actions: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Boolean columns, one per quality check, aligned to `panel`."""
    n = len(panel)
    if n == 0:
        return pd.DataFrame(
            {
                "ohlc_invalid": pd.Series(dtype=bool),
                "large_return": pd.Series(dtype=bool),
                "zero_volume": pd.Series(dtype=bool),
                "stale_close": pd.Series(dtype=bool),
                "not_in_calendar": pd.Series(dtype=bool),
            }
        )
    high = panel["high"].astype("float64")
    low = panel["low"].astype("float64")
    close = panel["close"].astype("float64")
    ohlc_invalid = (high < low) | (close < low) | (close > high)
    volume = panel["volume"].astype("float64")
    zero_volume = volume == 0.0
    not_in_calendar = _not_in_calendar(panel, calendar)
    large_return = _large_unexplained_return(panel, actions)
    stale_close = _stale_close(panel)
    return pd.DataFrame(
        {
            "ohlc_invalid": ohlc_invalid.to_numpy(),
            "large_return": large_return.to_numpy(),
            "zero_volume": zero_volume.to_numpy(),
            "stale_close": stale_close.to_numpy(),
            "not_in_calendar": not_in_calendar.to_numpy(),
        },
        index=panel.index,
    )


def suspect_rate(panel: pd.DataFrame) -> float:
    if panel.empty:
        return 0.0
    return float(panel["is_suspect"].mean())


def _panel_session_keys(panel: pd.DataFrame) -> pd.Series:
    if "session" in panel.columns:
        return panel["session"].map(_session_key)
    ts = pd.to_datetime(panel["ts"], utc=True)
    return ts.dt.tz_convert("America/New_York").dt.date


def _not_in_calendar(panel: pd.DataFrame, calendar: pd.DataFrame) -> pd.Series:
    if "session_index" in panel.columns:
        idx = pd.to_numeric(panel["session_index"], errors="coerce")
        missing_idx = idx.isna() | (idx < 0)
    else:
        missing_idx = pd.Series(False, index=panel.index)
    if calendar.empty:
        cal_sessions: set[object] = set()
    else:
        cal_sessions = {_session_key(v) for v in calendar["session"].tolist()}
    absent = _panel_session_keys(panel).map(lambda v: v not in cal_sessions)
    return missing_idx | absent


def _large_unexplained_return(panel: pd.DataFrame, actions: pd.DataFrame) -> pd.Series:
    close_raw = panel["close_raw"].astype("float64")
    session_index = pd.to_numeric(panel["session_index"], errors="coerce")
    prev_close = close_raw.groupby(panel["asset_id"], sort=False).shift(1)
    prev_idx = session_index.groupby(panel["asset_id"], sort=False).shift(1)
    adjacent = session_index == (prev_idx + 1)
    valid = (close_raw > 0.0) & (prev_close > 0.0) & adjacent
    logret = pd.Series(np.nan, index=panel.index, dtype="float64")
    logret = logret.mask(valid, np.log(close_raw / prev_close))
    big = valid & (logret.abs() > _LARGE_ABS_LOG_RETURN)
    has_action = _has_action_on_session(panel, actions)
    return big & ~has_action


def _has_action_on_session(panel: pd.DataFrame, actions: pd.DataFrame) -> pd.Series:
    empty = pd.Series(False, index=panel.index)
    if actions.empty:
        return empty
    priced = actions.loc[
        actions["action_type"].isin(
            [ActionType.SPLIT.value, ActionType.DIVIDEND.value, ActionType.SPINOFF.value]
        )
    ]
    if priced.empty:
        return empty
    keys = priced.loc[:, ["asset_id", "ex_date"]].copy()
    keys = keys.assign(
        asset_id=keys["asset_id"].astype(str),
        ex_key=keys["ex_date"].map(_session_key),
    )
    keys = keys.drop_duplicates(subset=["asset_id", "ex_key"])
    keys = keys.assign(_has_action=True)
    left = panel.loc[:, ["asset_id"]].copy()
    left = left.assign(
        asset_id=left["asset_id"].astype(str),
        ex_key=_panel_session_keys(panel),
    )
    before = len(left)
    merged = left.merge(
        keys.loc[:, ["asset_id", "ex_key", "_has_action"]],
        on=["asset_id", "ex_key"],
        how="left",
        validate="many_to_one",
    )
    if len(merged) != before:
        raise ValueError("action merge duplicated panel rows")
    flag = merged["_has_action"].eq(True)
    return pd.Series(flag.to_numpy(), index=panel.index, dtype=bool)


def _stale_close(panel: pd.DataFrame) -> pd.Series:
    close_raw = panel["close_raw"].astype("float64")
    volume = panel["volume"].astype("float64")
    session_index = pd.to_numeric(panel["session_index"], errors="coerce")
    flags = pd.Series(False, index=panel.index)
    for _, grp in panel.groupby("asset_id", sort=False):
        order = np.argsort(session_index.loc[grp.index].to_numpy(), kind="mergesort")
        idx = grp.index.to_numpy()[order]
        c = close_raw.loc[idx].to_numpy()
        v = volume.loc[idx].to_numpy()
        s = session_index.loc[idx].to_numpy()
        n = len(idx)
        if n == 0:
            continue
        continues = np.zeros(n, dtype=bool)
        for i in range(1, n):
            continues[i] = (
                np.isfinite(s[i])
                and np.isfinite(s[i - 1])
                and s[i] == s[i - 1] + 1
                and v[i] > 0.0
                and v[i - 1] > 0.0
                and c[i] == c[i - 1]
            )
        run_id = np.cumsum(~continues)
        _, inverse, counts = np.unique(run_id, return_inverse=True, return_counts=True)
        long_run = counts[inverse] >= _STALE_CLOSE_SESSIONS
        vol_ok = v > 0.0
        flags.loc[idx] = long_run & vol_ok
    return flags


def _session_key(value: object) -> object:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    return ts.normalize().date()
