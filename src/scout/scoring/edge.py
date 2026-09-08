"""Walk-forward bin statistics. Sample rule: resolution_ts < as_of. Always."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

from scout.config.schema import EdgeConfig, LcbMethod
from scout.domain.edge import BinKey, BinStats, EdgeTable
from scout.domain.enums import Direction, SetupOutcome, VolBucket
from scout.utils.errors import ScoutConfigError, ScoutDataError, ScoutLookaheadError
from scout.utils.stats import bootstrap_lcb

REQUIRED_EDGE_COLUMNS: tuple[str, ...] = (
    "strategy_id",
    "direction",
    "vol_bucket",
    "resolution_ts",
    "realised_r_gross",
    "outcome",
    "bars_held",
)

_DOCUMENTED_BIN_DIMENSIONS: tuple[str, ...] = ("strategy_id", "direction", "vol_bucket")
_BOOTSTRAP_QUANTILE = 0.10


def normal_lcb(mean_r: float, std_r: float, n: int, z: float) -> float:
    """One-sided normal LCB: mean - z * std / sqrt(n)."""
    if n < 1:
        return float("nan")
    return mean_r - z * std_r / math.sqrt(n)


def monthly_as_of_grid(
    sessions: Sequence[datetime] | pd.DatetimeIndex,
) -> tuple[datetime, ...]:
    """First session close of each month, ascending UTC."""
    if isinstance(sessions, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(pd.to_datetime(sessions, utc=True))
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(list(sessions), utc=True))
    if len(idx) == 0:
        return ()
    idx = idx.sort_values()
    work = pd.DataFrame({"ts": idx})
    key = work["ts"].dt.strftime("%Y-%m")
    first = work.groupby(key, sort=True).head(1)["ts"]
    out: list[datetime] = []
    for ts in first:
        dt = pd.Timestamp(ts)
        dt = dt.tz_localize("UTC") if dt.tzinfo is None else dt.tz_convert("UTC")
        py = dt.to_pydatetime()
        if py.tzinfo is None:
            py = py.replace(tzinfo=UTC)
        out.append(py)
    return tuple(out)


def compute_bin_stats(
    sample: pd.DataFrame,
    key: BinKey,
    as_of: datetime,
    cfg: EdgeConfig,
) -> BinStats:
    """Statistics of `sample`. Caller must already have filtered to the bin.

    Raises ScoutLookaheadError if any row has resolution_ts >= as_of.
    """
    _assert_causal(sample, as_of)
    work = _finite_outcomes(sample)
    n = len(work)
    if n == 0:
        return _empty_stats(key, as_of)
    r = work["realised_r_gross"].to_numpy(dtype=np.float64)
    mean_r = float(np.mean(r))
    if n < 2:
        std_r = float("nan")
        ev_r_lcb = float("nan")
    else:
        std_r = float(np.std(r, ddof=1))
        ev_r_lcb = _lcb(r, mean_r, std_r, n, cfg)
    wins = r[r > 0.0]
    losses = r[r < 0.0]
    outcomes = work["outcome"].astype(str)
    return BinStats(
        key=key,
        as_of=as_of,
        n=n,
        mean_r=mean_r,
        std_r=std_r,
        ev_r_lcb=ev_r_lcb,
        mean_bars_held=float(work["bars_held"].mean()),
        win_rate=float(np.mean(r > 0.0)),
        avg_win_r=float(wins.mean()) if wins.size else 0.0,
        avg_loss_r=float(losses.mean()) if losses.size else 0.0,
        target_rate=float((outcomes == SetupOutcome.TARGET.value).mean()),
        stop_rate=float((outcomes == SetupOutcome.STOP.value).mean()),
        time_rate=float((outcomes == SetupOutcome.TIME.value).mean()),
        median_r=float(np.median(r)),
        p05_r=float(np.quantile(r, 0.05)),
        p95_r=float(np.quantile(r, 0.95)),
    )


def build_edge_table(
    resolved: pd.DataFrame,
    cfg: EdgeConfig,
    *,
    sessions: Sequence[datetime] | pd.DatetimeIndex | None = None,
    config_hash: str = "",
    data_snapshot_id: str = "",
    on_progress: Callable[[int, int, BinKey], None] | None = None,
) -> EdgeTable:
    """Bin, then compute BinStats at each as_of grid point using only setups
    with resolution_ts < as_of.
    """
    _validate_edge_config(cfg)
    prepared = _prepare_resolved(resolved)
    grid = _grid(sessions, prepared)
    if prepared.empty or not grid:
        return EdgeTable((), config_hash=config_hash, data_snapshot_id=data_snapshot_id)
    stats: list[BinStats] = []
    grouped = prepared.groupby(["strategy_id", "direction", "vol_bucket"], sort=True)
    n_groups = int(grouped.ngroups)
    for i, ((strategy_id, direction, vol_bucket), group) in enumerate(grouped, start=1):
        key = BinKey(
            strategy_id=str(strategy_id),
            direction=Direction(str(direction)),
            vol_bucket=VolBucket(str(vol_bucket)),
        )
        if on_progress is not None:
            on_progress(i, n_groups, key)
        ordered = group.sort_values("resolution_ts", kind="mergesort")
        res_index = pd.DatetimeIndex(pd.to_datetime(ordered["resolution_ts"], utc=True))
        for as_of in grid:
            n = int(res_index.searchsorted(pd.Timestamp(as_of), side="left"))
            sample = ordered.iloc[:n]
            stats.append(compute_bin_stats(sample, key, as_of, cfg))
    return EdgeTable(stats, config_hash=config_hash, data_snapshot_id=data_snapshot_id)


def _validate_edge_config(cfg: EdgeConfig) -> None:
    if cfg.as_of_grid != "monthly":
        raise ScoutConfigError(
            f"edge.as_of_grid must be 'monthly'; got {cfg.as_of_grid!r}"
        )
    got = tuple(cfg.bin_dimensions)
    if got != _DOCUMENTED_BIN_DIMENSIONS:
        raise ScoutConfigError(
            f"edge.bin_dimensions must be {list(_DOCUMENTED_BIN_DIMENSIONS)}; got {list(got)}"
        )
    if cfg.lcb_method not in (LcbMethod.NORMAL, LcbMethod.BOOTSTRAP):
        raise ScoutConfigError(
            f"unknown lcb_method {cfg.lcb_method!r}; known: "
            f"{sorted(m.value for m in LcbMethod)}"
        )


def _grid(
    sessions: Sequence[datetime] | pd.DatetimeIndex | None,
    prepared: pd.DataFrame,
) -> tuple[datetime, ...]:
    if sessions is None:
        if prepared.empty:
            return ()
        sessions = pd.DatetimeIndex(pd.to_datetime(prepared["setup_ts"], utc=True))
    return monthly_as_of_grid(sessions)


def _prepare_resolved(resolved: pd.DataFrame) -> pd.DataFrame:
    if resolved.empty:
        return pd.DataFrame(columns=[*list(REQUIRED_EDGE_COLUMNS), "setup_ts"])
    missing = [c for c in REQUIRED_EDGE_COLUMNS if c not in resolved.columns]
    if missing:
        raise ScoutDataError(f"resolved setups missing columns: {missing}")
    cols = list(REQUIRED_EDGE_COLUMNS)
    if "setup_ts" in resolved.columns:
        cols = [*cols, "setup_ts"]
    work = resolved.loc[:, cols]
    work = work.assign(resolution_ts=pd.to_datetime(work["resolution_ts"], utc=True))
    if "setup_ts" in work.columns:
        work = work.assign(setup_ts=pd.to_datetime(work["setup_ts"], utc=True))
    return work.loc[work["resolution_ts"].notna()].reset_index(drop=True)


def _finite_outcomes(sample: pd.DataFrame) -> pd.DataFrame:
    if sample.empty:
        return sample
    r = pd.to_numeric(sample["realised_r_gross"], errors="coerce")
    return sample.loc[np.isfinite(r.to_numpy(dtype=np.float64))]


def _assert_causal(sample: pd.DataFrame, as_of: datetime) -> None:
    if sample.empty or "resolution_ts" not in sample.columns:
        return
    res = pd.to_datetime(sample["resolution_ts"], utc=True)
    valid = res.notna()
    if valid.any() and bool((res.loc[valid] >= pd.Timestamp(as_of)).any()):
        raise ScoutLookaheadError(
            f"sample contains resolution_ts >= as_of {as_of.isoformat()}"
        )


def _lcb(
    r: NDArray[np.float64],
    mean_r: float,
    std_r: float,
    n: int,
    cfg: EdgeConfig,
) -> float:
    if cfg.lcb_method is LcbMethod.BOOTSTRAP:
        return bootstrap_lcb(
            r,
            n_iterations=cfg.bootstrap_iterations,
            seed=cfg.bootstrap_seed,
            quantile=_BOOTSTRAP_QUANTILE,
        )
    return normal_lcb(mean_r, std_r, n, cfg.z)


def _empty_stats(key: BinKey, as_of: datetime) -> BinStats:
    nan = float("nan")
    return BinStats(
        key=key,
        as_of=as_of,
        n=0,
        mean_r=nan,
        std_r=nan,
        ev_r_lcb=nan,
        mean_bars_held=nan,
        win_rate=nan,
        avg_win_r=nan,
        avg_loss_r=nan,
        target_rate=nan,
        stop_rate=nan,
        time_rate=nan,
        median_r=nan,
        p05_r=nan,
        p95_r=nan,
    )
