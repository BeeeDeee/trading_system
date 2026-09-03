"""Stability tables: is the sign consistent, and is the dispersion explicable?"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

# ADR-019 names April 2009, January 2021, and the classic 1932 / 2009 episodes
# in the literature. Daniel and Moskowitz (2016) "Momentum Crashes" supplies
# the remaining worst months. Windows are calendar months, inclusive.
# 1932/1939 predate the v1 sample and will be empty; they stay so the go/no-go
# criterion names a fixed list rather than a sample-dependent one.
MOMENTUM_CRASH_WINDOWS: tuple[tuple[str, datetime, datetime], ...] = (
    ("1932-07", datetime(1932, 7, 1, tzinfo=UTC), datetime(1932, 8, 1, tzinfo=UTC)),
    ("1932-08", datetime(1932, 8, 1, tzinfo=UTC), datetime(1932, 9, 1, tzinfo=UTC)),
    ("1939-09", datetime(1939, 9, 1, tzinfo=UTC), datetime(1939, 10, 1, tzinfo=UTC)),
    ("2001-01", datetime(2001, 1, 1, tzinfo=UTC), datetime(2001, 2, 1, tzinfo=UTC)),
    ("2009-04", datetime(2009, 4, 1, tzinfo=UTC), datetime(2009, 5, 1, tzinfo=UTC)),
    ("2021-01", datetime(2021, 1, 1, tzinfo=UTC), datetime(2021, 2, 1, tzinfo=UTC)),
)


def by_year(trades: pd.DataFrame) -> pd.DataFrame:
    return _group_mean_r(trades, _year_key(trades))


def by_quarter(trades: pd.DataFrame) -> pd.DataFrame:
    return _group_mean_r(trades, _quarter_key(trades))


def by_regime(trades: pd.DataFrame) -> pd.DataFrame:
    return _group_mean_r(trades, trades["regime"] if "regime" in trades.columns else None)


def by_vol_bucket(trades: pd.DataFrame) -> pd.DataFrame:
    return _group_mean_r(
        trades, trades["vol_bucket"] if "vol_bucket" in trades.columns else None
    )


def by_direction(trades: pd.DataFrame) -> pd.DataFrame:
    return _group_mean_r(
        trades, trades["direction"] if "direction" in trades.columns else None
    )


def by_symbol(trades: pd.DataFrame) -> pd.DataFrame:
    table = _group_mean_r(trades, trades["symbol"] if "symbol" in trades.columns else None)
    if table.empty or "sum_r" not in table.columns:
        return table
    total = float(table["sum_r"].abs().sum())
    table = table.assign(
        pnl_share=table["sum_r"] / total if total else float("nan")
    )
    return table.sort_values("sum_r", ascending=False)


def top_n_pnl_share(trades: pd.DataFrame, n: int = 3) -> float:
    table = by_symbol(trades)
    if table.empty or "sum_r" not in table.columns:
        return float("nan")
    total = float(table["sum_r"].sum())
    if total == 0.0:
        return float("nan")
    top = float(table.head(n)["sum_r"].sum())
    return top / total


def by_cluster(trades: pd.DataFrame) -> pd.DataFrame:
    return _group_mean_r(
        trades, trades["cluster"] if "cluster" in trades.columns else None
    )


def by_strategy(trades: pd.DataFrame) -> pd.DataFrame:
    return _group_mean_r(
        trades, trades["strategy_id"] if "strategy_id" in trades.columns else None
    )


def spy_beta(equity: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """OLS of strategy session returns on SPY session returns.

    Returns beta, alpha_annualised_pct (fraction), t_alpha.
    """
    y = equity.pct_change().dropna()
    x = benchmark.pct_change().dropna()
    aligned = pd.concat([y, x], axis=1, join="inner").dropna()
    aligned.columns = ["y", "x"]
    if len(aligned) < 3:
        return {
            "beta": float("nan"),
            "alpha_annualised_pct": float("nan"),
            "t_alpha": float("nan"),
        }
    y_arr = aligned["y"].to_numpy(dtype=np.float64)
    x_arr = aligned["x"].to_numpy(dtype=np.float64)
    n = y_arr.size
    design = np.column_stack([np.ones(n), x_arr])
    coef, _, _, _ = np.linalg.lstsq(design, y_arr, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y_arr - design @ coef
    dof = n - 2
    mse = float(resid @ resid) / dof if dof > 0 else float("nan")
    xtx_inv = np.linalg.inv(design.T @ design)
    se_alpha = math.sqrt(mse * float(xtx_inv[0, 0])) if mse == mse else float("nan")
    t_alpha = alpha / se_alpha if se_alpha and se_alpha != 0.0 else float("nan")
    return {
        "beta": beta,
        "alpha_annualised_pct": alpha * 252.0,
        "t_alpha": t_alpha,
    }


def momentum_crash_returns(equity: pd.Series) -> pd.DataFrame:
    """Total return of the equity curve inside each named crash window."""
    eq = equity.copy()
    eq.index = pd.DatetimeIndex(pd.to_datetime(eq.index, utc=True))
    rows = []
    for name, start, end in MOMENTUM_CRASH_WINDOWS:
        window = eq[(eq.index >= start) & (eq.index < end)]
        ret = (
            float("nan")
            if len(window) < 2
            else float(window.iloc[-1] / window.iloc[0] - 1.0)
        )
        rows.append(
            {
                "window": name,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "return_pct": ret,
                "fails_15pct": bool(ret == ret and ret < -0.15),
            }
        )
    return pd.DataFrame(rows)


def _year_key(trades: pd.DataFrame) -> pd.Series | None:
    if "entry_ts" not in trades.columns:
        return None
    return pd.to_datetime(trades["entry_ts"], utc=True).dt.year


def _quarter_key(trades: pd.DataFrame) -> pd.Series | None:
    if "entry_ts" not in trades.columns:
        return None
    ts = pd.to_datetime(trades["entry_ts"], utc=True)
    return ts.dt.to_period("Q").astype(str)


def _group_mean_r(trades: pd.DataFrame, key: pd.Series | None) -> pd.DataFrame:
    if trades.empty or key is None or "realised_r" not in trades.columns:
        return pd.DataFrame(columns=["group", "n", "mean_r", "sum_r"])
    r = pd.to_numeric(trades["realised_r"], errors="coerce")
    frame = pd.DataFrame({"group": key, "realised_r": r}).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["group", "n", "mean_r", "sum_r"])
    grouped = frame.groupby("group", observed=True)["realised_r"]
    return grouped.agg(n="count", mean_r="mean", sum_r="sum").reset_index()
