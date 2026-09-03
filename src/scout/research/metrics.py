"""Backtest metrics from trades.csv and equity.csv. 12-RESEARCH_PROTOCOL.md §4."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.stats import kurtosis, skew  # type: ignore[import-untyped]

from scout.research.registry import development_trial_count
from scout.utils.stats import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_BOOTSTRAP_SEED,
    bootstrap_ci,
    deflated_sharpe,
)

BARS_PER_YEAR = 252.0
_DAYS_PER_YEAR = 365.25


def compute_metrics(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    *,
    registry_path: Path | None = None,
    bars_per_year: float = BARS_PER_YEAR,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    split: str = "DEVELOPMENT",
) -> dict[str, float]:
    """Every metric in 12-RESEARCH_PROTOCOL.md §4."""
    eq = _equity_series(equity)
    rets = eq.pct_change().dropna()
    rets_arr = np.asarray(rets.to_numpy(), dtype=np.float64)
    out: dict[str, float] = {}
    out.update(_return_risk(eq, rets_arr, bars_per_year))
    out.update(_trade_stats(trades))
    out.update(_activity_and_cost(trades, eq, out.get("n_trades", 0.0)))
    out.update(
        _significance(
            trades,
            rets_arr,
            registry_path=registry_path,
            split=split,
            bars_per_year=bars_per_year,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        )
    )
    return {k: float(v) for k, v in out.items()}


def _return_risk(
    eq: pd.Series,
    rets: NDArray[np.float64],
    bars_per_year: float,
) -> dict[str, float]:
    if eq.empty or float(eq.iloc[0]) == 0.0:
        return {
            "total_return_pct": float("nan"),
            "cagr_pct": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown_pct": float("nan"),
            "max_drawdown_duration_days": float("nan"),
            "calmar": float("nan"),
            "ulcer_index": float("nan"),
        }
    start = float(eq.iloc[0])
    end = float(eq.iloc[-1])
    total_return = end / start - 1.0
    years = _years(eq.index)
    cagr = (end / start) ** (1.0 / years) - 1.0 if years > 0.0 and start > 0.0 else float("nan")
    sharpe = _ann_sharpe(rets, bars_per_year)
    sortino = _ann_sortino(rets, bars_per_year)
    peak = eq.cummax()
    dd = (peak - eq) / peak.replace(0.0, float("nan"))
    dd = dd.fillna(0.0).clip(lower=0.0)
    max_dd = float(dd.max()) if len(dd) else 0.0
    duration = _max_underwater_days(dd)
    ulcer = float(math.sqrt(float((dd.to_numpy() ** 2).mean()))) if len(dd) else 0.0
    calmar = cagr / max_dd if max_dd > 0.0 and math.isfinite(cagr) else float("nan")
    return {
        "total_return_pct": total_return,
        "cagr_pct": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd,
        "max_drawdown_duration_days": duration,
        "calmar": calmar,
        "ulcer_index": ulcer,
    }


def _trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    r = _col(trades, "realised_r")
    n = float(r.size)
    if n == 0:
        nan = float("nan")
        return {
            "n_trades": 0.0,
            "win_rate": nan,
            "avg_win_r": nan,
            "avg_loss_r": nan,
            "mean_r": nan,
            "median_r": nan,
            "std_r": nan,
            "profit_factor": nan,
            "expectancy_r": nan,
            "payoff_ratio": nan,
            "max_consecutive_losses": 0.0,
            "avg_bars_held": nan,
            "mae_r_mean": nan,
            "mfe_r_mean": nan,
        }
    wins = r[r > 0.0]
    losses = r[r < 0.0]
    mean_r = float(r.mean())
    avg_win = float(wins.mean()) if wins.size else float("nan")
    avg_loss = float(losses.mean()) if losses.size else float("nan")
    sum_win = float(wins.sum()) if wins.size else 0.0
    sum_loss = float(losses.sum()) if losses.size else 0.0
    profit_factor = (
        sum_win / abs(sum_loss)
        if sum_loss != 0.0
        else (float("inf") if sum_win > 0.0 else float("nan"))
    )
    payoff = (
        avg_win / abs(avg_loss)
        if math.isfinite(avg_win) and math.isfinite(avg_loss) and avg_loss != 0.0
        else float("nan")
    )
    std_r = float(r.std(ddof=1)) if r.size > 1 else float("nan")
    bars = _col(trades, "bars_held")
    mae = _col(trades, "mae_r")
    mfe = _col(trades, "mfe_r")
    return {
        "n_trades": n,
        "win_rate": float((r > 0.0).mean()),
        "avg_win_r": avg_win,
        "avg_loss_r": avg_loss,
        "mean_r": mean_r,
        "median_r": float(np.median(r)),
        "std_r": std_r,
        "profit_factor": profit_factor,
        "expectancy_r": mean_r,
        "payoff_ratio": payoff,
        "max_consecutive_losses": float(_max_consecutive_losses(r)),
        "avg_bars_held": float(bars.mean()) if bars.size else float("nan"),
        "mae_r_mean": float(mae.mean()) if mae.size else float("nan"),
        "mfe_r_mean": float(mfe.mean()) if mfe.size else float("nan"),
    }


def _activity_and_cost(
    trades: pd.DataFrame,
    eq: pd.Series,
    n_trades: float,
) -> dict[str, float]:
    years = _years(eq.index) if not eq.empty else float("nan")
    trades_per_month = (
        n_trades / (years * 12.0) if years and years > 0.0 else float("nan")
    )
    exposure = _exposure_pct(trades, eq)
    turnover = _turnover_annual(trades, eq, years)
    fees = float(_col(trades, "fees_usd").sum()) if not trades.empty else 0.0
    funding = float(_col(trades, "funding_usd").sum()) if not trades.empty else 0.0
    borrow = float(_col(trades, "borrow_usd").sum()) if not trades.empty else 0.0
    gross = float(_col(trades, "gross_pnl_usd").sum()) if not trades.empty else 0.0
    costs = fees + funding + borrow
    cost_drag = costs / gross if gross > 0.0 else float("nan")
    return {
        "trades_per_month": trades_per_month,
        "exposure_pct": exposure,
        "turnover_annual": turnover,
        "total_fees_usd": fees,
        "total_funding_usd": funding,
        "cost_drag_pct": cost_drag,
    }


def _significance(
    trades: pd.DataFrame,
    rets: NDArray[np.float64],
    *,
    registry_path: Path | None,
    split: str,
    bars_per_year: float,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, float]:
    r = _col(trades, "realised_r")
    if r.size:
        mean_lo, mean_hi = bootstrap_ci(
            r.tolist(), n_iterations=bootstrap_iterations, seed=seed
        )
        p_mean = _p_value_mean_gt_zero(r, bootstrap_iterations, seed)
    else:
        mean_lo = mean_hi = p_mean = float("nan")
    if rets.size >= 2:

        def _stat(sample: NDArray[np.float64]) -> float:
            return _ann_sharpe(sample, bars_per_year)

        sharpe_lo, sharpe_hi = bootstrap_ci(
            rets.tolist(),
            statistic=_stat,
            n_iterations=bootstrap_iterations,
            seed=seed,
        )
    else:
        sharpe_lo = sharpe_hi = float("nan")
    n_dev = development_trial_count(registry_path)
    n_trials = n_dev + 1 if split == "DEVELOPMENT" else max(n_dev, 1)
    dsr = float("nan")
    per_bar = float("nan")
    if rets.size >= 2:
        std = float(np.std(rets, ddof=1))
        if std > 0.0:
            per_bar = float(rets.mean()) / std
            sk = float(skew(rets, bias=True))
            ku = float(kurtosis(rets, fisher=False, bias=True))
            dsr = deflated_sharpe(
                per_bar,
                n_trials=n_trials,
                n_obs=int(rets.size),
                skew=sk,
                kurtosis=ku,
            )
    return {
        "mean_r_ci_low": float(mean_lo),
        "mean_r_ci_high": float(mean_hi),
        "sharpe_ci_low": float(sharpe_lo),
        "sharpe_ci_high": float(sharpe_hi),
        "deflated_sharpe": dsr,
        "p_value_mean_r": p_mean,
        "n_trials": float(n_trials),
        "sharpe_per_bar": per_bar,
    }


def _p_value_mean_gt_zero(
    values: NDArray[np.float64], n_iterations: int, seed: int
) -> float:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_iterations, values.size), replace=True)
    return float((samples.mean(axis=1) <= 0.0).mean())


def _ann_sharpe(rets: NDArray[np.float64], bars_per_year: float) -> float:
    if rets.size < 2:
        return float("nan")
    std = float(np.std(rets, ddof=1))
    if std == 0.0 or not math.isfinite(std):
        return float("nan")
    return float(rets.mean()) / std * math.sqrt(bars_per_year)


def _ann_sortino(rets: NDArray[np.float64], bars_per_year: float) -> float:
    if rets.size < 1:
        return float("nan")
    downside = math.sqrt(float(np.mean(np.minimum(rets, 0.0) ** 2)))
    if downside == 0.0:
        return float("nan")
    return float(rets.mean()) / downside * math.sqrt(bars_per_year)


def _max_underwater_days(dd: pd.Series) -> float:
    if dd.empty:
        return 0.0
    under = dd.to_numpy() > 0.0
    best = 0.0
    i = 0
    index = dd.index
    n = len(under)
    while i < n:
        if under[i]:
            j = i
            while j < n and under[j]:
                j += 1
            span = (index[j - 1] - index[i]).days
            if span > best:
                best = float(span)
            i = j
        else:
            i += 1
    return best


def _max_consecutive_losses(r: NDArray[np.float64]) -> int:
    best = 0
    current = 0
    for value in r:
        if value < 0.0:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


def _exposure_pct(trades: pd.DataFrame, eq: pd.Series) -> float:
    if eq.empty:
        return float("nan")
    if trades.empty or "entry_ts" not in trades.columns:
        return 0.0
    entry = pd.to_datetime(trades["entry_ts"], utc=True)
    exit_ts = pd.to_datetime(trades["exit_ts"], utc=True)
    exposed = 0
    for ts in eq.index:
        hit = bool(((entry <= ts) & (exit_ts >= ts)).any())
        if hit:
            exposed += 1
    return exposed / float(len(eq))


def _turnover_annual(trades: pd.DataFrame, eq: pd.Series, years: float) -> float:
    if trades.empty or eq.empty or not years or years <= 0.0:
        return float("nan")
    qty = _col(trades, "qty")
    entry_px = _col(trades, "entry_price")
    exit_px = _col(trades, "exit_price")
    n = min(qty.size, entry_px.size, exit_px.size)
    if n == 0:
        return float("nan")
    notional = 0.5 * (
        np.abs(qty[:n] * entry_px[:n]) + np.abs(qty[:n] * exit_px[:n])
    ).sum()
    mean_eq = float(eq.mean())
    if mean_eq <= 0.0:
        return float("nan")
    return float(notional) / mean_eq / years


def _years(index: pd.Index) -> float:
    if len(index) < 2:
        return float("nan")
    delta = index[-1] - index[0]
    return float(delta.days) / _DAYS_PER_YEAR


def _equity_series(equity: pd.DataFrame) -> pd.Series:
    if equity is None or equity.empty:
        return pd.Series(dtype="float64", name="equity")
    frame = equity.copy()
    if "ts" in frame.columns:
        frame = frame.set_index("ts")
    if "equity" not in frame.columns:
        raise ValueError("equity frame must contain an 'equity' column")
    series = pd.to_numeric(frame["equity"], errors="coerce")
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True))
    series.name = "equity"
    return series.sort_index()


def _col(trades: pd.DataFrame, name: str) -> NDArray[np.float64]:
    if trades is None or trades.empty or name not in trades.columns:
        return np.asarray([], dtype=np.float64)
    return np.asarray(pd.to_numeric(trades[name], errors="coerce").to_numpy(), dtype=np.float64)
