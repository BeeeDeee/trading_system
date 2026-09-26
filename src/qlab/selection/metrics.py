"""Column-wise metrics of candidate return matrices over a window (days x candidates)."""

import numpy as np

TRADING_DAYS = 252


def sharpe(R: np.ndarray, rf: np.ndarray) -> np.ndarray:
    """Annualized Sharpe of daily excess returns per column (0 where volatility is 0)."""
    ex = np.asarray(R, dtype=np.float64) - np.asarray(rf, dtype=np.float64)[:, None]
    sd = ex.std(axis=0, ddof=1)
    return np.where(sd > 0, ex.mean(axis=0) / np.where(sd > 0, sd, 1.0), 0.0) * np.sqrt(TRADING_DAYS)


def cagr(R: np.ndarray) -> np.ndarray:
    growth = np.exp(np.log1p(np.asarray(R, dtype=np.float64)).sum(axis=0))
    return growth ** (TRADING_DAYS / R.shape[0]) - 1.0


def max_drawdown(R: np.ndarray) -> np.ndarray:
    nav = np.exp(np.cumsum(np.log1p(np.asarray(R, dtype=np.float64)), axis=0))
    peak = np.maximum(np.maximum.accumulate(nav, axis=0), 1.0)
    return (1.0 - nav / peak).max(axis=0)


def share_positive_years(R: np.ndarray, rf: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Share of calendar years with a positive excess return (partial years count)."""
    years = np.asarray(dates, dtype="datetime64[Y]")
    ex = np.log1p(np.asarray(R, dtype=np.float64)) - np.log1p(np.asarray(rf))[:, None]
    uniq = np.unique(years)
    by_year = np.array([ex[years == y].sum(axis=0) for y in uniq])
    return (by_year > 0).mean(axis=0)


def window_summary(R, rf, dates, turnover=None, n_pos=None, exposure=None) -> dict[str, np.ndarray]:
    years = R.shape[0] / TRADING_DAYS
    out = {"sharpe": sharpe(R, rf), "cagr": cagr(R), "max_drawdown": max_drawdown(R),
           "share_positive_years": share_positive_years(R, rf, dates)}
    if turnover is not None:
        out["turnover_annual"] = np.asarray(turnover, dtype=np.float64).sum(axis=0) / years
    if n_pos is not None:
        out["avg_positions"] = np.asarray(n_pos, dtype=np.float64).mean(axis=0)
    if exposure is not None:
        out["avg_exposure"] = np.asarray(exposure, dtype=np.float64).mean(axis=0)
    return out
