"""Performance metrics of a daily return series (spec §10)."""

import numpy as np

TRADING_DAYS = 252


def max_drawdown(nav: np.ndarray) -> tuple[float, int]:
    """Largest peak-to-trough loss (positive fraction) and longest underwater spell in days."""
    nav = np.asarray(nav, dtype=float)
    peak = np.maximum.accumulate(np.concatenate(([1.0], nav)))[1:]
    dd = 1.0 - nav / peak
    underwater = dd > 0
    longest = run = 0
    for u in underwater:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return float(dd.max(initial=0.0)), longest


def summary(returns: np.ndarray, cash_ret: np.ndarray | None = None,
            turnover: np.ndarray | None = None, costs: np.ndarray | None = None,
            exposure: np.ndarray | None = None) -> dict[str, float]:
    r = np.asarray(returns, dtype=float)
    rf = np.zeros_like(r) if cash_ret is None else np.asarray(cash_ret, dtype=float)
    years = len(r) / TRADING_DAYS
    nav = np.cumprod(1.0 + r)
    excess = r - rf
    downside = excess[excess < 0]
    mdd, mdd_days = max_drawdown(nav)
    sd = excess.std(ddof=1)
    cagr = nav[-1] ** (1 / years) - 1 if years > 0 else np.nan
    out = {
        "cagr": cagr,
        "total_return": nav[-1] - 1.0,
        "volatility": r.std(ddof=1) * np.sqrt(TRADING_DAYS),
        "sharpe": excess.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan,
        "sortino": (excess.mean() / np.sqrt((downside ** 2).sum() / len(excess))
                    * np.sqrt(TRADING_DAYS)) if len(downside) else np.inf,
        "max_drawdown": mdd,
        "max_drawdown_days": float(mdd_days),
        "calmar": cagr / mdd if mdd > 0 else np.inf,
        "cvar_5": -np.sort(r)[: max(1, int(0.05 * len(r)))].mean(),
    }
    if turnover is not None:
        out["turnover_annual"] = float(np.sum(turnover) / years)
    if costs is not None:
        out["costs_bps_annual"] = float(np.sum(costs) / years * 1e4)
    if exposure is not None:
        out["avg_exposure"] = float(np.mean(exposure))
        out["time_in_cash"] = float(np.mean(np.asarray(exposure) < 1e-9))
    return out
