"""Metrics of research 8 (prereg §6): excess over T-bill, annualized on a 365-day calendar."""

import numpy as np

from qlab.validation.stats import bootstrap_ci, sharpe

ANN = 365


def max_dd(ret: np.ndarray) -> float:
    nav = np.cumprod(1 + ret)
    return float((nav / np.maximum.accumulate(nav) - 1).min()) if len(nav) else 0.0


def summary(ret: np.ndarray, tbill: np.ndarray, funding: np.ndarray | None = None,
            costs: np.ndarray | None = None, turnover: np.ndarray | None = None) -> dict:
    ex = ret - tbill
    n = len(ret)
    out = {
        "days": n,
        "cagr": float(np.prod(1 + ret) ** (ANN / n) - 1) if n else None,
        "excess_ann": float(ex.mean() * ANN),
        "sharpe_excess": float(sharpe(ex) * np.sqrt(ANN)),
        "vol_ann": float(ret.std(ddof=1) * np.sqrt(ANN)),
        "max_dd": max_dd(ret),
        "tbill_ann": float(tbill.mean() * ANN),
    }
    if funding is not None:
        out["funding_ann"] = float(funding.mean() * ANN)
    if costs is not None:
        out["costs_ann"] = float(costs.mean() * ANN)
    if turnover is not None:
        out["turnover_ann"] = float(turnover.mean() * ANN)
    return out


def excess_ci(ret: np.ndarray, tbill: np.ndarray) -> tuple[float, float, float]:
    """Annualized mean excess and its 90 % stationary-bootstrap interval (block 21, 1000, seed 0)."""
    ex = (ret - tbill) * ANN
    return bootstrap_ci(ex, lambda x: float(x.mean()), n_boot=1000, mean_block=21.0, alpha=0.10, seed=0)
