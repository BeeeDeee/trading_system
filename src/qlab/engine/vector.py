"""Vector engine: target weights -> daily net returns (spec §7.1, §7.2).

Scale-invariant (NAV starts at 1), costs proportional to traded value. Timing contract:
- `targets[t]` is decided after the close of day t (NaN row = no decision that day),
- it is executed at the open of day t+1: sells first, then buys scaled down if cash is short,
- positions carry the overnight return with the old holdings and the intraday return with the new.
"""

from dataclasses import dataclass

import numpy as np

from qlab.data.panel import Panel

WEIGHT_TOL = 1e-9


@dataclass(frozen=True)
class SimResult:
    dates: np.ndarray
    nav: np.ndarray          # NAV at each close, NAV before day 0 = 1
    returns: np.ndarray      # daily net return
    turnover: np.ndarray     # traded value / NAV at the open, per day
    costs: np.ndarray        # costs / NAV at the open, per day
    exposure: np.ndarray     # invested fraction at the close
    n_positions: np.ndarray  # number of positions at the close


def check_targets(targets: np.ndarray, shape: tuple[int, int]) -> None:
    if targets.shape != shape:
        raise ValueError(f"targets shape {targets.shape} != panel shape {shape}")
    rows = targets[~np.isnan(targets).all(axis=1)]
    if np.isnan(rows).any():
        raise ValueError("a decision row must not mix NaN and numbers (use 0 for no position)")
    if (rows < -WEIGHT_TOL).any():
        raise ValueError("negative target weight (long-only)")
    if (rows.sum(axis=1) > 1 + WEIGHT_TOL).any():
        raise ValueError("target weights sum above 1 (no leverage)")


def simulate(panel: Panel, targets: np.ndarray, cost_rate: np.ndarray | float,
             cash_ret: np.ndarray | None = None) -> SimResult:
    """Simulate a long-only portfolio following `targets` (T x N weights, NaN rows = hold)."""
    n_days, n_assets = panel.shape
    check_targets(targets, panel.shape)
    cost_rate = np.broadcast_to(np.asarray(cost_rate, dtype=float), panel.shape)
    cash_ret = np.zeros(n_days) if cash_ret is None else np.asarray(cash_ret, dtype=float)
    decided = ~np.isnan(targets).all(axis=1)

    v = np.zeros(n_assets)  # position values
    cash = 1.0
    nav = np.empty(n_days)
    turnover = np.zeros(n_days)
    costs = np.zeros(n_days)
    exposure = np.empty(n_days)
    n_pos = np.empty(n_days, dtype=int)
    pending = None

    for t in range(n_days):
        v *= 1.0 + panel.ret_co[t]
        cash *= 1.0 + cash_ret[t]
        paid = panel.delisting[t]
        cash += v[paid].sum()
        v[paid] = 0.0

        if pending is not None:
            nav_open = cash + v.sum()
            delta = np.where(panel.tradable[t], pending * nav_open - v, 0.0)
            sells = np.minimum(delta, 0.0)
            sell_cost = (-sells * cost_rate[t]).sum()
            v += sells
            cash += -sells.sum() - sell_cost
            buys = np.maximum(delta, 0.0)
            need = (buys * (1.0 + cost_rate[t])).sum()
            if need > cash:
                buys *= max(cash, 0.0) / need
            buy_cost = (buys * cost_rate[t]).sum()
            v += buys
            cash -= buys.sum() + buy_cost
            turnover[t] = (buys.sum() - sells.sum()) / nav_open
            costs[t] = (sell_cost + buy_cost) / nav_open
            pending = None

        v *= 1.0 + panel.ret_oc[t]
        nav[t] = cash + v.sum()
        exposure[t] = v.sum() / nav[t]
        n_pos[t] = int((v > 0).sum())
        if decided[t]:
            pending = targets[t]

    returns = np.diff(nav, prepend=1.0) / np.concatenate(([1.0], nav[:-1]))
    return SimResult(panel.dates, nav, returns, turnover, costs, exposure, n_pos)
