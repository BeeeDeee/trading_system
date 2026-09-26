"""Vector engine: target weights -> daily net returns (spec §7.1, §7.2).

Scale-invariant (NAV starts at 1), costs proportional to traded value. Timing contract:
- `targets[t]` is decided after the close of day t (NaN row = no decision that day); targets can
  be a dense (T, N) matrix or `Decisions` (only the decision days, for large grids),
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


@dataclass(frozen=True)
class Decisions:
    """Sparse targets: `weights[k]` is decided after the close of day `days[k]`."""

    days: np.ndarray     # (D,) int, strictly increasing
    weights: np.ndarray  # (D, N)

    def row(self) -> dict[int, int]:
        return {int(d): k for k, d in enumerate(self.days)}


def as_decisions(targets: "np.ndarray | Decisions", shape: tuple[int, int]) -> Decisions:
    """Validate targets (dense with NaN rows, or Decisions) and return them as Decisions."""
    if isinstance(targets, Decisions):
        days, rows = np.asarray(targets.days), np.asarray(targets.weights, dtype=float)
        if rows.shape != (len(days), shape[1]):
            raise ValueError(f"decision weights shape {rows.shape} != ({len(days)}, {shape[1]})")
        if len(days) and (np.any(np.diff(days) <= 0) or days[0] < 0 or days[-1] >= shape[0]):
            raise ValueError("decision days must be strictly increasing and inside the panel")
    else:
        if targets.shape != shape:
            raise ValueError(f"targets shape {targets.shape} != panel shape {shape}")
        days = np.flatnonzero(~np.isnan(targets).all(axis=1))
        rows = targets[days]
    if np.isnan(rows).any():
        raise ValueError("a decision row must not mix NaN and numbers (use 0 for no position)")
    if (rows < -WEIGHT_TOL).any():
        raise ValueError("negative target weight (long-only)")
    if (rows.sum(axis=1) > 1 + WEIGHT_TOL).any():
        raise ValueError("target weights sum above 1 (no leverage)")
    return Decisions(days, rows)


def simulate(panel: Panel, targets: "np.ndarray | Decisions", cost_rate: np.ndarray | float,
             cash_ret: np.ndarray | None = None) -> SimResult:
    """Simulate a long-only portfolio following `targets` (T x N weights, NaN rows = hold)."""
    n_days, n_assets = panel.shape
    decisions = as_decisions(targets, panel.shape)
    row_of = decisions.row()
    cost_rate = np.broadcast_to(np.asarray(cost_rate, dtype=float), panel.shape)
    cash_ret = np.zeros(n_days) if cash_ret is None else np.asarray(cash_ret, dtype=float)

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
        if t in row_of:
            pending = decisions.weights[row_of[t]]

    returns = np.diff(nav, prepend=1.0) / np.concatenate(([1.0], nav[:-1]))
    return SimResult(panel.dates, nav, returns, turnover, costs, exposure, n_pos)
