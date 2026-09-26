"""Benchmarks (spec §8).

- EW_UNIV: equal weight of the universe, rebalanced after the first trading day of each month, run
  through the engine with the same cost model as strategies.
- BH_UNIV: buy-and-hold of the universe members at the start, equal initial weights, never
  rebalanced; delisting proceeds are reinvested pro rata into the remaining positions.
- SPY_TR and CASH come straight from data (see scripts).
"""

import numpy as np

from qlab.data.panel import Panel


def equal_weight_targets(members: np.ndarray, decision_days: np.ndarray) -> np.ndarray:
    """(T, N) targets: 1/n over `members` on decision days, NaN rows otherwise."""
    targets = np.full(members.shape, np.nan)
    rows = np.flatnonzero(decision_days)
    counts = members[rows].sum(axis=1, keepdims=True)
    targets[rows] = np.where(counts > 0, members[rows] / np.maximum(counts, 1), 0.0)
    return targets


def buy_and_hold_returns(panel: Panel, members: np.ndarray, start: int,
                         entry_cost: float = 0.0) -> np.ndarray:
    """Daily returns from day `start` on; bought at the open of `start`, equal weights.

    Days before `start` return 0. Each day the portfolio earns the value-weighted return of its
    positions; a delisting's terminal payout is spread over the remaining positions pro rata
    (which is what reinvesting proportionally means). If everything delists, it holds cash at 0.
    """
    n_days = panel.shape[0]
    held = members.astype(bool) & panel.tradable[start]
    v = np.where(held, 1.0 / max(held.sum(), 1) / (1.0 + entry_cost), 0.0)
    out = np.zeros(n_days)
    out[start] = (v * (1.0 + np.asarray(panel.ret_oc[start]))).sum() - 1.0
    v = v * (1.0 + np.asarray(panel.ret_oc[start]))
    for t in range(start + 1, n_days):
        before = v.sum()
        if before <= 0:
            break
        v = v * (1.0 + np.asarray(panel.ret_co[t]))
        paid = np.asarray(panel.delisting[t]) & (v > 0)
        if paid.any():
            payout = v[paid].sum()
            v[paid] = 0.0
            rest = v.sum()
            v = v * (1.0 + payout / rest) if rest > 0 else v
            if rest <= 0:
                out[t] = payout / before - 1.0
                break
        v = v * (1.0 + np.asarray(panel.ret_oc[t]))
        out[t] = v.sum() / before - 1.0
    return out
