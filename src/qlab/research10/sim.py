"""Signed-weight simulator and sleeve mixing of research 10 (prereg §3).

Same timing as `qlab.engine.vector`: a decision after the close of day t is filled at the open of t+1;
positions carry the overnight return with the old holdings and the intraday return with the new.
Shorts: value v < 0 earns -r. Interest (`cash_ret`) is paid only on free equity, cash + sum of short
values, so short-sale proceeds earn nothing. `short_carry[t, j]` is credited on the short value held
over day t, valued after the open of t (funding received by a perp short over day t, or minus a borrow fee). A NaN target keeps the position.
"""

from dataclasses import dataclass

import numpy as np

from qlab.schedule import period_starts

WEIGHT_TOL = 1e-9


@dataclass(frozen=True)
class SignedResult:
    nav: np.ndarray
    returns: np.ndarray
    turnover: np.ndarray     # traded value / NAV at the open
    costs: np.ndarray        # costs / NAV at the open
    net: np.ndarray          # sum of position values / NAV at the close
    gross: np.ndarray        # sum of |position values| / NAV at the close


def simulate_signed(ret_co: np.ndarray, ret_oc: np.ndarray, days: np.ndarray, weights: np.ndarray,
                    cost_rate: np.ndarray | float, cash_ret: np.ndarray | None = None,
                    short_carry: np.ndarray | None = None, tradable: np.ndarray | None = None) -> SignedResult:
    T, N = ret_co.shape
    days, weights = np.asarray(days, dtype=int), np.asarray(weights, dtype=float)
    if weights.shape != (len(days), N):
        raise ValueError(f"weights shape {weights.shape} != ({len(days)}, {N})")
    if len(days) and (np.any(np.diff(days) <= 0) or days[0] < 0 or days[-1] >= T):
        raise ValueError("decision days must be strictly increasing and inside the panel")
    if (np.nansum(np.abs(weights), axis=1) > 1 + WEIGHT_TOL).any():
        raise ValueError("gross target weights above 1 (no leverage)")
    cost_rate = np.broadcast_to(np.asarray(cost_rate, dtype=float), (T, N))
    cash_ret = np.zeros(T) if cash_ret is None else np.asarray(cash_ret, dtype=float)
    short_carry = np.zeros((T, N)) if short_carry is None else np.nan_to_num(np.asarray(short_carry, dtype=float))
    tradable = np.ones((T, N), dtype=bool) if tradable is None else np.asarray(tradable)
    row_of = {int(d): k for k, d in enumerate(days)}

    v, cash = np.zeros(N), 1.0
    nav, turn, costs = np.empty(T), np.zeros(T), np.zeros(T)
    net, gross = np.empty(T), np.empty(T)
    pending = None
    for t in range(T):
        cash += (cash + np.minimum(v, 0.0).sum()) * cash_ret[t]
        v *= 1.0 + ret_co[t]
        if pending is not None:
            nav_open = cash + v.sum()
            target = np.where(np.isnan(pending), v, pending * nav_open)
            delta = np.where(tradable[t], target - v, 0.0)
            c = (np.abs(delta) * cost_rate[t]).sum()
            v += delta
            cash -= delta.sum() + c
            turn[t], costs[t] = np.abs(delta).sum() / nav_open, c / nav_open
            pending = None
        cash -= (short_carry[t] * np.minimum(v, 0.0)).sum()   # short held over day t
        v *= 1.0 + ret_oc[t]
        nav[t] = cash + v.sum()
        net[t], gross[t] = v.sum() / nav[t], np.abs(v).sum() / nav[t]
        if t in row_of:
            pending = weights[row_of[t]]
    returns = np.diff(nav, prepend=1.0) / np.concatenate(([1.0], nav[:-1]))
    return SignedResult(nav, returns, turn, costs, net, gross)


def monthly_mix(returns: np.ndarray, w: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Daily returns of a portfolio of sleeves (columns of `returns`) reset to weights `w` after the close of
    the first trading day of each month and drifting in between. `returns` (T, S), `w` (S,) or (P, S) for P
    portfolios at once -> (T,) or (T, P). NaN returns count as 0 (no position)."""
    r = np.nan_to_num(np.asarray(returns, dtype=float))
    reset = np.r_[False, period_starts(dates, "M")[:-1]]   # weights reset before day t's return
    reset[0] = True
    block = np.cumsum(reset) - 1
    g = np.empty_like(r)                                   # growth since the last reset, per sleeve
    for b in range(block[-1] + 1):
        k = np.flatnonzero(block == b)
        g[k] = np.cumprod(1 + r[k], axis=0)
    g_prev = np.vstack([np.ones((1, r.shape[1])), g[:-1]])
    g_prev[reset] = 1.0
    W = np.atleast_2d(w).T                                 # (S, P)
    out = (g @ W) / (g_prev @ W) - 1
    return out[:, 0] if np.ndim(w) == 1 else out
