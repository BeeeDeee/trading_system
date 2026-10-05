"""Signed multi-asset engine: target weights -> daily net returns. The lab's single execution model.

Same timing as `qlab.engine.vector`: a decision after the close of day t (row t of `targets`, NaN row = no
decision) is executed at the open of day t+1; positions carry the overnight return with the old holdings and
the intraday return with the new. Built on `qlab.research10.sim.simulate_signed` (shorts, gross <= 1, cash
interest only on free equity) and adds what that one lacks:
- delistings: a position is paid out to cash at the open of its DELISTED row, like the vector engine,
- per-instrument non-tradable days (closed markets on the union calendar, halts): no fill, NaN target
  element = keep the position,
- buys scaled down when cash is short (long-only books), so a book that is fully invested and pays costs
  cannot borrow.
Parity tests: long-only vs `qlab.engine.vector.simulate`, long/short vs `simulate_signed`.
"""

from dataclasses import dataclass

import numpy as np

WEIGHT_TOL = 1e-9


@dataclass(frozen=True)
class SimResult:
    returns: np.ndarray    # daily net return
    nav: np.ndarray
    turnover: np.ndarray   # traded value / NAV at the open
    costs: np.ndarray      # costs / NAV at the open
    net: np.ndarray        # sum of position values / NAV at the close
    gross: np.ndarray      # sum of |position values| / NAV at the close
    held: np.ndarray       # (T, N) position value / NAV at the close


def validate_targets(targets: np.ndarray, shape: tuple[int, int]) -> list[str]:
    errors = []
    if targets.shape != shape:
        return [f"targets shape {targets.shape} != {shape}"]
    rows = targets[~np.isnan(targets).all(axis=1)]
    if np.isinf(rows).any():
        errors.append("infinite weight")
    if (np.nansum(np.abs(rows), axis=1) > 1 + WEIGHT_TOL).any():
        errors.append("gross weight above 1 (no leverage, decision Q8)")
    return errors


def simulate(ret_co: np.ndarray, ret_oc: np.ndarray, targets: np.ndarray, cost_rate: np.ndarray | float,
             tradable: np.ndarray, delisting: np.ndarray, cash_ret: np.ndarray | None = None,
             short_carry: np.ndarray | None = None, funding: np.ndarray | None = None) -> SimResult:
    """`funding` (T, N): rate charged on the value of a position held over day t (perpetual funding):
    a long pays it, a short receives it (negative rates the other way round)."""
    T, N = ret_co.shape
    errors = validate_targets(targets, (T, N))
    if errors:
        raise ValueError("; ".join(errors))
    cost_rate = np.broadcast_to(np.asarray(cost_rate, dtype=float), (T, N))
    cash_ret = np.zeros(T) if cash_ret is None else np.asarray(cash_ret, dtype=float)
    short_carry = np.zeros((T, N)) if short_carry is None else np.nan_to_num(np.asarray(short_carry, dtype=float))
    decided = ~np.isnan(targets).all(axis=1)
    shorts = bool((np.nan_to_num(targets) < -WEIGHT_TOL).any())     # long-only books skip the short bookkeeping
    carry = shorts and bool(short_carry.any())
    funding = None if funding is None else np.nan_to_num(np.asarray(funding, dtype=float))
    if funding is not None and not funding.any():
        funding = None
    paid_rows = delisting.any(axis=1)

    v, cash = np.zeros(N), 1.0
    nav, turn, costs = np.empty(T), np.zeros(T), np.zeros(T)
    values = np.empty((T, N))
    pending = None
    for t in range(T):
        if cash_ret[t]:
            cash += (cash + (np.minimum(v, 0.0).sum() if shorts else 0.0)) * cash_ret[t]
        v *= 1.0 + ret_co[t]
        if paid_rows[t]:
            paid = delisting[t] & (v != 0)
            cash += v[paid].sum()
            v[paid] = 0.0
        if pending is not None:
            nav_open = cash + v.sum()
            target = np.where(np.isnan(pending), v, pending * nav_open)
            delta = np.where(tradable[t], target - v, 0.0)
            sells, buys = np.minimum(delta, 0.0), np.maximum(delta, 0.0)
            c_sell = (-sells * cost_rate[t]).sum()
            cash -= sells.sum() + c_sell
            if not (pending < -WEIGHT_TOL).any():   # long-only decision: never borrow (vector engine)
                need = (buys * (1.0 + cost_rate[t])).sum()
                if need > 0.0 and need > cash:
                    buys = buys * (max(cash, 0.0) / need)
            c_buy = (buys * cost_rate[t]).sum()
            cash -= buys.sum() + c_buy
            v += sells + buys
            turn[t] = (buys.sum() - sells.sum()) / nav_open
            costs[t] = (c_sell + c_buy) / nav_open
            pending = None
        if carry:
            cash -= (short_carry[t] * np.minimum(v, 0.0)).sum()
        if funding is not None:
            cash -= (funding[t] * v).sum()
        v *= 1.0 + ret_oc[t]
        values[t] = v
        nav[t] = cash + v.sum()
        if decided[t]:
            pending = targets[t]
    held = values / nav[:, None]
    net, gross = held.sum(axis=1), np.abs(held).sum(axis=1)
    returns = np.diff(nav, prepend=1.0) / np.concatenate(([1.0], nav[:-1]))
    return SimResult(returns, nav, turn, costs, net, gross, held)
