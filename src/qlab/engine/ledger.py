"""Ledger engine: units, cash in USD, orders and fills (spec §7.2).

Deliberately written independently of the vector engine (per-order loop, dict of positions) so that
the parity test between the two is a meaningful check of both.

Positions are held in total-return units: the unit price is a total-return index per asset, i.e.
dividends and spinoffs are reinvested in the same security (see docs/PHASE0_PLAN.md). Values, fees
and cash are real USD.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from qlab.data.panel import Panel
from qlab.engine.vector import SimResult, check_targets


@dataclass(frozen=True)
class LedgerConfig:
    capital: float = 10_000.0
    fee_per_order: float = 0.0     # fixed USD fee per filled order
    min_order_value: float = 0.0   # skip orders smaller than this (USD), except full exits
    no_trade_band: float = 0.0     # skip rebalancing when |target - current weight| is below this


@dataclass(frozen=True)
class LedgerResult:
    sim: SimResult          # nav is in USD
    orders: pl.DataFrame    # date, asset, side, units, price, value, cost


def _unit_prices(panel: Panel) -> tuple[np.ndarray, np.ndarray]:
    """Total-return unit price at each open and close, starting at 1 for every asset."""
    growth_co = 1.0 + panel.ret_co
    growth_oc = 1.0 + panel.ret_oc
    close = np.cumprod(growth_co * growth_oc, axis=0)
    prev_close = np.vstack([np.ones((1, panel.shape[1])), close[:-1]])
    return prev_close * growth_co, close


def simulate_ledger(panel: Panel, targets: np.ndarray, cost_rate: np.ndarray | float,
                    cash_ret: np.ndarray | None = None,
                    config: LedgerConfig = LedgerConfig()) -> LedgerResult:
    n_days, n_assets = panel.shape
    check_targets(targets, panel.shape)
    cost_rate = np.broadcast_to(np.asarray(cost_rate, dtype=float), panel.shape)
    cash_ret = np.zeros(n_days) if cash_ret is None else np.asarray(cash_ret, dtype=float)
    p_open, p_close = _unit_prices(panel)

    units: dict[int, float] = {}
    cash = config.capital
    nav = np.empty(n_days)
    turnover = np.zeros(n_days)
    costs = np.zeros(n_days)
    exposure = np.empty(n_days)
    n_pos = np.empty(n_days, dtype=int)
    orders: list[tuple] = []
    pending: np.ndarray | None = None

    for t in range(n_days):
        cash *= 1.0 + cash_ret[t]
        for a in [a for a in units if panel.delisting[t, a]]:
            payout = units.pop(a) * p_open[t, a]
            cash += payout
            orders.append((t, a, "delisting", 0.0, p_open[t, a], payout, 0.0))

        if pending is not None:
            nav_open = cash + sum(u * p_open[t, a] for a, u in units.items())
            planned = []
            for a in range(n_assets):
                held = units.get(a, 0.0) * p_open[t, a]
                if not panel.tradable[t, a]:
                    continue
                target = pending[a] * nav_open
                diff = target - held
                if diff == 0.0:
                    continue
                full_exit = target == 0.0 and held > 0.0
                if not full_exit and (abs(diff) / nav_open < config.no_trade_band
                                      or abs(diff) < config.min_order_value):
                    continue
                planned.append((a, diff, full_exit))

            traded = fees = 0.0
            for a, diff, full_exit in [o for o in planned if o[1] < 0]:
                value = -diff
                fee = min(value * cost_rate[t, a] + config.fee_per_order, value)
                units[a] = 0.0 if full_exit else units[a] - value / p_open[t, a]
                if units[a] <= 0.0:
                    del units[a]
                cash += value - fee
                traded, fees = traded + value, fees + fee
                orders.append((t, a, "sell", value / p_open[t, a], p_open[t, a], value, fee))

            # Fixed fees do not scale with the order: drop the smallest buys until the rest can
            # pay their fees, then scale the remaining buys to the cash left.
            buys = sorted((o for o in planned if o[1] > 0), key=lambda o: -o[1])
            while buys and cash - len(buys) * config.fee_per_order <= 0.0:
                buys.pop()
            need = sum(d * (1.0 + cost_rate[t, a]) for a, d, _ in buys)
            budget = cash - len(buys) * config.fee_per_order
            scale = min(1.0, budget / need) if need > 0 else 1.0
            for a, diff, _ in buys:
                value = diff * scale
                fee = value * cost_rate[t, a] + config.fee_per_order
                units[a] = units.get(a, 0.0) + value / p_open[t, a]
                cash -= value + fee
                traded, fees = traded + value, fees + fee
                orders.append((t, a, "buy", value / p_open[t, a], p_open[t, a], value, fee))

            turnover[t] = traded / nav_open
            costs[t] = fees / nav_open
            pending = None

        invested = sum(u * p_close[t, a] for a, u in units.items())
        nav[t] = cash + invested
        exposure[t] = invested / nav[t]
        n_pos[t] = len(units)
        if not np.isnan(targets[t]).all():
            pending = targets[t]

    prev = np.concatenate(([config.capital], nav[:-1]))
    sim = SimResult(panel.dates, nav, nav / prev - 1.0, turnover, costs, exposure, n_pos)
    cols = list(zip(*orders)) if orders else [[] for _ in range(7)]
    t_idx, a_idx = np.asarray(cols[0], dtype=int), np.asarray(cols[1], dtype=int)
    order_df = pl.DataFrame({
        "date": pl.Series(panel.dates[t_idx], dtype=pl.Date),
        "asset": pl.Series(panel.assets[a_idx], dtype=pl.Int64),
        "side": pl.Series(cols[2], dtype=pl.Utf8),
        "units": pl.Series(cols[3], dtype=pl.Float64),
        "price": pl.Series(cols[4], dtype=pl.Float64),
        "value": pl.Series(cols[5], dtype=pl.Float64),
        "cost": pl.Series(cols[6], dtype=pl.Float64),
    })
    return LedgerResult(sim, order_df)
