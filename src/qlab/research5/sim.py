"""Event-driven STR-TF simulator (research 5 pre-registration §5).

Unlike the target-weight engines of research 1, positions are opened with a size fixed at entry and
held without rebalancing until an exit event. Timing contract (all in NAV units, NAV before the
first day = 1):

- candidates of day t are decided after the close of t (ranked, best first),
- entry_delay 1: bought at the open of t+1; 2: open of t+2; 0: at the close of t (MOC upper bound),
- after each close, every position counts one more day held; the exit signal, the time stop and
  the optional stop are checked on that close and executed at the next open where the security is
  tradable,
- a delisting row pays the terminal value into cash at that open without costs,
- positions still open on the last day are marked at the close (reason "end").
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from qlab.data.panel import Panel

REASONS = ("exit_signal", "time_stop", "stop", "delisted", "end")


@dataclass(frozen=True)
class SimSpec:
    max_positions: int
    max_hold: int
    entry_delay: int = 1
    capital: float = 1e6      # USD at the start; converts the ADV cap into NAV units
    adv_cap: float = 0.01     # max position value as a fraction of ADV20 at the signal day
    vol_target: float | None = None
    vol_window: int = 63


@dataclass(frozen=True)
class Candidates:
    """Sparse entry candidates sorted by (day, rank). Optional per-candidate overrides."""
    t: np.ndarray
    a: np.ndarray
    score: np.ndarray
    hold: np.ndarray | None = None   # per-candidate max hold (random benchmark)
    stop: np.ndarray | None = None   # per-candidate stop distance as a fraction (ATR variant)
    max_entries: np.ndarray | None = None  # (T,) cap of new entries per decision day

    def bounds(self, n_days: int) -> np.ndarray:
        return np.searchsorted(self.t, np.arange(n_days + 1))


@dataclass
class SimOutput:
    dates: np.ndarray
    returns: np.ndarray     # daily net returns on [start, end)
    nav: np.ndarray
    exposure: np.ndarray
    n_positions: np.ndarray
    costs: np.ndarray       # costs / NAV per day
    turnover: np.ndarray    # traded value / NAV per day
    trades: pl.DataFrame


class _Pos:
    __slots__ = ("value", "entry_value", "entry_cost", "entry_day", "signal_day", "held",
                 "max_hold", "stop", "score", "adv")

    def __init__(self, value, cost, entry_day, signal_day, max_hold, stop, score, adv):
        self.value, self.entry_value, self.entry_cost = value, value, cost
        self.entry_day, self.signal_day, self.held = entry_day, signal_day, 0
        self.max_hold, self.stop, self.score, self.adv = max_hold, stop, score, adv


CostFn = Callable[[int, int], float]


def simulate(panel: Panel, cands: Candidates, spec: SimSpec, adv20: np.ndarray, start: int,
             end: int, entry_cost: CostFn, exit_cost: CostFn,
             exit_signal: np.ndarray | None = None, allow_entry: np.ndarray | None = None,
             cash_ret: np.ndarray | None = None) -> SimOutput:
    """Simulate days [start, end). Decisions from the close of `start`; nothing is held before.

    `exit_signal` is a (T, N) array whose value > 0 on the close of t means "exit" (e.g. the TR
    index above its short SMA); None disables the signal exit. `allow_entry` (T,) bool gates new
    decisions (market filter).
    """
    if spec.entry_delay not in (0, 1, 2):
        raise ValueError("entry_delay must be 0, 1 or 2")
    n = end - start
    bounds = cands.bounds(panel.shape[0])
    pos: dict[int, _Pos] = {}
    pending_exit: dict[int, str] = {}
    pending_buy: dict[int, int] = {}   # execution day -> decision day
    cash = 1.0
    nav = np.empty(n)
    exposure = np.empty(n)
    n_pos = np.empty(n, dtype=int)
    costs = np.zeros(n)
    turnover = np.zeros(n)
    rets: list[float] = []
    trades: list[tuple] = []

    def close_trade(a: int, p: _Pos, t: int, reason: str, cost: float) -> None:
        gross = p.value / p.entry_value - 1.0
        net = (p.value - cost) / (p.entry_value + p.entry_cost) - 1.0
        trades.append((a, p.signal_day, p.entry_day, t, reason, p.entry_value, p.entry_cost,
                       p.value, cost, gross, net, p.held, p.score, p.adv))

    def buy(t: int, sig: int, fill_at_close: bool) -> None:
        nonlocal cash
        lo, hi = bounds[sig], bounds[sig + 1]
        if lo == hi:
            return
        nav_now = cash + sum(p.value for p in pos.values())
        size = nav_now / spec.max_positions
        if spec.vol_target is not None and len(rets) >= spec.vol_window:
            vol = np.std(rets[-spec.vol_window:], ddof=1) * np.sqrt(252)
            size *= min(1.0, spec.vol_target / vol) if vol > 0 else 1.0
        cap = np.inf if cands.max_entries is None else cands.max_entries[sig]
        free, done = spec.max_positions - len(pos), 0
        tradable = panel.tradable[t]
        for k in range(lo, hi):
            if free <= 0 or done >= cap:
                break
            a = int(cands.a[k])
            if a in pos or not tradable[a]:
                continue
            adv = float(adv20[sig, a])
            v = min(size, spec.adv_cap * adv / spec.capital) if np.isfinite(adv) else size
            rate = entry_cost(t, a)
            v = min(v, max(cash, 0.0) / (1.0 + rate))
            if v <= 1e-9:
                break
            cash -= v * (1.0 + rate)
            costs[t - start] += v * rate / nav_now
            turnover[t - start] += v / nav_now
            hold = spec.max_hold if cands.hold is None else int(cands.hold[k])
            stop = None if cands.stop is None else float(cands.stop[k])
            pos[a] = _Pos(v, v * rate, t, sig, hold, stop, float(cands.score[k]), adv)
            free, done = free - 1, done + 1

    for t in range(start, end):
        i = t - start
        ret_co, ret_oc = panel.ret_co[t], panel.ret_oc[t]
        nav_prev = cash + sum(p.value for p in pos.values())
        for p_a, p in pos.items():
            p.value *= 1.0 + ret_co[p_a]
        if cash_ret is not None:
            cash *= 1.0 + cash_ret[t]

        delisting = panel.delisting[t]
        for a in [a for a in pos if delisting[a]]:
            p = pos.pop(a)
            pending_exit.pop(a, None)
            cash += p.value
            close_trade(a, p, t, "delisted", 0.0)

        if pending_exit:
            nav_open = cash + sum(p.value for p in pos.values())
            tradable = panel.tradable[t]
            for a in [a for a in pending_exit if tradable[a]]:
                reason = pending_exit.pop(a)
                p = pos.pop(a)
                c = p.value * exit_cost(t, a)
                cash += p.value - c
                costs[i] += c / nav_open
                turnover[i] += p.value / nav_open
                close_trade(a, p, t, reason, c)

        if t in pending_buy:
            buy(t, pending_buy.pop(t), fill_at_close=False)

        for p_a, p in pos.items():
            p.value *= 1.0 + ret_oc[p_a]
            p.held += 1

        ex = None if exit_signal is None else exit_signal[t]
        for a, p in pos.items():
            if a in pending_exit:
                continue
            if ex is not None and ex[a] > 0:
                pending_exit[a] = "exit_signal"
            elif p.held >= p.max_hold:
                pending_exit[a] = "time_stop"
            elif p.stop is not None and p.value / p.entry_value < 1.0 - p.stop:
                pending_exit[a] = "stop"

        if allow_entry is None or allow_entry[t]:
            if spec.entry_delay == 0:
                buy(t, t, fill_at_close=True)
            elif t + spec.entry_delay < end:
                pending_buy[t + spec.entry_delay] = t

        invested = sum(p.value for p in pos.values())
        nav[i] = cash + invested
        exposure[i] = invested / nav[i] if nav[i] > 0 else 0.0
        n_pos[i] = len(pos)
        rets.append(nav[i] / nav_prev - 1.0 if nav_prev > 0 else 0.0)

    for a, p in pos.items():
        close_trade(a, p, end - 1, "end", 0.0)
    cols = list(zip(*trades)) if trades else [[] for _ in range(14)]
    trade_df = pl.DataFrame({
        "asset": pl.Series(cols[0], dtype=pl.Int64), "signal_day": pl.Series(cols[1], dtype=pl.Int64),
        "entry_day": pl.Series(cols[2], dtype=pl.Int64), "exit_day": pl.Series(cols[3], dtype=pl.Int64),
        "reason": pl.Series(cols[4], dtype=pl.Utf8), "entry_value": pl.Series(cols[5], dtype=pl.Float64),
        "entry_cost": pl.Series(cols[6], dtype=pl.Float64), "exit_value": pl.Series(cols[7], dtype=pl.Float64),
        "exit_cost": pl.Series(cols[8], dtype=pl.Float64), "gross_ret": pl.Series(cols[9], dtype=pl.Float64),
        "net_ret": pl.Series(cols[10], dtype=pl.Float64), "held": pl.Series(cols[11], dtype=pl.Int64),
        "score": pl.Series(cols[12], dtype=pl.Float64), "adv20": pl.Series(cols[13], dtype=pl.Float64),
    }).sort("entry_day", "asset")
    return SimOutput(panel.dates[start:end], np.asarray(rets), nav, exposure, n_pos, costs,
                     turnover, trade_df)
