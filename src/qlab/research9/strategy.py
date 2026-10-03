"""Signals, candidates and the trend-sleeve simulator of research 9 (prereg §4).

Point in time: a decision after the close of day t uses rows <= t only and is filled at the open of t+1.
"""

from dataclasses import asdict, dataclass

import numpy as np

from qlab.data.panel import Panel
from qlab.engine.vector import Decisions, SimResult, simulate

UNIVERSE_N = 20
MIN_TRADED_DAYS = 60
MARKET_SMA = 100


@dataclass(frozen=True)
class Config:
    family: str               # T | X
    rule: str = ""            # T: sma | mom
    n: int = 0                # T: SMA length or momentum lookback
    K: int = 0                # X
    L: int = 0                # X
    market_filter: bool = False
    universe_n: int = UNIVERSE_N

    @property
    def id(self) -> str:
        if self.family == "T":
            return f"T_{self.rule}{self.n}"
        return f"X_K{self.K}_L{self.L}" + ("_mf" if self.market_filter else "")

    def as_dict(self) -> dict:
        return {**asdict(self), "study": "research9"}


def grid() -> list[Config]:
    out = [Config("T", "sma", n) for n in (20, 50, 100, 200)] + [Config("T", "mom", n) for n in (7, 14, 30, 90)]
    out += [Config("X", K=K, L=L, market_filter=mf) for K in (3, 5, 10) for L in (7, 30, 90) for mf in (False, True)]
    return out


def sma(close: np.ndarray, n: int) -> np.ndarray:
    """Trailing n-day mean of closes including day t; NaN until n valid closes in a row."""
    c = np.asarray(close, dtype=float)
    out = np.full_like(c, np.nan)
    cs = np.cumsum(np.nan_to_num(c), axis=0)
    ok = np.cumsum(~np.isnan(c), axis=0)
    for t in range(n - 1, len(c)):
        lo = t - n
        s = cs[t] - (cs[lo] if lo >= 0 else 0)
        k = ok[t] - (ok[lo] if lo >= 0 else 0)
        out[t] = np.where(k == n, s / n, np.nan)
    return out


def trailing_return(close: np.ndarray, L: int) -> np.ndarray:
    out = np.full_like(close, np.nan, dtype=float)
    out[L:] = close[L:] / close[:-L] - 1
    return out


def trend_signal(close: np.ndarray, cfg: Config) -> np.ndarray:
    """1 = hold, 0 = cash, NaN = not enough history (treated as cash)."""
    ref = sma(close, cfg.n) if cfg.rule == "sma" else None
    raw = close > ref if cfg.rule == "sma" else trailing_return(close, cfg.n) > 0
    known = ~np.isnan(ref) if cfg.rule == "sma" else ~np.isnan(trailing_return(close, cfg.n))
    return np.where(known, raw.astype(float), 0.0)


def universe(p: Panel, qv: np.ndarray, t: int, n: int) -> np.ndarray:
    """Top-n pairs by 30-day mean quote volume among pairs traded on each of the last 60 days, at close t."""
    if t + 1 < MIN_TRADED_DAYS:
        return np.array([], dtype=int)
    ok = p.tradable[t + 1 - MIN_TRADED_DAYS:t + 1].all(axis=0)
    v = np.where(ok, np.nanmean(qv[max(0, t - 29):t + 1], axis=0), -np.inf)
    order = np.argsort(-v, kind="stable")
    return np.array([j for j in order[:n] if ok[j]], dtype=int)


def sundays(p: Panel) -> np.ndarray:
    wd = (p.dates.astype("datetime64[D]").view("int64") - 4) % 7       # 0 = Monday
    return np.flatnonzero(wd == 6)


def x_decisions(p: Panel, qv: np.ndarray, cfg: Config, btc: int, first: int = 0) -> Decisions:
    """Weekly targets decided after the Sunday close, filled Monday open."""
    btc_sma = sma(p.close_u[:, btc], MARKET_SMA)
    days, rows = [], []
    for t in sundays(p):
        if t < first or t + 1 >= len(p.dates):
            continue
        w = np.zeros(p.shape[1])
        risk_off = cfg.market_filter and not (p.close_u[t, btc] > btc_sma[t])
        if not risk_off and t >= cfg.L:
            U = universe(p, qv, t, cfg.universe_n)
            r = p.close_u[t, U] / p.close_u[t - cfg.L, U] - 1
            pick = [j for _, j in sorted(((-x, j) for x, j in zip(r, U) if np.isfinite(x) and x > 0))][:cfg.K]
            w[pick] = 1.0 / cfg.K
        days.append(t)
        rows.append(w)
    return Decisions(np.array(days, dtype=int), np.array(rows).reshape(len(days), p.shape[1]))


def ew_universe_decisions(p: Panel, qv: np.ndarray, n: int, first: int = 0) -> Decisions:
    days, rows = [], []
    for t in sundays(p):
        if t < first or t + 1 >= len(p.dates):
            continue
        w = np.zeros(p.shape[1])
        U = universe(p, qv, t, n)
        if len(U):
            w[U] = 1.0 / len(U)
        days.append(t)
        rows.append(w)
    return Decisions(np.array(days, dtype=int), np.array(rows).reshape(len(days), p.shape[1]))


def monthly_5050(p: Panel, a: int, b: int, first: int = 0) -> Decisions:
    """50/50 rebalanced at the open of the first day of each month (decided the day before)."""
    month = p.dates.astype("datetime64[M]")
    days = [t for t in range(first, len(p.dates) - 1) if month[t + 1] != month[t] or t == first]
    rows = np.zeros((len(days), p.shape[1]))
    rows[:, [a, b]] = 0.5
    return Decisions(np.array(days, dtype=int), rows)


def hold(p: Panel, j: int, first: int = 0) -> Decisions:
    """Buy asset j with all capital after the close of day `first` and hold."""
    w = np.zeros((1, p.shape[1]))
    w[0, j] = 1.0
    return Decisions(np.array([first]), w)


def simulate_trend(p: Panel, signals: dict[int, np.ndarray], cost: np.ndarray, first: int,
                   cash_ret: np.ndarray | None = None) -> SimResult:
    """Each coin trades only when its own signal changes; an entering coin buys min(1/len(signals) of NAV,
    cash). Same open/close accounting and costs as qlab.engine.vector."""
    T = len(p.dates)
    cash_ret = np.zeros(T) if cash_ret is None else cash_ret
    share = 1.0 / len(signals)
    v = {j: 0.0 for j in signals}
    cash = 1.0
    nav, turn, costs, expo = np.empty(T), np.zeros(T), np.zeros(T), np.empty(T)
    npos = np.zeros(T, dtype=int)
    pending: dict[int, float] | None = None
    for t in range(T):
        for j in v:
            v[j] *= 1 + p.ret_co[t, j]
            if p.delisting[t, j]:
                cash += v[j]
                v[j] = 0.0
        cash *= 1 + cash_ret[t]
        if pending:
            nav_open = cash + sum(v.values())
            for j, on in sorted(pending.items()):
                if not p.tradable[t, j]:
                    continue
                if on == 0.0 and v[j] > 0:
                    c = v[j] * cost[t, j]
                    cash += v[j] - c
                    turn[t] += v[j] / nav_open
                    costs[t] += c / nav_open
                    v[j] = 0.0
            for j, on in sorted(pending.items()):
                if not p.tradable[t, j] or on == 0.0 or v[j] > 0:
                    continue
                buy = min(share * nav_open, max(cash, 0.0) / (1 + cost[t, j]))
                c = buy * cost[t, j]
                v[j] += buy
                cash -= buy + c
                turn[t] += buy / nav_open
                costs[t] += c / nav_open
            pending = None
        for j in v:
            v[j] *= 1 + p.ret_oc[t, j]
        nav[t] = cash + sum(v.values())
        expo[t] = sum(v.values()) / nav[t]
        npos[t] = sum(1 for x in v.values() if x > 0)
        if t >= first and t + 1 < T:
            prev = {j: (s[t - 1] if t > first else -1.0) for j, s in signals.items()}
            changed = {j: float(s[t]) for j, s in signals.items() if s[t] != prev[j]}
            if changed:
                pending = changed
    ret = np.diff(nav, prepend=1.0) / np.concatenate(([1.0], nav[:-1]))
    return SimResult(p.dates, nav, ret, turn, costs, expo, npos)


def run(p: Panel, qv: np.ndarray, cfg: Config, cost: np.ndarray, first: int, btc: int, eth: int,
        cash_ret: np.ndarray | None = None) -> SimResult:
    if cfg.family == "T":
        sig = {j: trend_signal(p.close_u[:, j], cfg) for j in (btc, eth)}
        return simulate_trend(p, sig, cost, first, cash_ret)
    return simulate(p, x_decisions(p, qv, cfg, btc, first), cost, cash_ret)
