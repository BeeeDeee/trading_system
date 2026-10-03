"""Weekly targets of the 17 pre-registered candidates (prereg §4.2-4.3).

A decision for Monday t (filled at its open) uses only rows <= t-1: funding events with ts < t 00:00
(`fund_sig`), closes and volumes up to Sunday.
"""

from dataclasses import asdict, dataclass

import numpy as np

from .panel import Panel, mondays, pair_ok

UNIVERSE_N = 20
MIN_LISTED_DAYS = 30


@dataclass(frozen=True)
class Config:
    family: str               # S0 | S1 | S2
    L: int = 7                # trailing funding window (days)
    theta: float = 0.0        # annualized funding threshold
    K: int = 0                # S2: number of coins
    universe_n: int = UNIVERSE_N
    s: float = 2 / 3

    @property
    def id(self) -> str:
        if self.family == "S0":
            return "S0"
        th = f"{self.theta * 100:.0f}"
        return f"S1_L{self.L}_th{th}" if self.family == "S1" else f"S2_K{self.K}_L{self.L}_th{th}"

    def as_dict(self) -> dict:
        return asdict(self)


def grid() -> list[Config]:
    out = [Config("S0")]
    out += [Config("S1", L=L, theta=th) for L in (7, 30) for th in (0.0, 0.10)]
    out += [Config("S2", L=L, theta=th, K=K) for K in (3, 5, 10) for L in (7, 30) for th in (0.0, 0.10)]
    return out


def trailing_funding(p: Panel, t: int, L: int) -> np.ndarray:
    """Annualized sum of funding over days t-L..t-1; NaN if the perp did not trade on all L days."""
    if t < L:
        return np.full(len(p.symbols), np.nan)
    f = p.m["fund_sig"][t - L:t]
    live = ~np.isnan(p.m["pc"][t - L:t])
    out = np.nansum(f, axis=0) * 365 / L
    return np.where(live.all(axis=0), out, np.nan)


def universe(p: Panel, t: int, n: int) -> np.ndarray:
    """Indices of the top-n perps by 30-day mean quote volume, listed >= 30 days, with spot data, at t-1."""
    if t < MIN_LISTED_DAYS:
        return np.array([], dtype=int)
    win = slice(t - MIN_LISTED_DAYS, t)
    ok = pair_ok(p)[win].all(axis=0)
    qv = np.where(ok, np.nanmean(p.m["qv"][win], axis=0), -np.inf)
    order = np.argsort(-qv, kind="stable")
    return np.array([j for j in order[:n] if ok[j]], dtype=int)


def targets(p: Panel, cfg: Config) -> dict[int, dict[int, float]]:
    sym = {s: j for j, s in enumerate(p.symbols)}
    btc_eth = [sym["BTCUSDT"], sym["ETHUSDT"]]
    ok = pair_ok(p)
    out = {}
    for t in mondays(p.dates):
        if t == 0:
            continue
        if cfg.family == "S0":
            out[int(t)] = {j: 0.5 for j in btc_eth if ok[t - 1, j]}
        elif cfg.family == "S1":
            F = trailing_funding(p, t, cfg.L)
            out[int(t)] = {j: 0.5 for j in btc_eth if ok[t - 1, j] and F[j] > cfg.theta}
        else:
            F = trailing_funding(p, t, cfg.L)
            U = [j for j in universe(p, t, cfg.universe_n) if F[j] > cfg.theta]
            U.sort(key=lambda j: (-F[j], p.symbols[j]))
            out[int(t)] = {j: 1.0 / cfg.K for j in U[:cfg.K]}
    return out
