"""H1a: crypto Donchian ensemble with monthly top-N rotation (prereg §4.1) and the BTC-only replication.

Point in time: the decision after the close of t uses rows <= t (signals, volatility, the universe fixed at the
last month end <= t) and is filled at the open of t+1 by `qlab.engine.vector.simulate`.
"""

from dataclasses import asdict, dataclass

import numpy as np

from qlab.data.panel import Panel
from qlab.engine.vector import Decisions, SimResult, simulate
from qlab.features.basic import tr_index
from qlab.research9.panel import cost_rate
from qlab.research19 import setup as S
from qlab.research19.signals import ensemble, realized_vol

MIN_TRADED_DAYS = 60     # same liquidity condition as research9.strategy.universe


@dataclass(frozen=True)
class Config:
    name: str = "h1a"
    lookbacks: tuple[int, ...] = S.LOOKBACKS_CRYPTO
    universe_n: int = S.CRYPTO_TOP_N
    min_age: int = S.CRYPTO_MIN_AGE         # 0 = no age condition
    vol_target: float = S.CRYPTO_VOL_TARGET
    leverage_cap: float = S.LEVERAGE_CAP
    band: float | None = S.BAND
    cost_mult: float = 1.0
    cost_const: float | None = None         # per-side constant instead of the liquidity tiers (e.g. 0.0025)
    tbill_cash: bool = False
    delay: int = 0                          # 1: decision t is filled at the open of t+2
    btc_only: bool = False

    def as_dict(self) -> dict:
        return {**asdict(self), "study": "research19"}


def universe_aged(p: Panel, qv: np.ndarray, t: int, n: int, min_age: int, listed_cum: np.ndarray,
                  excluded: np.ndarray) -> np.ndarray:
    """Top-n by 30-day mean quote volume at close t among symbols traded on each of the last 60 days, not in
    `excluded` (prereg v1.1) and listed for >= min_age days; age and exclusion apply BEFORE the cut."""
    if t + 1 < MIN_TRADED_DAYS:
        return np.array([], dtype=int)
    ok = np.asarray(p.tradable[t + 1 - MIN_TRADED_DAYS:t + 1]).all(axis=0) & ~excluded
    if min_age > 0:
        ok &= listed_cum[t] >= min_age
    with np.errstate(invalid="ignore"):
        v = np.where(ok, np.nanmean(qv[max(0, t - 29):t + 1], axis=0), -np.inf)
    order = np.argsort(-v, kind="stable")
    return np.array([j for j in order[:n] if ok[j]], dtype=int)


def monthly_members(p: Panel, qv: np.ndarray, n: int, min_age: int, excluded: np.ndarray) -> np.ndarray:
    """(T, N) bool: universe in force for the decision after the close of t = the one computed at the last
    month end <= t (including t itself when it is a month end)."""
    T = len(p.dates)
    month = p.dates.astype("datetime64[M]")
    listed_cum = np.cumsum(np.asarray(p.listed), axis=0)
    out = np.zeros(p.shape, dtype=bool)
    cur = np.zeros(p.shape[1], dtype=bool)
    for t in range(T):
        if t + 1 < T and month[t + 1] != month[t]:
            cur = np.zeros(p.shape[1], dtype=bool)
            cur[universe_aged(p, qv, t, n, min_age, listed_cum, excluded)] = True
        out[t] = cur
    return out


def excluded_mask(symbols: list[str]) -> np.ndarray:
    return np.array([s in S.EXCLUDED_NONCRYPTO for s in symbols])


def signals(p: Panel, cfg: Config, cols: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble E and 90-day volatility (annualised sqrt(365)) as (T, N); columns outside `cols` stay 0 / NaN."""
    P = tr_index(p)
    valid = np.asarray(p.listed & p.tradable)
    cols = np.arange(p.shape[1]) if cols is None else cols
    E, sig = np.zeros(p.shape), np.full(p.shape, np.nan)
    E[:, cols] = ensemble(P[:, cols], valid[:, cols], cfg.lookbacks)
    sig[:, cols] = realized_vol(P[:, cols], valid[:, cols], S.VOL_WINDOW, S.CRYPTO_YEAR)
    return E, sig


def target_weights(E: np.ndarray, sig: np.ndarray, members: np.ndarray, cfg: Config,
                   scale: np.ndarray | None = None) -> np.ndarray:
    """s_i = E_i * min(cap, vol_target / sigma_i); w_i = s_i / N for members; scaled to sum 1 if above (spot)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        size = np.minimum(cfg.leverage_cap, cfg.vol_target / sig)
    s = np.where(np.isfinite(size), E * size, 0.0)
    w = np.where(members, s / cfg.universe_n, 0.0)
    if scale is not None:
        w = w * np.asarray(scale, dtype=float)[:, None]
    tot = w.sum(axis=1, keepdims=True)
    return np.where(tot > 1.0, w / np.maximum(tot, 1e-300), w)


def btc_weights(E: np.ndarray, sig: np.ndarray, btc: int, cfg: Config,
                scale: np.ndarray | None = None) -> np.ndarray:
    """w = E * min(1, vol_target / sigma) for BTC only."""
    with np.errstate(divide="ignore", invalid="ignore"):
        size = np.minimum(1.0, cfg.vol_target / sig[:, btc])
    w = np.zeros(E.shape)
    w[:, btc] = np.where(np.isfinite(size), E[:, btc] * size, 0.0)
    if scale is not None:
        w *= np.asarray(scale, dtype=float)[:, None]
    return w


def decisions(w: np.ndarray, first: int, delay: int = 0) -> Decisions:
    """One decision per day from `first`; `delay` days later fills (rows beyond the panel are dropped)."""
    days = np.arange(first, w.shape[0])
    rows = w[days]
    if delay:
        days, rows = days + delay, rows
        keep = days < w.shape[0]
        days, rows = days[keep], rows[keep]
    return Decisions(days.astype(int), rows)


def run_panel(p: Panel, qv: np.ndarray, symbols: list[str], cfg: Config, first: int,
              tbill: np.ndarray | None = None, E_override: np.ndarray | None = None,
              scale: np.ndarray | None = None, signal_cache: tuple | None = None) -> SimResult:
    """Simulate H1a (or BTC-only) from the close of day `first`. `E_override` replaces the ensemble (null runs)."""
    excl = excluded_mask(symbols)
    if cfg.btc_only:
        btc = symbols.index("BTCUSDT")
        E, sig = signal_cache or signals(p, cfg, np.array([btc]))
        w = btc_weights(E if E_override is None else E_override, sig, btc, cfg, scale)
    else:
        members = monthly_members(p, qv, cfg.universe_n, cfg.min_age, excl)
        cols = np.flatnonzero(members.any(axis=0))
        E, sig = signal_cache or signals(p, cfg, cols)
        w = target_weights(E if E_override is None else E_override, sig, members, cfg, scale)
    cost = (np.full(p.shape, cfg.cost_const) if cfg.cost_const is not None
            else cost_rate(qv, cfg.cost_mult))
    return simulate(p, decisions(w, first, cfg.delay), cost, tbill if cfg.tbill_cash else None, cfg.band)


def run(cfg: Config, tbill: np.ndarray | None = None, scale: np.ndarray | None = None) -> SimResult:
    """H1a on the real crypto panel from the close of 2018-12-31 (declared fully seen, see `setup.load_crypto`)."""
    p, ex, syms = S.load_crypto()
    first = S.index_of(p, S.PERIOD_H1A[0]) - 1
    return run_panel(p, np.asarray(ex["qv"]), syms, cfg, first, tbill, scale=scale)
