"""Smart Zones candidates, exits and the random-entry benchmark of research 13 (prereg §5–6).

Point in time: candidates of day t use rows <= t only and are filled at the open of t+1 by the
event-driven simulator of research 5. The zone is frozen at the signal close: stop and target are
price levels checked on each close and executed at the next open.
"""

import warnings
from dataclasses import asdict, dataclass

import numpy as np

from qlab.data.panel import Panel
from qlab.engine.vector import Decisions
from qlab.research5.sim import Candidates, SimOutput, SimSpec, simulate
from qlab.research9.strategy import UNIVERSE_N, sundays, universe
from qlab.research13.zones import ranges

MAX_POSITIONS = 5
MAX_HOLD = 60
STOP_BUFFER = 0.01
# fiat / fiat-backed pairs that slipped through the research 9 symbol filter (decision log 2026-10-03)
EXCLUDED = ("AUDUSDT", "BKRWUSDT")


@dataclass(frozen=True)
class Config:
    k: int                    # pivot lookback
    z: float                  # zone share of the range
    target: str               # EQ | PREM
    entry: str                # touch | bounce
    universe_n: int = UNIVERSE_N
    pivot_src: str = "hl"     # hl | close (robustness, prereg §9)

    @property
    def id(self) -> str:
        s = f"Z_k{self.k}_z{round(self.z * 100)}_{self.target}_{self.entry}"
        return s + ("_c" if self.pivot_src == "close" else "") + (
            f"_u{self.universe_n}" if self.universe_n != UNIVERSE_N else "")

    def as_dict(self) -> dict:
        return {**asdict(self), "study": "research13"}


def grid() -> list[Config]:
    """The 24 pre-registered candidates (prereg §5). With z = 0.5 the PREM target equals EQ."""
    return [Config(k, z, tg, e) for k in (5, 10, 20) for z in (0.25, 0.5) for tg in ("EQ", "PREM")
            for e in ("touch", "bounce")]


def excluded(symbols: list[str]) -> list[int]:
    return [symbols.index(s) for s in EXCLUDED if s in symbols]


def universe_mask(p: Panel, qv: np.ndarray, n: int, exclude: list[int] = ()) -> np.ndarray:
    """(T, N) bool: the research 9 universe (top n by 30-day quote volume) at the close of t, with the
    `exclude` columns removed before ranking."""
    m = np.zeros(p.shape, dtype=bool)
    drop = set(exclude)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # nanmean of a column without volume
        for t in range(p.shape[0]):
            U = [j for j in universe(p, qv, t, n + len(drop)) if j not in drop][:n]
            m[t, U] = True
    return m


def mean_qv30(qv: np.ndarray) -> np.ndarray:
    """Mean quote volume over days t-29..t, ignoring missing days (tie-break of the ranking)."""
    v = np.nan_to_num(qv)
    ok = (~np.isnan(qv)).astype(float)
    cs_v = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)])
    cs_n = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(ok, axis=0)])
    lo = np.maximum(np.arange(len(v)) - 29, 0)
    hi = np.arange(1, len(v) + 1)
    n = cs_n[hi] - cs_n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs_v[hi] - cs_v[lo]) / n, np.nan)


def entry_signal(close: np.ndarray, high: np.ndarray, low: np.ndarray, in_univ: np.ndarray,
                 cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(depth, bottom, top), each (T, N). depth = (close - bottom) / range where the close of t is in
    the discount zone (bottom < close <= bottom + z * range) and the coin is in the universe; NaN
    elsewhere."""
    if cfg.pivot_src == "close":
        high = low = close
    top, bot, valid = ranges(high, low, cfg.k)
    R = top - bot
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = in_univ & valid & (close > bot) & (close <= bot + cfg.z * R)
        if cfg.entry == "bounce":
            prev = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]])
            ok &= close > prev
        elif cfg.entry != "touch":
            raise ValueError(cfg.entry)
        depth = np.where(ok, (close - bot) / R, np.nan)
    return depth, bot, top


def candidates(depth: np.ndarray, bot: np.ndarray, top: np.ndarray, qv30: np.ndarray, cfg: Config,
               first: int, last: int) -> Candidates:
    """Candidates of signal days [first, last): deepest in the zone first, tie -> higher qv30.
    Stop and target come from the zone frozen at the signal close."""
    if cfg.target not in ("EQ", "PREM"):
        raise ValueError(cfg.target)
    t, a = np.nonzero(np.isfinite(depth[first:last]))
    t = t + first
    order = np.lexsort((a, -np.nan_to_num(qv30[t, a], nan=-np.inf), depth[t, a], t))
    t, a = t[order], a[order]
    b, R = bot[t, a], top[t, a] - bot[t, a]
    target = b + 0.5 * R if cfg.target == "EQ" else top[t, a] - cfg.z * R
    return Candidates(t, a, depth[t, a], stop_px=b * (1 - STOP_BUFFER), target_px=target)


def spec(entry_delay: int = 1, stop_at_level: bool = False) -> SimSpec:
    return SimSpec(MAX_POSITIONS, MAX_HOLD, entry_delay=entry_delay, reentry_same_open=False,
                   stop_px_fill_at_level=stop_at_level)


def ew_decisions(p: Panel, in_univ: np.ndarray, first: int) -> Decisions:
    """B_EW: equal weights of the universe mask, decided after the Sunday close, filled Monday open."""
    days = [t for t in sundays(p) if first <= t < len(p.dates) - 1]
    w = in_univ[days].astype(float)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1)
    return Decisions(np.array(days, dtype=int), w)


def run(p: Panel, extra: dict, cfg: Config, rate: np.ndarray, start: int, end: int,
        in_univ: np.ndarray, entry_delay: int = 1, stop_at_level: bool = False) -> SimOutput:
    """Simulate days [start, end) from an empty book. `rate`: per-side cost matrix of research 9,
    `in_univ`: universe_mask for cfg.universe_n."""
    qv = extra["qv"]
    depth, bot, top = entry_signal(np.asarray(p.close_u), extra["high"], extra["low"], in_univ, cfg)
    cands = candidates(depth, bot, top, mean_qv30(qv), cfg, start, end - 1)
    no_adv = np.full(p.shape, np.nan)
    fee = lambda t, a: float(rate[t, a])  # noqa: E731
    return simulate(p, cands, spec(entry_delay, stop_at_level), no_adv, start, end, fee, fee)


def random_benchmark(p: Panel, in_univ: np.ndarray, out: SimOutput, rate: np.ndarray, start: int,
                     end: int, seed: int) -> SimOutput:
    """B_RND (prereg §6): on each signal day with m strategy entries, m random universe coins without an
    open position; holding periods drawn from the strategy's empirical distribution; same slots, size,
    costs, fills."""
    rng = np.random.default_rng(seed)
    m = np.zeros(p.shape[0], dtype=np.int64)
    np.add.at(m, out.trades["signal_day"].to_numpy(), 1)
    holds = out.trades["held"].to_numpy()
    ts, as_ = [], []
    for t in np.flatnonzero(m):
        U = np.flatnonzero(in_univ[t])
        ts.append(np.full(len(U), t))
        as_.append(rng.permutation(U))
    t = np.concatenate(ts) if ts else np.array([], dtype=np.int64)
    a = np.concatenate(as_) if as_ else np.array([], dtype=np.int64)
    hold = rng.choice(np.maximum(holds, 1), len(t)) if len(holds) else np.ones(len(t), dtype=np.int64)
    cands = Candidates(t, a, np.zeros(len(t)), hold=hold, max_entries=m)
    fee = lambda t, a: float(rate[t, a])  # noqa: E731
    return simulate(p, cands, spec(), np.full(p.shape, np.nan), start, end, fee, fee)
