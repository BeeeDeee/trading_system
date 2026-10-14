"""LEAD on altcoins (prereg §4-8): BTC leader features, signals on the follower's own grid, member sleeves,
event-level null and bootstrap. Signal logic is that of `qlab.research16.families.lead` (tested against it).

Everything is keyed by absolute 15m index (`t = epoch_ms // 900_000`). A follower's panel starts on its own
multiple of 16, so the leader is matched by time, never by array position.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from qlab.research15 import strategy as S
from qlab.research16 import engine as G, families as F
from qlab.research17.setup import COST, HOLD, K, WEIGHT, day0, n_days

DEFAULT_COST = COST


@dataclass(frozen=True)
class Scenario:
    name: str
    leader: str = "spot"               # "spot" | "perp"
    K: float = K
    side: str = "LONG"                 # "LONG" | "LS"
    lag: bool = True                   # require the follower to lag (rA / rB < 0.5)
    cost: float = COST
    delay: int = 0
    use_funding: bool = True


@dataclass
class Leader:
    t0: int
    move: np.ndarray                   # 1h log move, NaN unless candles k-4..k are real
    sig: np.ndarray                    # 2 * sigma of 15m returns, NaN without enough data


def leader(d: dict) -> Leader:
    a = SimpleNamespace(c=d["c"], real=d["real"])
    return Leader(int(d["t"][0]), F.move_1h(a), 2.0 * F.sigma(a))


def signals(a: S.Asset, ld: Leader, K: float, side: str, lag: bool = True, ra: np.ndarray | None = None) -> G.Cand:
    """Long (and with side "LS" short) candidates of the follower `a`: leader move >= K * 2 sigma and, with `lag`,
    follower move / leader move < 0.5. Decided at the close of candle k, filled at close[k]; held 4 h."""
    idx = a.t - ld.t0
    ok = (idx >= 0) & (idx < len(ld.move))
    idx = np.where(ok, idx, 0)
    rb = np.where(ok, ld.move[idx], np.nan)
    sg = np.where(ok, ld.sig[idx], np.nan)
    ra = F.move_1h(a) if ra is None else ra
    with np.errstate(invalid="ignore", divide="ignore"):
        big = (np.abs(rb) >= K * sg) & a.real
        if lag:
            big &= (ra / rb < 0.5)
    lg, sh = big & (rb > 0), big & (rb < 0) & (side == "LS")
    return F._mk(lg, sh, HOLD)


# ---------------------------------------------------------------- one follower, all scenarios

@dataclass
class Result:
    entries: np.ndarray                # absolute 15m index of each fill
    rets: np.ndarray                   # net return of each trade
    daily: np.ndarray                  # weighted contribution to the portfolio, fixed daily grid


def _to_grid(a_t: np.ndarray, dates: np.ndarray, vals: np.ndarray, w: float, out: np.ndarray) -> None:
    i = dates.astype("datetime64[D]").astype(np.int64) - day0()
    ok = (i >= 0) & (i < len(out))
    out[i[ok]] += w * vals[ok]


def light(a: S.Asset) -> SimpleNamespace:
    """What the null needs of a follower: closes, funding and real flags on its own grid."""
    return SimpleNamespace(t0=int(a.t[0]), n=len(a.c), c=a.c, fund=a.fund, real=a.real)


def run_follower(a: S.Asset, leaders: dict[str, Leader], scenarios: list[Scenario], start_idx: int,
                 allowed: np.ndarray | None, weight: float = WEIGHT) -> dict[str, Result]:
    """All scenarios of one follower on [start_idx, end of data); `allowed` restricts entries (membership)."""
    ra = F.move_1h(a)
    n, out = len(a.c), {}
    for sc in scenarios:
        cs = signals(a, leaders[sc.leader], sc.K, sc.side, sc.lag, ra)
        b = G.trade_book(a, cs, start_idx, n, delay=sc.delay, cost=sc.cost, use_funding=sc.use_funding,
                         allowed=allowed)
        daily = np.zeros(n_days())
        _to_grid(a.t, *S.daily(a.t, b.net, start_idx, n), weight, daily)
        out[sc.name] = Result(a.t[b.entries] if len(b.entries) else np.zeros(0, np.int64), b.ret, daily)
    return out


def member_mask(a: S.Asset, sym: str, members: dict[str, list[str]]) -> np.ndarray:
    """Per candle: the symbol is a member in that candle's month (prereg §3)."""
    um, inv = np.unique(a.month, return_inverse=True)
    mem = np.array([sym in members.get(f"{1970 + m // 12}-{m % 12 + 1:02d}", []) for m in um])
    return mem[inv]


# ---------------------------------------------------------------- null (prereg §6)

def month_key(t: np.ndarray) -> np.ndarray:
    return (np.asarray(t) * 900).astype("datetime64[s]").astype("datetime64[M]").astype(np.int64)


def events_of(results: dict[str, Result], delay: int = 0) -> dict[int, list[str]]:
    """Signal candle (the leader's candle k) -> symbols that traded on it."""
    ev: dict[int, list[str]] = {}
    for sym, r in results.items():
        for t in r.entries:
            ev.setdefault(int(t) - delay, []).append(sym)
    return ev


def candidate_times(leader_t: np.ndarray, leader_real: np.ndarray, first: int, last: int) -> dict[int, np.ndarray]:
    """Month -> absolute candles where the leader exists, inside [first, last] (the null's random draws)."""
    m = leader_real & (leader_t >= first) & (leader_t <= last)
    t = leader_t[m]
    mo = month_key(t)
    return {int(k): t[mo == k] for k in np.unique(mo)}


def null_run(lights: dict[str, SimpleNamespace], events: dict[int, list[str]], times: dict[int, np.ndarray],
             rng, cost: float = COST, weight: float = WEIGHT, use_funding: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """B_RND_EVT: every event moves to a random candle of its own calendar month and all of its trades move
    with it; a trade is dropped if the follower has no real candle there or the follower already holds a
    position (events in time order of the original). Returns (daily grid, trade returns)."""
    daily, rets, busy = np.zeros(n_days()), [], {}
    for k in sorted(events):
        pool = times[int(month_key(np.array([k]))[0])]
        tau = int(pool[rng.integers(len(pool))])
        for sym in events[k]:
            L = lights[sym]
            e = tau - L.t0
            if e < 0 or e >= L.n - 1 or not L.real[e]:
                continue
            last = min(e + HOLD, L.n - 1)
            if any(e <= y and x <= last for x, y in busy.get(sym, ())):
                continue
            busy.setdefault(sym, []).append((e, last))
            hr = S._bar_returns(L, e, last, float(L.c[last]), 1, cost, use_funding)
            rets.append(float(np.prod(1.0 + hr) - 1.0))
            d = (L.t0 + np.arange(e + 1, last + 1)) // 96 - day0()
            ok = (d >= 0) & (d < len(daily))
            np.add.at(daily, d[ok], weight * hr[ok])
    return daily, np.array(rets)


def event_ci(entries: list[np.ndarray], rets: list[np.ndarray], delay: int = 0, n_boot: int = 2000,
             alpha: float = 0.10, seed: int = 0) -> tuple[float, float, float, int]:
    """Mean trade return and bootstrap interval resampling whole events (prereg §8.1).
    Returns (mean, lo, hi, number of events)."""
    t = np.concatenate(entries) - delay if entries else np.zeros(0, np.int64)
    r = np.concatenate(rets) if rets else np.zeros(0)
    if not len(r):
        return float("nan"), float("nan"), float("nan"), 0
    u, inv = np.unique(t, return_inverse=True)
    s, c = np.bincount(inv, r), np.bincount(inv).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(u), (n_boot, len(u)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return float(r.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2)), int(len(u))


# ---------------------------------------------------------------- B_HOLD (prereg §6)

def hold_daily(a: S.Asset, start_idx: int) -> np.ndarray:
    """Buy-and-hold simple daily return of the follower (no costs) on the fixed daily grid."""
    lr = np.zeros(len(a.c))
    lr[1:] = np.log(a.c[1:] / a.c[:-1])
    dates, v = S.daily(a.t, lr, start_idx, len(a.c))
    out = np.zeros(n_days())
    i = dates.astype("datetime64[D]").astype(np.int64) - day0()
    ok = (i >= 0) & (i < len(out))
    out[i[ok]] = np.expm1(v[ok])
    return out


def member_days(sym: str, members: dict[str, list[str]]) -> np.ndarray:
    """Per day of the fixed grid: the symbol is a member in that day's month."""
    days = (day0() + np.arange(n_days())).astype("datetime64[D]")
    keys = days.astype("datetime64[M]").astype(str)
    return np.array([sym in members.get(k, []) for k in keys])
