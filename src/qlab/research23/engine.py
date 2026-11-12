"""Trade simulator and the day-matched random null of research 23 (prereg §5-6).

A signal decided at the close of row k fills at close[k + delay]. Stop and target are watched from the next
row: stop first in the same candle, a gap through the stop fills at the open, a target (limit) fills at its
price or at a better open. Every trade is closed at the close of the last candle of its UTC session at the
latest, so all trades open and close on the same UTC day. One position per asset; a signal on a row <= the
previous exit row is ignored. Net return = side * (exit / fill - 1) - funding of settlements held - 2 * cost.
"""

import numpy as np
from numba import njit

from qlab.research23.data import PER_DAY
from qlab.research23.setups import WIN_HI, WIN_LO, Signals

COST = 7.5e-4
STOP, TARGET, END = 0, 1, 2
T_NONE, T_2R, T_VWAP, T_BAND = 0, 1, 2, 3
TMODE = {"EOD": T_NONE, "2R": T_2R, "VWAP": T_VWAP, "OPP": T_BAND, "BAND": T_BAND}


@njit(cache=True)
def _exit(o, h, l, c, real, e, last, s, St, T):
    """(exit row, exit price, reason) for a position filled at row e (T NaN = no target)."""
    for j in range(e + 1, last + 1):
        if not real[j]:
            continue
        if (s == 1 and l[j] <= St) or (s == -1 and h[j] >= St):
            return j, (min(o[j], St) if s == 1 else max(o[j], St)), STOP
        if T == T and ((s == 1 and h[j] >= T) or (s == -1 and l[j] <= T)):
            return j, (max(o[j], T) if s == 1 else min(o[j], T)), TARGET
    return last, c[last], END


@njit(cache=True)
def _book(o, h, l, c, real, fund, pos, ks, sides, stops, tg, rmult, start, end, delay, cost):
    n = len(ks)
    E = np.empty(n, np.int64); X = np.empty(n, np.int64); SD = np.empty(n, np.int8)
    RISK = np.empty(n); TREL = np.empty(n); RET = np.empty(n); WHY = np.empty(n, np.int8)
    m, blocked = 0, -1
    for i in range(n):
        k = ks[i]
        if k < start or k >= end - 1 or k <= blocked:
            continue
        s, St = sides[i], stops[i]
        e = k + delay
        last = min(k - pos[k] + PER_DAY - 1, end - 1)
        if e >= last:
            continue
        if delay > 0:
            hit = False
            for j in range(k + 1, e + 1):
                if real[j] and ((s == 1 and l[j] <= St) or (s == -1 and h[j] >= St)):
                    hit = True
            if hit:
                continue
        p = c[e]
        risk = s * (p - St) / p
        if not risk > 0:
            continue
        T = tg[i]
        if T == T:
            if not s * (T - p) > 0:
                continue
        elif rmult == rmult:
            T = p + s * rmult * abs(p - St)
        x, px, why = _exit(o, h, l, c, real, e, last, s, St, T)
        f = 0.0
        for j in range(e + 1, x + 1):
            f += fund[j]
        E[m], X[m], SD[m], RISK[m] = e, x, s, risk
        TREL[m] = s * (T - p) / p if T == T else np.nan
        RET[m] = s * (px / p - 1.0) - s * f - 2.0 * cost
        WHY[m] = why
        m += 1
        blocked = x
    return E[:m], X[:m], SD[:m], RISK[:m], TREL[:m], RET[:m], WHY[:m]


class Book:
    """Trades of one asset: entry/exit rows, side, relative risk and target distance, net return, reason."""

    def __init__(self, e, x, side, risk, trel, ret, why):
        self.e, self.x, self.side, self.risk, self.trel, self.ret, self.why = e, x, side, risk, trel, ret, why

    def __len__(self) -> int:
        return len(self.ret)


def targets(sig: Signals, target: str) -> tuple[np.ndarray, float]:
    """Absolute target per signal and the R multiple (NaN where unused)."""
    mode = TMODE[target]
    nan = np.full(len(sig), np.nan)
    if mode == T_VWAP:
        return sig.vw, np.nan
    if mode == T_BAND:
        return sig.band, np.nan
    return nan, (2.0 if mode == T_2R else np.nan)


def trade_book(a: dict, sig: Signals, target: str, start: int, end: int, delay: int = 0,
               cost: float = COST) -> Book:
    tg, rm = targets(sig, target)
    return Book(*_book(a["o"], a["h"], a["l"], a["c"], a["real"], a["fund"], a["pos"], sig.k, sig.side.astype(np.int64),
                       sig.stop, tg, rm, start, end, delay, cost))


@njit(cache=True)
def _null(o, h, l, c, real, fund, pos, E, SD, RISK, TREL, start, end, cost, n_rep, seed, tries):
    """Per replication: sum and count of net returns of random same-day entries (prereg §6, v1.0.2).

    Random trades do not block each other: a non-overlap rule would accept or reject a draw depending on how
    long the trade lasted, i.e. on its outcome (quick stop-outs fit, winners held to the session end collide),
    which biases the null downwards."""
    np.random.seed(seed)
    sums = np.zeros(n_rep)
    cnts = np.zeros(n_rep, np.int64)
    for r in range(n_rep):
        for i in range(len(E)):
            day0 = E[i] - pos[E[i]]
            lo, hi = max(day0 + WIN_LO, start), min(day0 + WIN_HI, end - 2)
            last = min(day0 + PER_DAY - 1, end - 1)
            if hi < lo:
                continue
            s = SD[i]
            for _ in range(tries):
                e = np.random.randint(lo, hi + 1)
                if not real[e] or e >= last:
                    continue
                p = c[e]
                St = p * (1.0 - s * RISK[i])
                T = p * (1.0 + s * TREL[i]) if TREL[i] == TREL[i] else np.nan
                x, px, why = _exit(o, h, l, c, real, e, last, s, St, T)
                f = 0.0
                for j in range(e + 1, x + 1):
                    f += fund[j]
                sums[r] += s * (px / p - 1.0) - s * f - 2.0 * cost
                cnts[r] += 1
                break
    return sums, cnts


def null_sums(a: dict, b: Book, start: int, end: int, n_rep: int, seed: int, cost: float = COST,
              tries: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """B_RND sums/counts per replication; a trade that finds no real candle in `tries` draws is dropped."""
    return _null(a["o"], a["h"], a["l"], a["c"], a["real"], a["fund"], a["pos"], b.e, b.side.astype(np.int64),
                 b.risk, b.trel, start, end, cost, n_rep, seed, tries)
