"""Generic trade simulator and context-matched random null of research 16 (prereg §5-6).

A family emits candidates (`Cand`): decided at the close of candle k from candles <= k, filled at close[k+delay],
with an absolute stop and target (NaN = none), a target as a multiple of the risk (`rmult`, set at the fill) and
a `deadline` (last candle the trade may be held to; exit at its close). Exit rules, costs and funding are those
of research 15 (`qlab.research15.strategy`): stop before target in the same candle, gaps fill at the open.
"""

from dataclasses import dataclass

import numpy as np

from qlab.research15 import strategy as S

COST = S.COST
STOP, TARGET, TIME, END = S.STOP, S.TARGET, S.TIME, S.END


@dataclass
class Asset(S.Asset):
    fraw: np.ndarray | None = None      # perp funding as recorded (NaN where none), for signals

    @classmethod
    def build(cls, d: dict, funding: np.ndarray | None = None) -> "Asset":
        base = S.Asset.build(d, funding)
        raw = d.get("funding", np.full(len(d["c"]), np.nan)) if funding is None else funding
        return cls(**vars(base), fraw=np.asarray(raw, float))


@dataclass
class Cand:
    k: np.ndarray
    side: np.ndarray
    stop: np.ndarray
    target: np.ndarray
    rmult: np.ndarray
    deadline: np.ndarray

    @classmethod
    def make(cls, k, side, deadline, stop=None, target=None, rmult=None) -> "Cand":
        k = np.asarray(k, np.int64)
        nan = np.full(len(k), np.nan)
        f = lambda x: nan.copy() if x is None else np.broadcast_to(np.asarray(x, float), k.shape).copy()
        o = np.argsort(k, kind="stable")
        return cls(k[o], np.broadcast_to(np.asarray(side, np.int8), k.shape)[o].copy(), f(stop)[o], f(target)[o],
                   f(rmult)[o], np.broadcast_to(np.asarray(deadline, np.int64), k.shape)[o].copy())

    @classmethod
    def empty(cls) -> "Cand":
        return cls.make([], [], [])

    def before(self, n: int) -> "Cand":
        m = self.k < n
        return Cand(*(getattr(self, f)[m] for f in ("k", "side", "stop", "target", "rmult", "deadline")))

    def __len__(self) -> int:
        return len(self.k)


def exit_(a: S.Asset, e: int, s: int, St: float, T: float, last: int, deadline: int) -> tuple[int, float, int]:
    """First real candle in (e, last] touching stop or target (stop first); else the close of `last`."""
    sl = slice(e + 1, last + 1)
    with np.errstate(invalid="ignore"):
        st, tg = (a.l[sl] <= St, a.h[sl] >= T) if s == 1 else (a.h[sl] >= St, a.l[sl] <= T)
    hit = (st | tg) & a.real[sl]
    if hit.any():
        i = int(hit.argmax())
        j = e + 1 + i
        if st[i]:
            return j, (min(a.o[j], St) if s == 1 else max(a.o[j], St)), STOP
        return j, (max(a.o[j], T) if s == 1 else min(a.o[j], T)), TARGET
    return last, float(a.c[last]), (TIME if last == deadline else END)


def trade_book(a: S.Asset, c: Cand, start: int, end: int, delay: int = 0, cost: float = COST,
               use_funding: bool = True, allowed: np.ndarray | None = None) -> S.Book:
    """Sequential trades on [start, end): one position at a time, candidates during a position are ignored.
    A candidate with a stop is skipped if its risk at the fill is outside 0.15-5 % or (with a delay) the stop
    was touched before the fill. `allowed` (per candle) restricts entries only."""
    n = len(a.c)
    net, rec, blocked = np.zeros(n), [], -1
    for i in range(len(c)):
        k = int(c.k[i])
        if k < start or k >= end - 1 or k <= blocked:
            continue
        s, St, dl = int(c.side[i]), float(c.stop[i]), int(c.deadline[i])
        e, last = k + delay, min(dl, end - 1)
        if e >= last or (allowed is not None and not allowed[e]):
            continue
        if delay and np.isfinite(St):
            sl = slice(k + 1, e + 1)
            if (a.real[sl] & ((a.l[sl] <= St) if s == 1 else (a.h[sl] >= St))).any():
                continue
        p = float(a.c[e])
        risk = s * (p - St) / p
        if np.isfinite(St) and not (S.R_MIN <= risk <= S.R_MAX):
            continue
        T = float(c.target[i]) if np.isfinite(c.target[i]) else p + s * float(c.rmult[i]) * abs(p - St)
        x, price, why = exit_(a, e, s, St, T, last, dl)
        hr = S._bar_returns(a, e, x, price, s, cost, use_funding)
        net[e + 1:x + 1] += hr
        rec.append((e, s, risk, x, why, float(np.prod(1.0 + hr) - 1.0)))
        blocked = x
    return S._book(rec, net)


@dataclass
class Context:
    """Where B_RND may enter (prereg §6): per side a candle mask, optional last candle per entry candle
    (`day_end`, ORB) and `same_entries` (CME: the reference entry candles with a random side)."""
    long: np.ndarray
    short: np.ndarray
    day_end: np.ndarray | None = None
    same_entries: bool = False


def random_book(a: S.Asset, ref: S.Book, ctx: Context, start: int, end: int, rng, cost: float = COST,
                use_funding: bool = True) -> S.Book:
    """Per (month, side) as many trades as `ref`, at random context candles without an open position; holding
    period drawn from `ref` (same side), exit at the close (no stop or target)."""
    n = len(a.c)
    occ, net, rec = np.zeros(n, bool), np.zeros(n), []
    hold = ref.exits - ref.entries

    def place(e: int, s: int, d: int) -> bool:
        last = min(e + d, end - 1, int(ctx.day_end[e]) if ctx.day_end is not None else n)
        if last <= e or occ[e:last + 1].any():
            return False
        hr = S._bar_returns(a, e, last, float(a.c[last]), s, cost, use_funding)
        net[e + 1:last + 1] += hr
        occ[e:last + 1] = True
        rec.append((e, s, np.nan, last, TIME, float(np.prod(1.0 + hr) - 1.0)))
        return True

    if ctx.same_entries:
        for e in ref.entries:
            if start <= e < end - 1:
                place(int(e), int(rng.choice((1, -1))), int(rng.choice(hold)))
    else:
        need: dict[tuple[int, int], int] = {}
        for e, s in zip(ref.entries, ref.sides):
            need[(int(a.month[e]), int(s))] = need.get((int(a.month[e]), int(s)), 0) + 1
        for (m, s), cnt in sorted(need.items()):
            lo = max(int(np.searchsorted(a.month, m, "left")), start)
            hi = min(int(np.searchsorted(a.month, m, "right")), end - 1)
            mask = ctx.long if s == 1 else ctx.short
            cand = rng.permutation(lo + np.flatnonzero(mask[lo:hi]))
            draw = rng.choice(hold[ref.sides == s], size=len(cand))
            got = 0
            for e, d in zip(cand, draw):
                if not occ[e] and place(int(e), s, int(d)):
                    got += 1
                    if got == cnt:
                        break
    rec.sort()
    return S._book(rec, net)


def holm(p: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down over the families that reached the holdout: the i-th smallest p (from 0) is compared with
    alpha / (k - i); once one fails, all larger ones fail."""
    out, ok, k = {}, True, len(p)
    for i, name in enumerate(sorted(p, key=lambda x: (p[x], x))):
        thr = alpha / (k - i)
        ok = ok and p[name] <= thr
        out[name] = {"p": p[name], "threshold": thr, "pass": bool(ok)}
    return out
