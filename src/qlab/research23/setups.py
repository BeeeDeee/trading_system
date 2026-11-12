"""The four VWAP Wave setups as state machines (prereg §4).

Each machine is written for the LONG side only; the short side runs the same machine on the mirrored series
(prices negated, high <-> low), which maps U <-> L, "above" <-> "below" and strength <-> weakness exactly.
A signal at row t uses rows <= t only. Output per signal: row t, side, stop, V[t] and the band in the trade
direction at t (U for a long), from which the variant picks its target. States reset at every session start.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from qlab.research23.data import PER_DAY

WIN_LO, WIN_HI = 6, 275          # entry window: position in the session
BUF, TOL = 0.25, 0.1             # stop buffer (x ATR), band proximity (x k*sigma)
ACC_RUN, ACC_DIST = 3, 0.5       # S1 acceptance: closes outside in a row, or distance beyond the band (x sigma)
PB_WIN, TRIG_WIN = 6, 3          # trigger windows after a backtest (S1, S3) / a test (S2, S4)
CTX_IN = 6                       # S2: closes inside value before the test
OUT_RUN, OUT_DIST = 3, 0.5       # S3: closes outside in a row and low beyond the band (x sigma)
RTV_IN, RTV_LIFE = 2, 24         # S3: closes back inside before a pullback; life of the return (candles)
BOUNCE_LOOK = 24                 # S4: a return into value within the last 24 candles

SETUPS = ("S1", "S2", "S3", "S4")
ENTRIES = {"S1": ("PB", "BRK"), "S2": ("REJ", "TST"), "S3": ("PB", "BRK"), "S4": ("STR", "TST")}
TARGETS = {"S1": ("EOD", "2R"), "S2": ("VWAP", "OPP"), "S3": ("VWAP", "OPP"), "S4": ("BAND", "2R")}
PRIMARY = {"S1": ("PB", "EOD"), "S2": ("REJ", "VWAP"), "S3": ("PB", "VWAP"), "S4": ("STR", "BAND")}


@dataclass(frozen=True)
class Variant:
    setup: str
    entry: str
    target: str
    k: float = 1.0

    @property
    def id(self) -> str:
        return f"{self.setup}_{self.entry}_{self.target}" + ("" if self.k == 1.0 else f"_k{self.k:g}")

    @property
    def primary(self) -> bool:
        return PRIMARY[self.setup] == (self.entry, self.target)

    def as_dict(self) -> dict:
        return {"study": "research23", "setup": self.setup, "entry": self.entry, "target": self.target, "k": self.k}


def grid() -> list[Variant]:
    """16 variants at k = 1 and the 4 primaries at k = 2 (N = 20, prereg §4)."""
    g = [Variant(s, e, x) for s in SETUPS for e in ENTRIES[s] for x in TARGETS[s]]
    return g + [Variant(s, *PRIMARY[s], k=2.0) for s in SETUPS]


@dataclass
class Signals:
    k: np.ndarray
    side: np.ndarray
    stop: np.ndarray
    vw: np.ndarray
    band: np.ndarray

    def __len__(self) -> int:
        return len(self.k)


# ---------------------------------------------------------------------------- machines (long side)

@njit(cache=True)
def _s1(h, l, c, V, S, ok, pos, buf, k, brk, out_k, out_s):
    m = 0
    run, acc, bt, minlow, bth = 0, False, -1, 0.0, 0.0
    for t in range(len(c)):
        if pos[t] == 0:
            run, acc, bt = 0, False, -1
        if not ok[t]:
            continue
        U = V[t] + k * S[t]
        win = WIN_LO <= pos[t] <= WIN_HI
        run = run + 1 if c[t] > U else 0
        if brk:
            if run == 1 and pos[t] > 0 and win:
                out_k[m], out_s[m] = t, l[t] - buf[t]
                m += 1
            continue
        if c[t] < V[t]:                                     # back deep into value: acceptance lost
            acc, bt = False, -1
            continue
        if bt >= 0:
            if t - bt > PB_WIN:
                bt = -1
            else:
                minlow = min(minlow, l[t])
                if c[t] > bth:
                    if win:
                        out_k[m], out_s[m] = t, minlow - buf[t]
                        m += 1
                    bt = -1
                    continue
        tl = TOL * k * S[t]
        if acc and l[t] <= U + tl and c[t] >= U - tl:       # a (new) backtest candle replaces the old one
            bt, minlow, bth = t, l[t], h[t]
        if not acc and (run >= ACC_RUN or c[t] >= V[t] + (k + ACC_DIST) * S[t]):
            acc = True
    return m


@njit(cache=True)
def _s2(h, l, c, V, S, ok, pos, buf, k, tst, out_k, out_s):
    m = 0
    inside, b, minlow, bh = 0, -1, 0.0, 0.0
    for t in range(len(c)):
        if pos[t] == 0:
            inside, b = 0, -1
        if not ok[t]:
            continue
        U, L = V[t] + k * S[t], V[t] - k * S[t]
        win = WIN_LO <= pos[t] <= WIN_HI
        prev_inside = inside
        inside = inside + 1 if L <= c[t] <= U else 0
        if b >= 0:
            if t - b > TRIG_WIN or c[t] < L:
                b = -1
            else:
                minlow = min(minlow, l[t])
                if c[t] > bh:
                    if win:
                        out_k[m], out_s[m] = t, minlow - buf[t]
                        m += 1
                    b = -1
                    continue
        if b < 0 and prev_inside >= CTX_IN and l[t] <= L + TOL * k * S[t] and c[t] >= L:
            if tst:
                if win:
                    out_k[m], out_s[m] = t, l[t] - buf[t]
                    m += 1
            else:
                b, minlow, bh = t, l[t], h[t]
    return m


@njit(cache=True)
def _s3(h, l, c, V, S, ok, pos, buf, k, brk, out_k, out_s):
    m = 0
    out_run, far, qual, prev_out = 0, False, False, False
    ret, nin, b, minlow, bh = -1, 0, -1, 0.0, 0.0
    for t in range(len(c)):
        if pos[t] == 0:
            out_run, far, qual, prev_out, ret, b = 0, False, False, False, -1, -1
        if not ok[t]:
            continue
        L = V[t] - k * S[t]
        win = WIN_LO <= pos[t] <= WIN_HI
        if c[t] < L:                                        # outside below: (re)start the excursion
            out_run += 1
            far = far or l[t] <= L - OUT_DIST * S[t]
            qual = out_run >= OUT_RUN and far
            ret, b, prev_out = -1, -1, True
            continue
        if prev_out and qual:                               # the return into value
            ret, nin, b = t, 1, -1
            if brk and win:
                out_k[m], out_s[m] = t, l[t] - buf[t]
                m += 1
                ret = -1
        prev_out, out_run, far, qual = False, 0, False, False
        if brk or ret < 0 or t == ret:
            continue
        if t - ret > RTV_LIFE:
            ret, b = -1, -1
            continue
        if b >= 0:
            if t - b > PB_WIN:
                b = -1
            else:
                minlow = min(minlow, l[t])
                if c[t] > bh:
                    if win:
                        out_k[m], out_s[m] = t, minlow - buf[t]
                        m += 1
                    ret, b = -1, -1                         # one trade per return
                    continue
        if b < 0 and nin >= RTV_IN and l[t] <= L + TOL * k * S[t]:
            b, minlow, bh = t, l[t], h[t]
        nin += 1
    return m


@njit(cache=True)
def _s4(h, l, c, V, S, ok, pos, buf, k, tst, out_k, out_s):
    m = 0
    last_ret, was_out, prev_c, prev_V, prev_in, b, minlow, bh = -10**9, False, 0.0, 0.0, False, -1, 0.0, 0.0
    have_prev = False
    for t in range(len(c)):
        if pos[t] == 0:
            last_ret, was_out, have_prev, b = -10**9, False, False, -1
        if not ok[t]:
            continue
        U, L = V[t] + k * S[t], V[t] - k * S[t]
        win = WIN_LO <= pos[t] <= WIN_HI
        now_in = L <= c[t] <= U
        if now_in and have_prev and was_out:
            last_ret = t
        if b >= 0:
            if t - b > TRIG_WIN or c[t] < V[t]:
                b = -1
            else:
                minlow = min(minlow, l[t])
                if c[t] > bh:
                    if win:
                        out_k[m], out_s[m] = t, minlow - buf[t]
                        m += 1
                    b = -1
        elif (have_prev and prev_in and prev_c > prev_V and t - last_ret <= BOUNCE_LOOK
              and l[t] <= V[t] + TOL * k * S[t] and c[t] > V[t]):
            if tst:
                if win:
                    out_k[m], out_s[m] = t, l[t] - buf[t]
                    m += 1
            else:
                b, minlow, bh = t, l[t], h[t]
        was_out, prev_in, prev_c, prev_V, have_prev = not now_in, now_in, c[t], V[t], True
    return m


_MACHINES = {"S1": (_s1, "BRK"), "S2": (_s2, "TST"), "S3": (_s3, "BRK"), "S4": (_s4, "TST")}


@dataclass
class Market:
    """Arrays one asset's signals need (float64)."""
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    V: np.ndarray
    S: np.ndarray
    ok: np.ndarray
    pos: np.ndarray
    buf: np.ndarray

    @classmethod
    def build(cls, t, h, l, c, real, V, S, atr) -> "Market":
        with np.errstate(invalid="ignore"):
            ok = real & np.isfinite(V) & (S > 0) & np.isfinite(atr)
        return cls(h, l, c, V, S, ok, (np.asarray(t, np.int64) % PER_DAY), BUF * np.nan_to_num(atr))


def signals(mk: Market, v: Variant) -> Signals:
    """Long and mirrored short signals of the variant's setup and entry, sorted by row."""
    fn, alt = _MACHINES[v.setup]
    flag = v.entry == alt
    parts = []
    for side in (1, -1):
        if side == 1:
            h, l, c, V = mk.h, mk.l, mk.c, mk.V
        else:
            h, l, c, V = -mk.l, -mk.h, -mk.c, -mk.V
        ok_k = np.empty(len(c), np.int64)
        ok_s = np.empty(len(c))
        n = fn(h, l, c, V, mk.S, mk.ok, mk.pos, mk.buf, v.k, flag, ok_k, ok_s)
        ks = ok_k[:n]
        parts.append((ks, np.full(n, side, np.int8), side * ok_s[:n], mk.V[ks], mk.V[ks] + side * v.k * mk.S[ks]))
    k, side, stop, vw, band = (np.concatenate(x) for x in zip(*parts))
    o = np.lexsort((-side, k))                              # same row: long first (deterministic)
    return Signals(k[o], side[o], stop[o], vw[o], band[o])
