"""MTF 4h / 1h / 15m: regime, setups, confirmation, stop/target trade simulator and benchmarks (prereg §4-6).

Point-in-time contract: a 15m candle index k is "the close of candle k". Everything decided at k uses candles
<= k only; 1h/4h bars count once closed (1h bar j closes at candle 4j+3, 4h bar b at 16b+15). A trade decided
at k is filled at close[k] (candle k+delay), evaluated on candles e+1.. and pays funding at settlement
candles (t % 32 == 0) it is held through. Setups only ever look forward through candles that exist at k.
"""

from dataclasses import asdict, dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

COST = 7.5e-4
FUNDING_DEFAULT = 1e-4                 # per 8 h before perp funding data exist
EMA_FAST, EMA_SLOW = 20, 50
WARMUP_4H = 150                        # valid 4h closes before the regime may be non-zero
ACTIVE = 16                            # candles a setup stays active after its 1h close
CONFIRM_N = 4
MAX_HOLD = 192                         # 48 h
R_MIN, R_MAX = 0.0015, 0.05
MIN_FRAC = 0.75
PER_DAY, FUNDING_STEP = 96, 32
STOP, TARGET, TIME, END = 0, 1, 2, 3


@dataclass(frozen=True)
class Config:
    kind: str                          # "BRK" or "SWP"
    N: int
    R: float
    side: str                          # "LS" or "LONG"

    @property
    def id(self) -> str:
        return f"MTF_{self.kind}_N{self.N}_R{int(self.R * 10)}_{self.side}"

    def as_dict(self) -> dict:
        return {"study": "research15", **asdict(self)}


def grid() -> list[Config]:
    return [Config(k, N, R, side) for k in ("BRK", "SWP") for N in (12, 48) for R in (1.5, 3.0)
            for side in ("LS", "LONG")]


# ---------------------------------------------------------------- higher timeframes

def aggregate(o, h, l, c, real, k: int):
    """k-candle bars (o, h, l, c); NaN where fewer than 75 % of the candles are real. The tail that does not
    fill a bar is dropped (it has not closed)."""
    nb = len(c) // k
    m = nb * k
    r = real[:m].reshape(nb, k)
    ok = r.sum(1) >= int(np.ceil(MIN_FRAC * k))
    first, last = r.argmax(1), k - 1 - r[:, ::-1].argmax(1)
    rows = np.arange(nb)
    O = o[:m].reshape(nb, k)[rows, first]
    C = c[:m].reshape(nb, k)[rows, last]
    H = np.where(r, h[:m].reshape(nb, k), -np.inf).max(1)
    L = np.where(r, l[:m].reshape(nb, k), np.inf).min(1)
    return [np.where(ok, x, np.nan) for x in (O, H, L, C)]


def ema(x: np.ndarray, n: int) -> np.ndarray:
    """EMA seeded at the first finite value; missing values carry the previous EMA."""
    a = 2.0 / (n + 1)
    out = np.full(len(x), np.nan)
    prev = np.nan
    for i, v in enumerate(x):
        if np.isfinite(v):
            prev = v if not np.isfinite(prev) else prev + a * (v - prev)
        out[i] = prev
    return out


def regime_4h(c4: np.ndarray) -> np.ndarray:
    e20, e50 = ema(c4, EMA_FAST), ema(c4, EMA_SLOW)
    with np.errstate(invalid="ignore"):
        up, dn = (c4 > e50) & (e20 > e50), (c4 < e50) & (e20 < e50)
    r = np.where(up, 1, np.where(dn, -1, 0))
    r[np.cumsum(np.isfinite(c4)) < WARMUP_4H] = 0
    return r.astype(np.int8)


def regime_15m(n: int, reg4: np.ndarray) -> np.ndarray:
    """Regime of the last closed 4h bar at each 15m candle (candle 16b+15 is the first to see bar b)."""
    out = np.zeros(n, np.int8)
    if n > 15:
        out[15:] = np.repeat(reg4, 16)[:n - 15]
    return out


def setups_1h(h1, l1, c1, N: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Direction (+1/-1/0) and stop anchor of the setup on each closed 1h bar (prereg §4).

    A bar that would be both a long and a short setup (SWP wider than the whole window) gives none."""
    nb = len(c1)
    d = np.zeros(nb, np.int8)
    if nb <= N:
        return d, np.full(nb, np.nan)
    HH, LL = np.full(nb, np.nan), np.full(nb, np.nan)
    HH[N:] = swv(h1, N)[:nb - N].max(1)
    LL[N:] = swv(l1, N)[:nb - N].min(1)
    with np.errstate(invalid="ignore"):
        if kind == "BRK":
            lg, sh = c1 > HH, c1 < LL
        else:
            lg, sh = (l1 < LL) & (c1 > LL), (h1 > HH) & (c1 < HH)
    both = lg & sh
    d[lg & ~both], d[sh & ~both] = 1, -1
    return d, np.where(d == 1, l1, np.where(d == -1, h1, np.nan))


def confirmations(h, l, c, real, n: int = CONFIRM_N) -> tuple[np.ndarray, np.ndarray]:
    """Long / short 15m confirmation at k: close beyond the extreme of the n previous candles (all real)."""
    T = len(c)
    up, dn = np.zeros(T, bool), np.zeros(T, bool)
    if T <= n:
        return up, dn
    allreal = swv(real, n + 1).all(1)
    pmax, pmin = swv(h, n)[:T - n].max(1), swv(l, n)[:T - n].min(1)
    with np.errstate(invalid="ignore"):
        up[n:] = allreal & (c[n:] > pmax)
        dn[n:] = allreal & (c[n:] < pmin)
    return up, dn


def funding_rate(t: np.ndarray, funding: np.ndarray) -> np.ndarray:
    """Funding paid by a long at each candle: data where present; 1 bp at 00/08/16 UTC before the first record."""
    f = np.where(np.isfinite(funding), funding, 0.0)
    first = int(np.argmax(np.isfinite(funding))) if np.isfinite(funding).any() else len(funding)
    f[(np.arange(len(t)) < first) & (t % FUNDING_STEP == 0)] = FUNDING_DEFAULT
    return f


@dataclass
class Asset:
    t: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    real: np.ndarray
    fund: np.ndarray
    reg15: np.ndarray
    up: np.ndarray
    dn: np.ndarray
    h1: np.ndarray
    l1: np.ndarray
    c1: np.ndarray
    month: np.ndarray

    @classmethod
    def build(cls, d: dict, funding: np.ndarray | None = None) -> "Asset":
        """`d`: t, o, h, l, c, real (+ optional funding aligned to t); t[0] must be a multiple of 16."""
        assert d["t"][0] % 16 == 0
        funding = d.get("funding", np.full(len(d["c"]), np.nan)) if funding is None else funding
        o, h, l, c, real = (d[k] for k in ("o", "h", "l", "c", "real"))
        _, _, _, c4 = aggregate(o, h, l, c, real, 16)
        _, h1, l1, c1 = aggregate(o, h, l, c, real, 4)
        up, dn = confirmations(h, l, c, real)
        month = (d["t"] * 900).astype("datetime64[s]").astype("datetime64[M]").astype(np.int64)
        return cls(d["t"], o, h, l, c, real, funding_rate(d["t"], funding), regime_15m(len(c), regime_4h(c4)),
                   up, dn, h1, l1, c1, month)

    def setups(self, kind: str, N: int):
        return setups_1h(self.h1, self.l1, self.c1, N, kind)


def align_funding(spot_t: np.ndarray, perp: dict) -> np.ndarray:
    """Perp funding placed on the spot candle grid (NaN where the perp has no data)."""
    out = np.full(len(spot_t), np.nan)
    i = perp["t"] - spot_t[0]
    ok = (i >= 0) & (i < len(spot_t))
    out[i[ok]] = perp["funding"][ok]
    return out


# ---------------------------------------------------------------- trades

@dataclass
class Book:
    entries: np.ndarray
    sides: np.ndarray
    risk: np.ndarray
    exits: np.ndarray
    reason: np.ndarray
    ret: np.ndarray                    # net return of each trade
    net: np.ndarray                    # per-candle net return of the sleeve (full exposure)


def _exit(a: Asset, e: int, s: int, S: float, T: float, end: int) -> tuple[int, float, int]:
    """First candle after e whose range touches stop or target (stop first); else the time stop."""
    last = min(e + MAX_HOLD, end - 1)
    sl = slice(e + 1, last + 1)
    if s == 1:
        st, tg = a.l[sl] <= S, a.h[sl] >= T
    else:
        st, tg = a.h[sl] >= S, a.l[sl] <= T
    hit = (st | tg) & a.real[sl]
    if hit.any():
        i = int(hit.argmax())
        j = e + 1 + i
        if st[i]:
            return j, (min(a.o[j], S) if s == 1 else max(a.o[j], S)), STOP
        return j, (max(a.o[j], T) if s == 1 else min(a.o[j], T)), TARGET
    return last, a.c[last], (TIME if last == e + MAX_HOLD else END)


def _bar_returns(a: Asset, e: int, x: int, price: float, s: int, cost: float, use_funding: bool) -> np.ndarray:
    cur = a.c[e + 1:x + 1].copy()
    cur[-1] = price
    hr = s * (cur / a.c[e:x] - 1.0)
    if use_funding:
        hr = hr - s * a.fund[e + 1:x + 1]
    hr[0] -= cost
    hr[-1] -= cost
    return hr


def _book(rec: list, net: np.ndarray) -> Book:
    if not rec:
        z = np.zeros(0)
        return Book(z.astype(np.int64), z.astype(np.int8), z, z.astype(np.int64), z.astype(np.int8), z, net)
    e, s, r, x, why, ret = zip(*rec)
    return Book(np.array(e, np.int64), np.array(s, np.int8), np.array(r), np.array(x, np.int64),
                np.array(why, np.int8), np.array(ret), net)


def trade_book(a: Asset, cfg: Config, start: int, end: int, confirm: bool = True, use_regime: bool = True,
               delay: int = 0, cost: float = COST, use_funding: bool = True,
               allowed: np.ndarray | None = None) -> Book:
    """Sequential trades on [start, end) (prereg §4-5). `confirm=False` enters at the setup close (B_NO15),
    `use_regime=False` drops the 4h filter (B_NO4H), `allowed` (per candle) restricts where a trade may be
    entered (basket membership); an open trade runs on regardless."""
    d, anchor = a.setups(cfg.kind, cfg.N)
    j = np.flatnonzero(d)
    ks, sd, an = 4 * j + 3, d[j], anchor[j]
    keep = (ks >= start) & (ks < end - 1)
    if cfg.side == "LONG":
        keep &= sd == 1
    if use_regime:
        keep &= a.reg15[ks] == sd
    ks, sd, an = ks[keep], sd[keep], an[keep]
    n = len(a.c)
    net = np.zeros(n)
    rec, blocked = [], -1
    for i in range(len(ks)):
        k0, s, anc = int(ks[i]), int(sd[i]), float(an[i])
        if k0 <= blocked:                                   # created while a position was open
            continue
        if confirm:
            nxt = int(ks[i + 1]) if i + 1 < len(ks) else n  # the next setup replaces this one
            sl = slice(k0 + 1, min(k0 + ACTIVE, nxt, end - 1) + 1)
            bad = a.real[sl] & ((a.l[sl] <= anc) if s == 1 else (a.h[sl] >= anc))
            m = int(bad.argmax()) if bad.any() else len(bad)    # the candle that breaks the anchor ends the setup
            cand = (a.up if s == 1 else a.dn)[sl][:m]
            if use_regime:
                cand = cand & (a.reg15[sl][:m] == s)
            if not cand.any():
                continue
            k = k0 + 1 + int(cand.argmax())
        else:
            k = k0
        e = k + delay
        if e > end - 2 or (allowed is not None and not allowed[e]):
            continue
        if delay and (a.real[k + 1:e + 1] & ((a.l[k + 1:e + 1] <= anc) if s == 1 else (a.h[k + 1:e + 1] >= anc))).any():
            continue
        p = float(a.c[e])
        risk = s * (p - anc) / p
        if not (R_MIN <= risk <= R_MAX):                    # the setup is spent either way
            continue
        S, T = anc, p + s * cfg.R * abs(p - anc)
        x, price, why = _exit(a, e, s, S, T, end)
        hr = _bar_returns(a, e, x, price, s, cost, use_funding)
        net[e + 1:x + 1] += hr
        rec.append((e, s, risk, x, why, float(np.prod(1.0 + hr) - 1.0)))
        blocked = x
    return _book(rec, net)


def random_book(a: Asset, cfg: Config, ref: Book, start: int, end: int, rng, cost: float = COST,
                use_funding: bool = True) -> Book:
    """B_RND (prereg §6): per (month, side) the same number of trades as `ref`, entered at random real candles
    with the same 4h regime and no open position; risk drawn from the reference trades of that asset and side."""
    need: dict[tuple[int, int], int] = {}
    for e, s in zip(ref.entries, ref.sides):
        key = (int(a.month[e]), int(s))
        need[key] = need.get(key, 0) + 1
    risks = {s: ref.risk[ref.sides == s] for s in (1, -1)}
    n = len(a.c)
    occ, net, rec = np.zeros(n, bool), np.zeros(n), []
    lo_all, hi_all = max(start, 0), end - 1                 # entries in [start, end-2]
    for (m, s), cnt in sorted(need.items()):
        lo = max(int(np.searchsorted(a.month, m, "left")), lo_all)
        hi = min(int(np.searchsorted(a.month, m, "right")), hi_all)
        ok = lo + np.flatnonzero(a.real[lo:hi] & (a.reg15[lo:hi] == s))
        cand = rng.permutation(ok)
        draw = rng.choice(risks[s], size=len(cand))
        got = 0
        for e, r in zip(cand, draw):
            if occ[e]:
                continue
            e = int(e)
            p = float(a.c[e])
            anc = p * (1.0 - s * r)
            x, price, why = _exit(a, e, s, anc, p + s * cfg.R * abs(p - anc), end)
            if occ[e:x + 1].any():
                continue
            hr = _bar_returns(a, e, x, price, s, cost, use_funding)
            net[e + 1:x + 1] += hr
            occ[e:x + 1] = True
            rec.append((e, s, float(r), x, why, float(np.prod(1.0 + hr) - 1.0)))
            got += 1
            if got == cnt:
                break
    rec.sort()
    return _book(rec, net)


def position_returns(a: Asset, pos: np.ndarray, cost: float = COST, use_funding: bool = True) -> np.ndarray:
    """Per-candle net returns of a position decided at each close and held over the next candle (B_4H)."""
    r = np.zeros(len(a.c))
    r[1:] = a.c[1:] / a.c[:-1] - 1.0
    held = np.zeros(len(a.c))
    held[1:] = pos[:-1]
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    return held * r - (held * a.fund if use_funding else 0.0) - turn * cost


def daily(t: np.ndarray, x: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum of 15m returns per UTC day over candles [start, end)."""
    u, inv = np.unique(t[start:end] // PER_DAY, return_inverse=True)
    return (u * 86400).astype("datetime64[s]").astype("datetime64[D]"), np.bincount(inv, x[start:end])
