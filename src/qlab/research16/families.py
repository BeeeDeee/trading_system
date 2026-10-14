"""The eight signal families of research 16 (prereg §4) and the context of their random null (§6).

`signals(assets, v)` -> {symbol: Cand}. Point-in-time: a candidate at k uses candles <= k only; the last closed
1h bar at candle k is j = (k - 3) // 4 (bar j closes at candle 4j+3), the 4h regime is `Asset.reg15`.
"""

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

from qlab.research16.engine import Asset, Cand, Context

SIGMA_N, MIN_FRAC = 672, 0.75
BB_N, SQZ_N, RSI_N = 20, 720, 14
FUND_N = 270
H4, H48, H72 = 16, 192, 288
PULL_ACTIVE, PULL_STOP_N = 16, 32
CHICAGO = ZoneInfo("America/Chicago")
CME_START = {"BTC": date(2017, 12, 17), "ETH": date(2021, 2, 8)}
FAMILIES = ("ORB", "SQZ", "SHOCK", "FUND", "CME", "PULL", "LEAD", "RANGE")
PARAMS = {"ORB": ("H", (1, 4)), "SQZ": ("Q", (0.1, 0.2)), "SHOCK": ("K", (4, 6)), "FUND": ("Q", (0.90, 0.95)),
          "CME": ("G", (0.01, 0.02)), "PULL": ("L", (30, 40)), "LEAD": ("K", (2, 3)), "RANGE": ("M", (2.0, 2.5))}


@dataclass(frozen=True)
class Variant:
    family: str
    p: float
    side: str                          # "LS", "LONG"; CME "BOTH" (side given by the gap)

    @property
    def id(self) -> str:
        name, _ = PARAMS[self.family]
        val = {"Q": lambda x: round(x * 100), "G": lambda x: round(x * 100), "M": lambda x: round(x * 10)}.get(
            name, int)(self.p)
        return f"{self.family}_{name}{val}" + ("" if self.side == "BOTH" else f"_{self.side}")

    def as_dict(self) -> dict:
        return {"study": "research16", **asdict(self)}


def grid() -> list[Variant]:
    out = []
    for f in FAMILIES:
        for p in PARAMS[f][1]:
            out += [Variant(f, p, "BOTH")] if f == "CME" else [Variant(f, p, s) for s in ("LS", "LONG")]
    return out


# ---------------------------------------------------------------- shared features

def sigma(a: Asset) -> np.ndarray:
    """Std of 15m log returns between real candles over the 672 candles ending at k (>= 75 % valid)."""
    n = len(a.c)
    lr = np.full(n, np.nan)
    lr[1:] = np.log(a.c[1:] / a.c[:-1])
    lr[1:][~(a.real[1:] & a.real[:-1])] = np.nan
    v = np.isfinite(lr)
    x = np.where(v, lr, 0.0)
    cs = lambda y: np.concatenate([[0.0], np.cumsum(y)])
    S0, S1, S2 = cs(v.astype(float)), cs(x), cs(x * x)
    out = np.full(n, np.nan)
    i = np.arange(SIGMA_N - 1, n)
    c0 = S0[i + 1] - S0[i + 1 - SIGMA_N]
    s1, s2 = S1[i + 1] - S1[i + 1 - SIGMA_N], S2[i + 1] - S2[i + 1 - SIGMA_N]
    ok = c0 >= MIN_FRAC * SIGMA_N
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 ** 2 / c0) / (c0 - 1)
    out[i[ok]] = np.sqrt(np.maximum(var[ok], 0.0))
    return out


def on15(x1: np.ndarray, n: int) -> np.ndarray:
    """Value of the last closed 1h bar at every 15m candle."""
    out = np.full(n, np.nan)
    if n > 3:
        j = (np.arange(3, n) - 3) // 4
        out[3:] = x1[j]
    return out


def bollinger(c1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SMA and population std of the last 20 1h closes (NaN unless all 20 bars are valid)."""
    nb = len(c1)
    mid, sd = np.full(nb, np.nan), np.full(nb, np.nan)
    if nb >= BB_N:
        w = swv(c1, BB_N)
        mid[BB_N - 1:], sd[BB_N - 1:] = w.mean(1), w.std(1)
    return mid, sd


def squeeze(c1: np.ndarray, q: float) -> np.ndarray:
    """1h bar j: Bollinger width (4 sd / mid) <= q-quantile of the widths of the 720 previous bars (>= 75 % valid)."""
    mid, sd = bollinger(c1)
    wd = 4 * sd / mid
    nb = len(wd)
    out = np.zeros(nb, bool)
    if nb <= SQZ_N:
        return out
    W = swv(wd, SQZ_N)[:nb - SQZ_N]                   # row i = bars i .. i+719, compared with bar i+720
    for lo in range(0, len(W), 4000):
        blk = W[lo:lo + 4000]
        fin = np.isfinite(blk).sum(1)
        qv = np.full(len(blk), np.nan)
        full, part = fin == SQZ_N, (fin >= MIN_FRAC * SQZ_N) & (fin < SQZ_N)
        if full.any():
            qv[full] = np.quantile(blk[full], q, axis=1)
        if part.any():
            qv[part] = np.nanquantile(blk[part], q, axis=1)
        with np.errstate(invalid="ignore"):
            out[SQZ_N + lo:SQZ_N + lo + len(blk)] = wd[SQZ_N + lo:SQZ_N + lo + len(blk)] <= qv
    return out


def rsi(c1: np.ndarray, n: int = RSI_N) -> np.ndarray:
    """Wilder RSI of 1h closes; a step with a missing bar on either side is skipped (state kept, RSI NaN)."""
    out = np.full(len(c1), np.nan)
    gains, losses, ag, al = [], [], np.nan, np.nan
    for j in range(1, len(c1)):
        d = c1[j] - c1[j - 1]
        if not np.isfinite(d):
            continue
        g, l_ = max(d, 0.0), max(-d, 0.0)
        if not np.isfinite(ag):
            gains.append(g), losses.append(l_)
            if len(gains) == n:
                ag, al = np.mean(gains), np.mean(losses)
            else:
                continue
        else:
            ag, al = (ag * (n - 1) + g) / n, (al * (n - 1) + l_) / n
        out[j] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def move_1h(a: Asset) -> np.ndarray:
    """log(c[k] / c[k-4]) where candles k-4..k are all real, else NaN."""
    n = len(a.c)
    out = np.full(n, np.nan)
    if n > 4:
        ok = swv(a.real, 5).all(1)
        out[4:] = np.where(ok, np.log(a.c[4:] / a.c[:-4]), np.nan)
    return out


def _sides(v: Variant, lg: np.ndarray, sh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sh = sh & (v.side == "LS")
    both = lg & sh
    return lg & ~both, sh & ~both


def _mk(lg, sh, deadline_len, stop_l=None, stop_s=None, tgt_l=None, tgt_s=None, rmult=None) -> Cand:
    """Candidates from long/short masks with per-candle level arrays (None = no level)."""
    kl, ks = np.flatnonzero(lg), np.flatnonzero(sh)
    k = np.concatenate([kl, ks])
    pick = lambda xl, xs: None if xl is None else np.concatenate([xl[kl], xs[ks]])
    return Cand.make(k, np.concatenate([np.ones(len(kl)), -np.ones(len(ks))]), k + deadline_len,
                     pick(stop_l, stop_s), pick(tgt_l, tgt_s), rmult)


# ---------------------------------------------------------------- families

def orb(assets: dict[str, Asset], v: Variant) -> dict[str, Cand]:
    """Range of the first H hours of the UTC day; the first close beyond it (LONG: above it) is the day's only
    candidate; stop = the opposite edge; held at most to the day's last candle (23:45)."""
    H = int(v.p) * 4
    out = {}
    for sym, a in assets.items():
        n = len(a.c)
        k_, s_, st_, dl_ = [], [], [], []
        for d0 in np.flatnonzero(a.t % 96 == 0):
            if d0 + H >= n:
                break
            if not a.real[d0:d0 + H].all():
                continue
            rh, rl = a.h[d0:d0 + H].max(), a.l[d0:d0 + H].min()
            seg = slice(d0 + H, min(d0 + 96, n))
            r = a.real[seg]
            lg, sh = _sides(v, r & (a.c[seg] > rh), r & (a.c[seg] < rl))
            hit = lg | sh
            if hit.any():
                i = int(hit.argmax())
                k_.append(d0 + H + i), s_.append(1 if lg[i] else -1), st_.append(rl if lg[i] else rh)
                dl_.append(d0 + 95)
        out[sym] = Cand.make(k_, s_, dl_, stop=st_)
    return out


def sqz(assets, v):
    """Last closed 1h bar is in a squeeze; 15m close above the upper (below the lower) 2-sd band; stop = middle
    band; target 2R; 48 h."""
    out = {}
    for sym, a in assets.items():
        n = len(a.c)
        mid, sd = bollinger(a.c1)
        sq = on15(squeeze(a.c1, v.p).astype(float), n) == 1
        m15, s15 = on15(mid, n), on15(sd, n)
        with np.errstate(invalid="ignore"):
            lg, sh = _sides(v, a.real & sq & (a.c > m15 + 2 * s15), a.real & sq & (a.c < m15 - 2 * s15))
        out[sym] = _mk(lg, sh, H48, m15, m15, rmult=2.0)
    return out


def shock(assets, v):
    """1h log move below -K * 2 sigma: long, target = c[k-4], stop = entry - half the move, 4 h (short mirrored)."""
    out = {}
    for sym, a in assets.items():
        m, sg = move_1h(a), sigma(a)
        with np.errstate(invalid="ignore"):
            lg, sh = _sides(v, a.real & (m < -v.p * 2 * sg), a.real & (m > v.p * 2 * sg))
        prev = np.full(len(a.c), np.nan)
        prev[4:] = a.c[:-4]
        stop = a.c - 0.5 * (prev - a.c)
        out[sym] = _mk(lg, sh, H4, stop, stop, prev, prev)
    return out


def fund(assets, v):
    """At each funding record: rate > q-quantile of the 270 previous records -> short (LS), rate < (1-q)-quantile
    -> long; held 72 h without stop. Strict inequalities: rates cluster at the 0.01 % baseline (prereg §12)."""
    out = {}
    for sym, a in assets.items():
        n = len(a.c)
        idx = np.flatnonzero(np.isfinite(a.fraw))
        lg, sh = np.zeros(n, bool), np.zeros(n, bool)
        if len(idx) > FUND_N:
            r = a.fraw[idx]
            W = swv(r, FUND_N)[:len(r) - FUND_N]
            hi, lo = np.quantile(W, v.p, axis=1), np.quantile(W, 1 - v.p, axis=1)
            k = idx[FUND_N:]
            lg[k], sh[k] = r[FUND_N:] < lo, r[FUND_N:] > hi
        lg, sh = _sides(v, lg & a.real, sh & a.real)
        out[sym] = _mk(lg, sh, H72)
    return out


def cme_weeks(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Friday close candle F, Sunday reopen candle k, Friday date as datetime64[D]) inside the grid: F is the
    candle ending at Friday 16:00 Chicago, k the candle starting at Sunday 17:00 Chicago."""
    t0, n = int(t[0]), len(t)
    d = datetime.fromtimestamp(t0 * 900, timezone.utc).date()
    d += timedelta(days=(4 - d.weekday()) % 7)
    F, K, D = [], [], []
    while True:
        fc = int(datetime(d.year, d.month, d.day, 16, tzinfo=CHICAGO).timestamp()) // 900 - 1 - t0
        sun = d + timedelta(days=2)
        ko = int(datetime(sun.year, sun.month, sun.day, 17, tzinfo=CHICAGO).timestamp()) // 900 - t0
        if ko >= n:
            break
        if fc >= 0:
            F.append(fc), K.append(ko), D.append(np.datetime64(d))
        d += timedelta(days=7)
    return np.array(F, np.int64), np.array(K, np.int64), np.array(D, "datetime64[D]")


def cme(assets, v):
    """Weekend gap g = c[k] / F - 1 at the CME reopen; |g| >= G -> trade towards F (target), stop half the
    distance beyond the entry, 72 h. Only weekends whose Friday is on/after the listing of the CME future."""
    out = {}
    for sym, a in assets.items():
        F, K, D = cme_weeks(a.t)
        listed = CME_START.get(sym[:3])
        keep = (D >= np.datetime64(listed)) if listed else np.zeros(len(F), bool)
        F, K = F[keep], K[keep]
        ok = a.real[F] & a.real[K]
        F, K = F[ok], K[ok]
        g = a.c[K] / a.c[F] - 1.0
        hit = np.abs(g) >= v.p
        F, K, g = F[hit], K[hit], g[hit]
        out[sym] = Cand.make(K, -np.sign(g), K + H72, stop=a.c[K] + np.sign(g) * 0.5 * np.abs(a.c[K] - a.c[F]),
                             target=a.c[F])
    return out


def pull(assets, v):
    """4h regime up and RSI(14) of a closed 1h bar <= L -> setup active the next 16 candles; entry on a 15m
    confirmation in the up regime; stop = lowest low of the last 32 candles; target 2R; 48 h (short mirrored)."""
    out = {}
    for sym, a in assets.items():
        n = len(a.c)
        rs = rsi(a.c1)
        nb = len(rs)
        reg1 = a.reg15[4 * np.arange(nb) + 3]
        act = {}
        for s, cond in ((1, (rs <= v.p) & (reg1 == 1)), (-1, (rs >= 100 - v.p) & (reg1 == -1))):
            j = np.flatnonzero(cond)
            diff = np.zeros(n + PULL_ACTIVE + 8, np.int64)
            np.add.at(diff, 4 * j + 4, 1)
            np.add.at(diff, 4 * j + 4 + PULL_ACTIVE, -1)
            act[s] = np.cumsum(diff)[:n] > 0
        lo, hi = np.full(n, np.nan), np.full(n, np.nan)
        if n >= PULL_STOP_N:
            lo[PULL_STOP_N - 1:] = swv(np.where(a.real, a.l, np.inf), PULL_STOP_N).min(1)
            hi[PULL_STOP_N - 1:] = swv(np.where(a.real, a.h, -np.inf), PULL_STOP_N).max(1)
        lg, sh = _sides(v, a.real & act[1] & a.up & (a.reg15 == 1), a.real & act[-1] & a.dn & (a.reg15 == -1))
        out[sym] = _mk(lg, sh, H48, lo, hi, rmult=2.0)
    return out


def lead(assets, v):
    """BTC 1h move |rB| >= K * 2 sigma(BTC) while ETH lags (rE / rB < 0.5) -> ETH in the direction of rB, 4 h,
    no stop. Only ETH trades."""
    btc = next((s for s in assets if s.startswith("BTC")), None)
    eth = next((s for s in assets if s.startswith("ETH")), None)
    out = {s: Cand.empty() for s in assets}
    if btc is None or eth is None:
        return out
    B, E = assets[btc], assets[eth]
    assert len(B.t) == len(E.t) and B.t[0] == E.t[0]
    rb, re, sg = move_1h(B), move_1h(E), sigma(B)
    with np.errstate(invalid="ignore", divide="ignore"):
        big = (np.abs(rb) >= v.p * 2 * sg) & (re / rb < 0.5) & E.real & B.real
    lg, sh = _sides(v, big & (rb > 0), big & (rb < 0))
    out[eth] = _mk(lg, sh, H4)
    return out


def range_(assets, v):
    """4h regime neutral; 15m close below the lower 1h band (m sd): long, target = middle band, stop the same
    distance below the entry (1R), 4 h (short mirrored at the upper band)."""
    out = {}
    for sym, a in assets.items():
        n = len(a.c)
        mid, sd = bollinger(a.c1)
        m15, s15 = on15(mid, n), on15(sd, n)
        neutral = a.real & (a.reg15 == 0)
        with np.errstate(invalid="ignore"):
            lg, sh = _sides(v, neutral & (a.c < m15 - v.p * s15), neutral & (a.c > m15 + v.p * s15))
        stop = 2 * a.c - m15
        out[sym] = _mk(lg, sh, H4, stop, stop, m15, m15)
    return out


SIGNALS = {"ORB": orb, "SQZ": sqz, "SHOCK": shock, "FUND": fund, "CME": cme, "PULL": pull, "LEAD": lead,
           "RANGE": range_}


def signals(assets: dict[str, Asset], v: Variant) -> dict[str, Cand]:
    return SIGNALS[v.family](assets, v)


def context(a: Asset, v: Variant) -> Context:
    """Candles where the random null may enter for this family (prereg §6)."""
    r = a.real.copy()
    if v.family == "ORB":
        tod = a.t % 96
        m = r & (tod >= 4 * int(v.p))
        return Context(m, m, day_end=np.arange(len(a.t)) - tod + 95)
    if v.family == "FUND":
        m = r & (a.t % 32 == 0)
        return Context(m, m)
    if v.family == "CME":
        return Context(r, r, same_entries=True)
    if v.family == "PULL":
        return Context(r & (a.reg15 == 1), r & (a.reg15 == -1))
    if v.family == "RANGE":
        m = r & (a.reg15 == 0)
        return Context(m, m)
    return Context(r, r)
