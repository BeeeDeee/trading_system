"""Multi-timeframe pullback (prereg §4-6): signal, fixed-hold trade simulator and benchmarks.

Point-in-time contract: every signal value at hour t uses closes up to the close of hour t only. A trade
decided at the close of hour e is filled at close[e], held for bars e+1..e+H and closed at close[e+H]; it
pays funding at settlement instants s with e+1 <= s <= e+H (bar s starts at instant s). A new entry on the
same asset needs t > e+H.
"""

from dataclasses import asdict, dataclass

import numpy as np

COST = 7.5e-4
FUNDING_DEFAULT = 1e-4                 # per 8 h before perp funding data exist
SIGMA_WIN, SIGMA_MIN = 720, 600
SMA_D, SMA_W = 50, 20
WEEK_CLOSE_MOD = 95                    # epoch hour % 168 == 95  <=>  Sunday 23:00 UTC


@dataclass(frozen=True)
class Config:
    L: int
    z0: float
    H: int
    side: str                          # "LS" or "LONG"

    @property
    def id(self) -> str:
        return f"MTF_L{self.L}_z{int(self.z0 * 10)}_H{self.H}_{self.side}"

    def as_dict(self) -> dict:
        return {"study": "research14", **asdict(self)}


def grid() -> list[Config]:
    return [Config(L, z0, H, side) for L in (6, 24) for z0 in (1.5, 2.5) for H in (12, 48)
            for side in ("LS", "LONG")]


# ---------------------------------------------------------------- signal

def _sma_of_closed_bars(close: np.ndarray, is_bar_close: np.ndarray, n: int) -> np.ndarray:
    """At hour t: mean of the last n bar closes at hours <= t (a bar closing at t counts)."""
    idx = np.flatnonzero(is_bar_close)
    bc = close[idx]
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(bc))])
    cn = np.concatenate([[0], np.cumsum(np.isfinite(bc))])
    k = np.cumsum(is_bar_close)                       # number of bar closes at hours <= t
    out = np.full(len(close), np.nan)
    ok = k >= n
    kk = k[ok]
    valid = (cn[kk] - cn[kk - n]) == n
    vals = (cs[kk] - cs[kk - n]) / n
    out[np.flatnonzero(ok)[valid]] = vals[valid]
    return out


def regime(hours: np.ndarray, close: np.ndarray) -> np.ndarray:
    """+1 when daily and weekly trend are up, -1 when both down, 0 otherwise (prereg §4)."""
    d_ma = _sma_of_closed_bars(close, hours % 24 == 23, SMA_D)
    w_ma = _sma_of_closed_bars(close, hours % 168 == WEEK_CLOSE_MOD, SMA_W)
    with np.errstate(invalid="ignore"):
        d, w = np.sign(close - d_ma), np.sign(close - w_ma)
    r = np.where((d == 1) & (w == 1), 1, np.where((d == -1) & (w == -1), -1, 0))
    r[np.isnan(d_ma) | np.isnan(w_ma) | np.isnan(close)] = 0
    return r.astype(np.int8)


def sigma(close: np.ndarray, real: np.ndarray) -> np.ndarray:
    """Std of real-to-real hourly log returns over the trailing 720 h (min 600 valid), up to hour t."""
    lr = np.full(len(close), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        lr[1:] = np.log(close[1:] / close[:-1])
    lr[1:][~(real[1:] & real[:-1])] = np.nan
    ok = np.isfinite(lr)
    x = np.where(ok, lr, 0.0)
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    cn = np.concatenate([[0], np.cumsum(ok)])
    t = np.arange(len(close)) + 1
    lo = np.maximum(t - SIGMA_WIN, 0)
    n = cn[t] - cn[lo]
    s1, s2 = c1[t] - c1[lo], c2[t] - c2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1)
    out = np.sqrt(np.maximum(var, 0.0))
    out[n < SIGMA_MIN] = np.nan
    return out


def zscore(close: np.ndarray, real: np.ndarray, L: int, sig: np.ndarray | None = None) -> np.ndarray:
    """ln(c_t / c_{t-L}) / (sigma_t sqrt L); NaN unless both candles are real (prereg §4)."""
    sig = sigma(close, real) if sig is None else sig
    z = np.full(len(close), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        z[L:] = np.log(close[L:] / close[:-L]) / (sig[L:] * np.sqrt(L))
    z[L:][~(real[L:] & real[:-L])] = np.nan
    z[:L] = np.nan
    return z


def signals(hours, close, real, cfg: Config, reg=None, sig=None) -> np.ndarray:
    """Desired side at hour t (+1 long, -1 short, 0 none), before the no-overlap rule."""
    reg = regime(hours, close) if reg is None else reg
    z = zscore(close, real, cfg.L, sig)
    with np.errstate(invalid="ignore"):
        s = np.where((reg == 1) & (z <= -cfg.z0), 1, 0)
        if cfg.side == "LS":
            s = np.where((reg == -1) & (z >= cfg.z0), -1, s)
    s[~real] = 0
    return s.astype(np.int8)


def signals_nohtf(close, real, cfg: Config, sig=None) -> np.ndarray:
    z = zscore(close, real, cfg.L, sig)
    with np.errstate(invalid="ignore"):
        s = np.where(z <= -cfg.z0, 1, np.where(z >= cfg.z0, -1, 0))
    s[~real] = 0
    return s.astype(np.int8)


# ---------------------------------------------------------------- simulation

def pick_entries(sig: np.ndarray, H: int, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Sequential no-overlap entries in [start, end): entry e blocks hours up to e+H."""
    cand = np.flatnonzero(sig[start:end]) + start
    ent, nxt = [], -1
    for e in cand:
        if e > nxt:
            ent.append(e)
            nxt = e + H
    ent = np.asarray(ent, dtype=np.int64)
    return ent, sig[ent].astype(np.int8)


def funding_rate(hours: np.ndarray, funding: np.ndarray) -> np.ndarray:
    """Funding paid by a long at each hour: data where present; 1 bp at 00/08/16 UTC before the first record."""
    f = np.where(np.isfinite(funding), funding, 0.0)
    first = int(np.argmax(np.isfinite(funding))) if np.isfinite(funding).any() else len(funding)
    pre = (np.arange(len(hours)) < first) & (hours % 8 == 0)
    f[pre] = FUNDING_DEFAULT
    return f


def simulate(close, fund, entries, sides, H: int, end: int, cost: float = COST,
             delay: int = 0, use_funding: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Hourly net returns of one sleeve (full exposure) and net return of each trade.

    Exits are forced at the last hour end-1 of the phase. `delay` shifts the fill by that many hours.
    """
    r = np.zeros(len(close))
    r[1:] = close[1:] / close[:-1] - 1.0
    net = np.zeros(len(close))
    trades = np.zeros(len(entries))
    for i, (e0, s) in enumerate(zip(entries, sides)):
        e = e0 + delay
        x = min(e0 + H, end - 1)
        if e >= x:
            trades[i] = np.nan
            continue
        hr = s * r[e + 1:x + 1]
        if use_funding:
            hr = hr - s * fund[e + 1:x + 1]
        hr[0] -= cost
        hr[-1] -= cost
        net[e + 1:x + 1] += hr
        trades[i] = np.prod(1.0 + hr) - 1.0
    return net, trades


def simulate_positions(close, fund, pos_decided: np.ndarray, cost: float = COST) -> np.ndarray:
    """Hourly net returns of a position decided at each close and held over the next bar (B_HTF)."""
    r = np.zeros(len(close))
    r[1:] = close[1:] / close[:-1] - 1.0
    held = np.zeros(len(close))
    held[1:] = pos_decided[:-1]
    turn = np.abs(np.diff(np.concatenate([[0.0], pos_decided])))
    net = held * r - held * fund
    net -= turn * cost                                # traded at the close of the deciding hour
    return net


def daily(hours: np.ndarray, hourly: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum of hourly returns per UTC day over hours [start, end)."""
    day = hours[start:end] // 24
    u, inv = np.unique(day, return_inverse=True)
    return (u * 86400).astype("datetime64[s]").astype("datetime64[D]"), np.bincount(inv, hourly[start:end])


# ---------------------------------------------------------------- random-entry null

def random_entries(rng, reg, real, sig_ok, hours, entries, sides, H: int, start: int, end: int):
    """Same count per (month, side) as the strategy, random hours in the matching regime (prereg §6)."""
    month = (hours * 3600).astype("datetime64[s]").astype("datetime64[M]").astype(np.int64)
    need: dict[tuple[int, int], int] = {}
    for e, s in zip(entries, sides):
        need[(month[e], int(s))] = need.get((month[e], int(s)), 0) + 1
    occupied = np.zeros(len(hours) + H + 2, bool)
    out_e, out_s = [], []
    base = np.arange(start, end)
    for (m, s), n in sorted(need.items()):
        ok = base[(month[start:end] == m) & (reg[start:end] == s) & real[start:end] & sig_ok[start:end]]
        got = 0
        for e in rng.permutation(ok):
            if not occupied[e:e + H + 1].any():
                occupied[e:e + H + 1] = True
                out_e.append(e)
                out_s.append(s)
                got += 1
                if got == n:
                    break
    order = np.argsort(out_e)
    return np.asarray(out_e, np.int64)[order], np.asarray(out_s, np.int8)[order]
