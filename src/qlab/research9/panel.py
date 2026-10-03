"""Binance spot daily klines -> qlab.data.panel.Panel (prereg §3).

ret_co = open_t / close_{t-1} - 1 (0 on the first row of a listing), ret_oc = close_t / open_t - 1.
A gap of more than MAX_GAP days, or the end of the data before the panel end, is a delisting: the position
is paid out at the open of the day after the last row at the last close - 2 %. A symbol that comes back
after such a gap starts as a new listing. Shorter gaps: not tradable, value carried at the last close.
"""

from datetime import date
from pathlib import Path

import numpy as np

from qlab.data.panel import Panel
from qlab.research8 import data as D8

MAX_GAP = 7
DELIST_HAIRCUT = -0.02


def build(raw: Path, symbols: list[str], start: date, end: date) -> tuple[Panel, dict[str, np.ndarray], list[str]]:
    days = np.arange(np.datetime64(start, "D"), np.datetime64(end, "D") + 1)
    T, N = len(days), len(symbols)
    f = lambda v=0.0, dt=float: np.full((T, N), v, dtype=dt)  # noqa: E731
    ret_co, ret_oc, close, qv, opn = f(), f(), f(np.nan), f(np.nan), f(np.nan)
    listed, tradable, delisting = f(False, bool), f(False, bool), f(False, bool)
    i0 = days[0]
    for j, s in enumerate(symbols):
        k = D8.read_klines(raw / "spot_1d" / s)
        if not k.height:
            continue
        d = (k["date"].to_numpy().astype("datetime64[D]") - i0).astype(int)
        keep = (d >= 0) & (d < T) & (k["open"].to_numpy() > 0) & (k["close"].to_numpy() > 0)
        d, o, c, v = d[keep], k["open"].to_numpy()[keep], k["close"].to_numpy()[keep], k["quote_volume"].to_numpy()[keep]
        if not len(d):
            continue
        listed[d, j] = tradable[d, j] = True
        opn[d, j], close[d, j], qv[d, j] = o, c, v
        ret_oc[d, j] = c / o - 1
        prev_c = np.r_[np.nan, c[:-1]]
        gap = np.r_[MAX_GAP + 1, np.diff(d)]
        cont = gap <= MAX_GAP
        ret_co[d[cont], j] = o[cont] / prev_c[cont] - 1
        # delisting rows: after a long gap and after the last row (if before the panel end)
        ends = list(d[np.r_[~cont[1:], False]]) + ([d[-1]] if d[-1] < T - 1 else [])
        for e in ends:
            if e + 1 < T:
                delisting[e + 1, j] = True
                ret_co[e + 1, j] = DELIST_HAIRCUT
    p = Panel(days, np.arange(N, dtype=np.int64), ret_co, ret_oc, tradable, listed | delisting, delisting, close, qv)
    return p, {"open": opn, "qv": qv}, symbols


def cost_rate(qv: np.ndarray, mult: float = 1.0) -> np.ndarray:
    """Per-side cost for an order filled at the open of day t: fee + slippage tier of the 30-day mean
    quote volume over days t-30..t-1 (prereg §4)."""
    v = np.nan_to_num(qv)
    cs = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)])
    T = v.shape[0]
    mean = np.zeros_like(v)
    for t in range(1, T):
        lo = max(0, t - 30)
        mean[t] = (cs[t] - cs[lo]) / (t - lo)
    slip = np.select([mean >= 1e9, mean >= 2e8, mean >= 5e7], [0.0002, 0.0005, 0.0010], 0.0025)
    return (0.0010 + slip) * mult
