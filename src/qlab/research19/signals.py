"""Signals of research 19 (PLAN §4.0, §4.3, §4.4). Row t uses data up to the close of t only.

All functions take `P` (T, N) total-return index and `valid` (T, N) bool (listed and tradable). A window is
usable only if every price in it is valid, otherwise the output is NaN (state functions: state 0, stop NaN).
`tom_window` is the one function that looks at the calendar ahead (trading days are known in advance).
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _swv


def _all_valid(valid: np.ndarray, w: int) -> np.ndarray:
    """ok[t] = valid[t-w+1 .. t] all True (False while t < w-1)."""
    c = np.vstack([np.zeros((1, valid.shape[1]), dtype=np.int64), np.cumsum(~valid, axis=0, dtype=np.int64)])
    ok = np.zeros(valid.shape, dtype=bool)
    ok[w - 1:] = (c[w:] - c[:-w]) == 0
    return ok


def _roll(P: np.ndarray, valid: np.ndarray, w: int, fn) -> np.ndarray:
    """Rolling reduce over P[t-w+1 .. t] (pad with NaN at the start); invalid prices are masked by the caller."""
    out = np.full(P.shape, np.nan)
    if w <= len(P):
        out[w - 1:] = fn(_swv(P, w, axis=0), axis=-1)
    return out


def donchian_state(P: np.ndarray, valid: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Donchian breakout with trailing channel-midpoint stop (PLAN §4.0). Returns (state 0/1, stop; NaN if flat)."""
    T, N = P.shape
    ok = _all_valid(valid, L + 1)                                   # P[t-L .. t] valid
    hi_in, lo_in = np.where(valid, P, -np.inf), np.where(valid, P, np.inf)
    mx, mn = _roll(hi_in, valid, L, np.max), _roll(lo_in, valid, L, np.min)       # windows ending at t
    U = np.full(P.shape, np.nan)
    U[1:] = mx[:-1]                                                 # max(P[t-L .. t-1]), without day t
    M = (mx + mn) / 2.0                                             # midpoint of P[t-L+1 .. t], with day t
    state, stop = np.zeros((T, N)), np.full((T, N), np.nan)
    s, st = np.zeros(N, dtype=bool), np.full(N, np.nan)
    for t in range(T):
        flat = ~ok[t]
        s[flat], st[flat] = False, np.nan
        act = ok[t]
        enter = act & ~s & (P[t] > U[t])
        hold = act & s
        st[hold] = np.maximum(st[hold], M[t][hold])
        leave = hold & (P[t] < st)
        s[enter], st[enter] = True, M[t][enter]
        s[leave], st[leave] = False, np.nan
        state[t], stop[t] = s, st
    return state, stop


def ensemble(P: np.ndarray, valid: np.ndarray, lookbacks) -> np.ndarray:
    """Mean Donchian state over lookbacks: 0, 1/K, ..., 1."""
    return np.mean([donchian_state(P, valid, L)[0] for L in lookbacks], axis=0)


def realized_vol(P: np.ndarray, valid: np.ndarray, n: int, ann: float) -> np.ndarray:
    """Std (ddof=1) of the last n daily returns up to t times sqrt(ann); NaN unless P[t-n .. t] all valid."""
    r = np.full(P.shape, np.nan)
    r[1:] = P[1:] / P[:-1] - 1.0
    ok = _all_valid(valid, n + 1)
    out = np.full(P.shape, np.nan)
    if n <= len(P) - 1:
        out[n:] = np.std(_swv(r[1:], n, axis=0), axis=-1, ddof=1) * np.sqrt(ann)
    return np.where(ok, out, np.nan)


def sma(P: np.ndarray, valid: np.ndarray, n: int) -> np.ndarray:
    """Mean of P[t-n+1 .. t]; NaN unless all n prices are valid."""
    out = _roll(np.where(valid, P, 0.0), valid, n, np.mean)
    return np.where(_all_valid(valid, n), out, np.nan)


def rsi_wilder(P: np.ndarray, valid: np.ndarray, n: int = 2) -> np.ndarray:
    """Wilder RSI(n). Seed = simple mean of the first n gains/losses; resets across invalid days (needs n+1
    consecutive valid prices again). AL = 0 gives RSI 100."""
    T, N = P.shape
    d = np.zeros(P.shape)
    d[1:] = P[1:] - P[:-1]
    g, l = np.where(valid, np.maximum(d, 0.0), 0.0), np.where(valid, np.maximum(-d, 0.0), 0.0)
    out = np.full(P.shape, np.nan)
    run = np.zeros(N, dtype=np.int64)
    AG, AL = np.full(N, np.nan), np.full(N, np.nan)
    for t in range(T):
        run = np.where(valid[t], run + 1, 0)
        AG[run == 0], AL[run == 0] = np.nan, np.nan
        seed = run == n + 1
        if seed.any():
            AG[seed] = g[t - n + 1:t + 1, seed].mean(axis=0)
            AL[seed] = l[t - n + 1:t + 1, seed].mean(axis=0)
        upd = run > n + 1
        AG[upd] = (AG[upd] * (n - 1) + g[t, upd]) / n
        AL[upd] = (AL[upd] * (n - 1) + l[t, upd]) / n
        have = run >= n + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            rsi = np.where(AL == 0, 100.0, 100.0 - 100.0 / (1.0 + AG / AL))
        out[t] = np.where(have, rsi, np.nan)
    return out


def tom_window(dates: np.ndarray, before: int = 1, after: int = 3) -> np.ndarray:
    """Turn-of-month window over a trading calendar: the last `before` trading days of a month and the first
    `after` of the next. The final calendar day is never flagged as a month end (next month unknown)."""
    m = np.asarray(dates).astype("datetime64[M]").astype(np.int64)
    T = len(m)
    idx = np.arange(T)
    first = np.r_[0, np.flatnonzero(m[1:] != m[:-1]) + 1]               # first trading day of each month
    start = np.repeat(first, np.diff(np.r_[first, T]))
    rank_from_start = idx - start                                      # 0-based
    last = np.r_[first[1:] - 1, T - 1]
    end = np.repeat(last, np.diff(np.r_[first, T]))
    rank_to_end = end - idx
    is_end = (rank_to_end < before) & (end < T - 1)                    # only months followed by another month
    return (rank_from_start < after) | is_end
