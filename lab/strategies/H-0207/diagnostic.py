"""H-0207 mechanism test: mark the six trusts after the close of d+1 for a large DGS2 move
on a scheduled CPI / payrolls / FOMC session d. Same event logic as strategy.py (copied)."""
import numpy as np

TRUSTS = ("FXE", "FXY", "FXB", "FXA", "FXC", "FXF")
CAL_KEYS = ("next_cpi_sessions", "next_empsit_sessions", "next_fomc_sessions")
SCALE_WINDOW = 252
SCALE_MIN_OBS = 126


def _series(data, key, T):
    x = np.asarray(data.series[key], dtype=np.float64).reshape(-1)
    if x.shape[0] != T:
        raise ValueError("series %s has %d rows, expected %d" % (key, x.shape[0], T))
    return x


def _yield_changes(v):
    T = v.shape[0]
    fin = np.isfinite(v)
    idx = np.where(fin, np.arange(T), -1)
    last = np.maximum.accumulate(idx) if T else idx
    prev = np.full(T, -1, dtype=np.int64)
    if T > 1:
        prev[1:] = last[:-1]
    dy = np.full(T, np.nan)
    ok = fin & (prev >= 0)
    dy[ok] = v[ok] - v[prev[ok]]
    return dy


def _rolling_scale(dy, window=SCALE_WINDOW, min_obs=SCALE_MIN_OBS):
    T = dy.shape[0]
    out = np.full(T, np.nan)
    if T == 0:
        return out
    padded = np.concatenate([np.full(window - 1, np.nan), dy])
    win = np.lib.stride_tricks.sliding_window_view(padded, window)
    fin = np.isfinite(win)
    n = fin.sum(axis=1)
    xs = np.where(fin, win, 0.0)
    s = xs.sum(axis=1)
    good = n >= max(int(min_obs), 2)
    mean = np.where(good, s / np.maximum(n, 1), 0.0)
    dev = np.where(fin, win - mean[:, None], 0.0)
    var = (dev * dev).sum(axis=1) / np.maximum(n - 1, 1)
    out[good] = np.sqrt(var[good])
    return out


def _event_side(dgs2, cal_list, z_threshold):
    T = dgs2.shape[0]
    dy = _yield_changes(dgs2)
    sig = _rolling_scale(dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(np.isfinite(sig) & (sig > 0), dy / sig, np.nan)
    hit = np.zeros(T, dtype=bool)
    for c in cal_list:
        hit |= (np.rint(c) == 1.0)            # integer session counts; NaN compares False
    rel = np.zeros(T, dtype=bool)
    rel[1:] = hit[:-1]
    rel_d = np.zeros(T, dtype=bool)
    rel_d[1:] = rel[:-1]
    ev = rel_d & np.isfinite(z) & (np.abs(z) >= float(z_threshold)) & (dy != 0)
    side = np.zeros(T)
    side[ev] = -np.sign(dy[ev])
    return side


def events(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    dgs2 = _series(data, "fred_macro.DGS2", T)
    cal = [_series(data, "macro_release_calendar_1d." + k, T) for k in CAL_KEYS]
    s = _event_side(dgs2, cal, params["z_threshold"])
    ok = np.asarray(data.tradable, dtype=bool) & np.asarray(data.listed, dtype=bool)
    if data.universe is not None:
        ok = ok & np.asarray(data.universe, dtype=bool)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N))
    for name in TRUSTS:
        j = data.col(name)
        m = (s != 0) & ok[:, j]
        mask[:, j] = m
        side[:, j] = np.where(m, s, 0.0)
    return mask, side
