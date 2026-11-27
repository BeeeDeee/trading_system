"""H-0207: delayed dollar response to scheduled US rate news.

Row t is decided after the close of XNYS session t and filled at the open of t+1.
fred_macro (clock next_morning) arrives aligned: row t holds the last DGS2 value
known after the close of t, i.e. DGS2 of session t-1. So the yield news of
release session d = t-1 is first usable at row t = d+1, as the card requires.
"""
import numpy as np

PARAMS = {"hold_days": 5, "z_threshold": 1.0}

TRUSTS = ("FXE", "FXY", "FXB", "FXA", "FXC", "FXF")
CAL_KEYS = ("next_cpi_sessions", "next_empsit_sessions", "next_fomc_sessions")
SCALE_WINDOW = 252
SCALE_MIN_OBS = 126


def _series(data, key, T):
    x = np.asarray(data.series[key], dtype=np.float64).reshape(-1)
    if x.shape[0] != T:
        raise ValueError("series %s has %d rows, expected %d" % (key, x.shape[0], T))
    return x


def yield_changes(v):
    """dy[t] = v[t] - last valid v before t; NaN where v[t] is missing or no earlier value."""
    v = np.asarray(v, dtype=np.float64)
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


def rolling_scale(dy, window=SCALE_WINDOW, min_obs=SCALE_MIN_OBS):
    """Sample std (ddof=1) of the valid changes in the `window` rows ending at t (trailing only)."""
    dy = np.asarray(dy, dtype=np.float64)
    T = dy.shape[0]
    out = np.full(T, np.nan)
    if T == 0:
        return out
    padded = np.concatenate([np.full(window - 1, np.nan), dy])
    win = np.lib.stride_tricks.sliding_window_view(padded, window)  # row t = dy[t-window+1 .. t]
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


def release_session(cal_list):
    """rel[d] = True if on session d-1 any next_*_sessions == 1 (known at the close of d-1)."""
    T = cal_list[0].shape[0]
    hit = np.zeros(T, dtype=bool)
    for c in cal_list:
        hit |= (np.rint(c) == 1.0)            # integer session counts; NaN compares False
    rel = np.zeros(T, dtype=bool)
    rel[1:] = hit[:-1]
    return rel


def event_side(dgs2, cal_list, z_threshold):
    """(T,) side of the event decided after the close of row t (0 = no event)."""
    T = dgs2.shape[0]
    dy = yield_changes(dgs2)                   # row t: DGS2(d) - DGS2(prev valid), d = t-1
    sig = rolling_scale(dy)                    # 252 rows ending at t, i.e. ending at dy_d
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(np.isfinite(sig) & (sig > 0), dy / sig, np.nan)
    rel = release_session(cal_list)            # rel[d]
    rel_d = np.zeros(T, dtype=bool)
    rel_d[1:] = rel[:-1]                       # at row t = d+1: was d a release day?
    ev = rel_d & np.isfinite(z) & (np.abs(z) >= float(z_threshold)) & (dy != 0)
    side = np.zeros(T)
    side[ev] = -np.sign(dy[ev])
    return side


def book_side(side, hold_days):
    """(T,) +1/-1/0: hold the side of the latest event for hold_days rows of decisions.

    Event at row t -> weights on rows t .. t+hold-1 (filled at the opens t+1 .. ), the zero
    target on row t+hold (= d+1+hold) is the exit order. A new event replaces / extends.
    """
    H = int(hold_days)
    T = side.shape[0]
    ar = np.arange(T)
    last = np.maximum.accumulate(np.where(side != 0, ar, -1)) if T else ar
    active = (last >= 0) & (ar - last < H)
    out = np.zeros(T)
    out[active] = side[last[active]]
    return out


def target_weights(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    dgs2 = _series(data, "fred_macro.DGS2", T)
    cal = [_series(data, "macro_release_calendar_1d." + k, T) for k in CAL_KEYS]
    side = event_side(dgs2, cal, params["z_threshold"])
    book = book_side(side, params["hold_days"])
    w = np.zeros((T, N))
    listed = np.asarray(data.listed, dtype=bool)
    close = np.asarray(data.close, dtype=np.float64)
    for name in TRUSTS:
        j = data.col(name)
        has_price = listed[:, j] & np.isfinite(close[:, j])
        w[:, j] = np.where(has_price, book / 6.0, 0.0)
    return w
