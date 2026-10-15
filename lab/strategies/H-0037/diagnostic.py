"""H-0037 mechanism diagnostic: every twin-pair trigger day is an event on both legs.

Rich leg side -1, cheap leg side +1. The trigger is the strategy's (|z_t| >= z_entry, |d_t| >= 25 bps, both legs
with a bar on t and on the L previous rows, both legs 20-session median dollar volume >= 10 M USD).
Same code as strategy.py (duplicated because a strategy file may only import numpy and lab.framework).
"""
import numpy as np

PAIRS = (("HYG", "JNK"), ("AGG", "BND"), ("TIP", "SCHP"), ("EEM", "IEMG"), ("EFA", "IEFA"))

GAP_FLOOR = 0.0025
MIN_DOLLAR_VOLUME = 1.0e7
DV_WINDOW = 20


def _has_bar(data):
    listed = np.asarray(data.listed, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    close = np.asarray(data.close, dtype=float)
    return listed & tradable & np.isfinite(close)


def _log_total_return(data):
    rco = np.asarray(data.ret_co, dtype=float)
    roc = np.asarray(data.ret_oc, dtype=float)
    rco = np.where(np.isfinite(rco), rco, 0.0)
    roc = np.where(np.isfinite(roc), roc, 0.0)
    return np.cumsum(np.log1p(rco) + np.log1p(roc), axis=0)


def _trailing_median(x, n):
    x = np.asarray(x, dtype=float)
    T, N = x.shape
    pad = np.full((n - 1, N), np.nan)
    xp = np.concatenate([pad, x], axis=0)
    win = np.lib.stride_tricks.sliding_window_view(xp, n, axis=0)
    srt = np.sort(win, axis=-1)
    cnt = np.sum(np.isfinite(win), axis=-1)
    lo = np.clip((cnt - 1) // 2, 0, n - 1)
    hi = np.clip(cnt // 2, 0, n - 1)
    a = np.take_along_axis(srt, lo[..., None], axis=-1)[..., 0]
    b = np.take_along_axis(srt, hi[..., None], axis=-1)[..., 0]
    med = 0.5 * (a + b)
    return np.where(cnt > 0, med, np.nan)


def _pair_trigger(ltr, bar, dv, ia, ib, L, z_entry):
    T = ltr.shape[0]
    x = ltr[:, ia] - ltr[:, ib]
    m = np.full(T, np.nan)
    s = np.full(T, np.nan)
    if T > L:
        win = np.lib.stride_tricks.sliding_window_view(x, L)
        m[L:] = win.mean(axis=-1)[: T - L]
        s[L:] = win.std(axis=-1, ddof=1)[: T - L]
    d = x - m
    ok_s = np.isfinite(s) & (s > 0)
    z = np.where(ok_s, d / np.where(ok_s, s, 1.0), np.nan)

    both = bar[:, ia] & bar[:, ib]
    cb = np.concatenate([[0], np.cumsum(both.astype(np.int64))])
    full = np.zeros(T, dtype=bool)
    if T > L:
        t_idx = np.arange(L, T)
        full[L:] = (cb[t_idx + 1] - cb[t_idx - L]) == (L + 1)

    med = _trailing_median(dv[:, [ia, ib]], DV_WINDOW)
    liquid = np.all(np.isfinite(med) & (med >= MIN_DOLLAR_VOLUME), axis=1)

    with np.errstate(invalid="ignore"):
        trig = (np.isfinite(z) & (np.abs(z) >= z_entry) & np.isfinite(d) & (np.abs(d) >= GAP_FLOOR)
                & full & liquid)
    return trig, d


def events(data, params):
    L = int(params["lookback_days"])
    z_entry = float(params["z_entry"])
    T = len(data.dates)
    N = len(data.instruments)

    ltr = _log_total_return(data)
    bar = _has_bar(data)
    dv = np.asarray(data.dollar_volume, dtype=float)

    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N))
    for a, b in PAIRS:
        ia = data.col(a)
        ib = data.col(b)
        trig, d = _pair_trigger(ltr, bar, dv, ia, ib, L, z_entry)
        sa = np.where(d > 0, -1.0, 1.0)          # A rich (d > 0) -> A short side
        mask[:, ia] |= trig
        mask[:, ib] |= trig
        side[:, ia] = np.where(trig, sa, side[:, ia])
        side[:, ib] = np.where(trig, -sa, side[:, ib])

    if data.universe is not None:
        mask &= np.asarray(data.universe, dtype=bool)
    mask &= bar
    side = np.where(mask, side, 0.0)
    return mask, side
