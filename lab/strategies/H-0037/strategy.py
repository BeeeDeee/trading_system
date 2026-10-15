"""H-0037 Twin-ETF premium reversion.

Pairs (A, B) fixed by the card. After the close of day t (rows 0..t only):
  x_t = ln TR_A - ln TR_B, TR = cumprod((1+ret_co)(1+ret_oc))
  m_t, s_t = mean and sample std (ddof=1) of x over rows t-L..t-1
  z_t = (x_t - m_t) / s_t, d_t = x_t - m_t
  trigger: |z_t| >= z_entry and |d_t| >= 25 bps and both legs have a bar on t and on the L previous rows
           and both legs have a 20-session median dollar volume >= 10 M USD.
  direction: d_t > 0 -> short A / long B; d_t < 0 -> long A / short B.
  hold: 5 sessions; same-direction trigger restarts the clock, opposite trigger flips; the exit decision is
        taken on the first row >= trigger + 5 on which both legs have a bar.
  weights: n open pairs -> +1/(2n) cheap leg, -1/(2n) rich leg.
"""
import numpy as np

PARAMS = {"lookback_days": 20, "z_entry": 2.0}

PAIRS = (("HYG", "JNK"), ("AGG", "BND"), ("TIP", "SCHP"), ("EEM", "IEMG"), ("EFA", "IEFA"))

GAP_FLOOR = 0.0025          # fixed 25 bps floor on |d_t| (card: not a grid parameter)
MIN_DOLLAR_VOLUME = 1.0e7   # 10 M USD, 20-session median (card: fixed)
DV_WINDOW = 20
HOLD_SESSIONS = 5


def has_bar(data):
    """(T, N) bool: the instrument has a real bar on row t (listed, tradable open, a price at the close)."""
    listed = np.asarray(data.listed, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    close = np.asarray(data.close, dtype=float)
    return listed & tradable & np.isfinite(close)


def log_total_return(data):
    """(T, N) ln of the total-return index built from ret_co and ret_oc (0 where there is no trading)."""
    rco = np.asarray(data.ret_co, dtype=float)
    roc = np.asarray(data.ret_oc, dtype=float)
    rco = np.where(np.isfinite(rco), rco, 0.0)
    roc = np.where(np.isfinite(roc), roc, 0.0)
    return np.cumsum(np.log1p(rco) + np.log1p(roc), axis=0)


def trailing_median(x, n):
    """(T, N) median of the finite values of x over rows t-n+1..t (NaN if none). Point-in-time."""
    x = np.asarray(x, dtype=float)
    T, N = x.shape
    pad = np.full((n - 1, N), np.nan)
    xp = np.concatenate([pad, x], axis=0)
    win = np.lib.stride_tricks.sliding_window_view(xp, n, axis=0)   # (T, N, n), window t = rows t-n+1..t
    srt = np.sort(win, axis=-1)                                        # NaN sorted to the end
    cnt = np.sum(np.isfinite(win), axis=-1)
    lo = np.clip((cnt - 1) // 2, 0, n - 1)
    hi = np.clip(cnt // 2, 0, n - 1)
    a = np.take_along_axis(srt, lo[..., None], axis=-1)[..., 0]
    b = np.take_along_axis(srt, hi[..., None], axis=-1)[..., 0]
    med = 0.5 * (a + b)
    return np.where(cnt > 0, med, np.nan)


def pair_signals(ltr, bar, dv, ia, ib, L, z_entry):
    """For one pair: trigger (T,) bool, d (T,) float, both-legs-bar (T,) bool."""
    T = ltr.shape[0]
    x = ltr[:, ia] - ltr[:, ib]
    m = np.full(T, np.nan)
    s = np.full(T, np.nan)
    if T > L:
        win = np.lib.stride_tricks.sliding_window_view(x, L)   # window k = rows k..k+L-1
        m[L:] = win.mean(axis=-1)[: T - L]                      # row t uses rows t-L..t-1
        s[L:] = win.std(axis=-1, ddof=1)[: T - L]
    d = x - m
    ok_s = np.isfinite(s) & (s > 0)
    z = np.where(ok_s, d / np.where(ok_s, s, 1.0), np.nan)

    both = bar[:, ia] & bar[:, ib]
    cb = np.concatenate([[0], np.cumsum(both.astype(np.int64))])
    full = np.zeros(T, dtype=bool)
    if T > L:
        t_idx = np.arange(L, T)
        full[L:] = (cb[t_idx + 1] - cb[t_idx - L]) == (L + 1)   # bars on t-L..t

    med = trailing_median(dv[:, [ia, ib]], DV_WINDOW)
    liquid = np.all(np.isfinite(med) & (med >= MIN_DOLLAR_VOLUME), axis=1)

    with np.errstate(invalid="ignore"):
        trig = (np.isfinite(z) & (np.abs(z) >= z_entry) & np.isfinite(d) & (np.abs(d) >= GAP_FLOOR)
                & full & liquid)
    return trig, d, both


def hold_state(trig, d, both, hold=HOLD_SESSIONS):
    """(T,) direction of the A leg after the close of each row (+1 long A, -1 short A, 0 flat)."""
    T = trig.shape[0]
    out = np.zeros(T)
    cur = 0.0
    last = -1
    for t in range(T):
        if trig[t]:
            cur = -1.0 if d[t] > 0 else 1.0
            last = t
        elif cur != 0.0 and t - last >= hold and both[t]:
            cur = 0.0
        out[t] = cur
    return out


def target_weights(data, params):
    L = int(params["lookback_days"])
    z_entry = float(params["z_entry"])
    T = len(data.dates)
    N = len(data.instruments)

    ltr = log_total_return(data)
    bar = has_bar(data)
    dv = np.asarray(data.dollar_volume, dtype=float)

    dirs = []
    for a, b in PAIRS:
        ia = data.col(a)
        ib = data.col(b)
        trig, d, both = pair_signals(ltr, bar, dv, ia, ib, L, z_entry)
        dirs.append((ia, ib, hold_state(trig, d, both)))

    n_open = np.zeros(T)
    for _, _, dr in dirs:
        n_open += (dr != 0).astype(float)
    scale = np.where(n_open > 0, 1.0 / (2.0 * np.maximum(n_open, 1.0)), 0.0)

    w = np.zeros((T, N))
    for ia, ib, dr in dirs:
        w[:, ia] = dr * scale
        w[:, ib] = -dr * scale
    return w
