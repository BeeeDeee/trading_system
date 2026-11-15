"""H-0164 Fed purchase-day local supply flow.

60/40 SPY/IEF book. After the close of session d, the buckets the NY Fed Desk has
scheduled (and announced by d) to buy on session d+1 receive a duration-matched
share of the bond sleeve for that one session (open d+1 to open d+2).
"""
import numpy as np
from lab.framework.api import period_starts

PARAMS = {"tilt_fraction": 0.75, "vol_lookback": 63}

DATASET = "nyfed_tsy_operations_1d"
# bucket key -> ETF holding that maturity sector
BUCKETS = (
    ("b0_3", "SHY"),
    ("b3_7", "IEI"),
    ("b7_10", "IEF"),
    ("b10_20", "TLH"),
    ("b20_30", "TLT"),
    ("tips", "TIP"),
)
CORE_EQ = 0.60
SLEEVE = 0.40


def daily_total_returns(ret_co, ret_oc, close):
    """Close-to-close total return and its validity (price on t and on t-1)."""
    r = (1.0 + ret_co.astype(np.float64)) * (1.0 + ret_oc.astype(np.float64)) - 1.0
    has_px = np.isfinite(close)
    valid = np.zeros_like(has_px)
    valid[1:] = has_px[1:] & has_px[:-1]
    valid &= np.isfinite(r)
    return r, valid


def trailing_vol(r, valid, lookback):
    """Annualized std (ddof=1) of the last `lookback` daily returns up to row t.

    NaN unless all `lookback` returns in the window are valid. Works for any T
    (shorter than the window gives all NaN). Uses rows <= t only.
    """
    T = r.shape[0]
    L = int(lookback)
    out = np.full(r.shape, np.nan)
    if L < 2 or T < L:
        return out
    x = np.where(valid, r, 0.0)
    win = np.lib.stride_tricks.sliding_window_view(x, L, axis=0)      # (T-L+1, N, L)
    cnt = np.lib.stride_tricks.sliding_window_view(valid.astype(np.int64), L, axis=0).sum(axis=-1)
    sd = win.std(axis=-1, ddof=1) * np.sqrt(252.0)
    sd = np.where(cnt == L, sd, np.nan)
    out[L - 1:] = sd
    return out


def book_from_arrays(flags, close, sigma, col_spy, col_ief, col_shy, bucket_cols, tilt_fraction):
    """Daily target book (T, N) and the structural key used to detect book changes.

    flags: (T, B) bool, buy_next per bucket (already NaN -> False).
    close: (T, N) prices (NaN = no price); sigma: (T, N) trailing vol (NaN = too short).
    bucket_cols: column index of each bucket's ETF (len B).
    """
    T, N = close.shape
    bucket_cols = np.asarray(bucket_cols, dtype=np.int64)
    B = bucket_cols.shape[0]
    s_ief = sigma[:, col_ief]
    # a bucket counts only if its ETF has a price on d and a full vol window,
    # and IEF itself has a full vol window (the duration-match ratio needs it)
    ok = (np.isfinite(close[:, bucket_cols]) & np.isfinite(sigma[:, bucket_cols])
          & np.isfinite(s_ief)[:, None])
    eff = flags & ok                                              # (T, B) effective B(d)
    k = eff.sum(axis=1).astype(np.float64)                       # |B(d)|
    is_ief_bucket = bucket_cols == col_ief

    T_amt = SLEEVE * float(tilt_fraction)
    share = np.where(k > 0, T_amt / np.maximum(k, 1.0), 0.0)     # T / k
    # only B(d) == {} or {IEF} keep the plain sleeve; otherwise T is split
    has_non_ief = (eff & ~is_ief_bucket[None, :]).any(axis=1)

    w = np.zeros((T, N), dtype=np.float64)
    w[:, col_spy] = CORE_EQ
    w[:, col_ief] = SLEEVE
    key = np.zeros((T, N), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for b in range(B):
            if is_ief_bucket[b]:
                continue                                          # an IEF share stays in IEF
            c = bucket_cols[b]
            on = eff[:, b] & has_non_ief
            ratio = np.minimum(1.0, s_ief / sigma[:, c])
            ratio = np.where(np.isfinite(ratio), ratio, 0.0)
            used = share * ratio
            w[:, col_ief] -= np.where(on, share, 0.0)
            w[:, c] += np.where(on, used, 0.0)
            w[:, col_shy] += np.where(on, share - used, 0.0)
            key[:, c] += np.where(on, 1.0 / np.maximum(k, 1.0), 0.0)
    return w, key


def target_weights(data, params):
    tilt_fraction = float(params["tilt_fraction"])
    lookback = int(params["vol_lookback"])

    col_spy = data.col("SPY")
    col_ief = data.col("IEF")
    col_shy = data.col("SHY")
    bucket_cols = [data.col(etf) for _, etf in BUCKETS]

    T = len(data.dates)
    flags = np.zeros((T, len(BUCKETS)), dtype=bool)
    for b, (key, _) in enumerate(BUCKETS):
        s = np.asarray(data.series[DATASET + ".buy_next." + key], dtype=np.float64)
        flags[:, b] = np.isfinite(s) & (s > 0.5)                # no value for d -> not scheduled

    close = np.asarray(data.close, dtype=np.float64)
    r, valid = daily_total_returns(np.asarray(data.ret_co), np.asarray(data.ret_oc), close)
    sigma = trailing_vol(r, valid, lookback)

    w, key = book_from_arrays(flags, close, sigma, col_spy, col_ief, col_shy, bucket_cols, tilt_fraction)

    # decide where the book's composition changes, and on the first session of each month
    changed = np.zeros(T, dtype=bool)
    if T > 0:
        changed[0] = True
        changed[1:] = np.any(key[1:] != key[:-1], axis=1)
    ps = np.asarray(period_starts(data.dates, "M"), dtype=bool)
    decide = changed | ps
    out = np.where(decide[:, None], w, np.nan)
    return out
