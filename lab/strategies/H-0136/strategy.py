import numpy as np

# H-0136: quarter-long earnings-reaction drift, LIQ-500 long-only, held release to release.

PARAMS = {"z_threshold": 2.0}

VOL_WINDOW = 60      # sessions of daily abnormal return for the scale s (ending at t-2)
VOL_MIN_VALID = 40   # minimum valid days in that window
MAX_HOLD = 80        # sessions after entry (or last confirming release) without a new release -> sell
MIN_SLOTS = 20       # below 20 held names each gets 1/20, the rest is cash


def daily_returns(ret_co, ret_oc, listed, close):
    """Close-to-close total return (ret_co, ret_oc compounded) and its validity mask.

    A day's return is valid when the stock is listed with a price on that row and on the
    previous row (so the first listed row, without a previous close, is not valid)."""
    r = (1.0 + ret_co.astype(np.float64)) * (1.0 + ret_oc.astype(np.float64)) - 1.0
    ok = listed.astype(bool) & np.isfinite(close) & np.isfinite(r)
    valid = np.zeros_like(ok)
    valid[1:] = ok[1:] & ok[:-1]
    r = np.where(valid, r, np.nan)
    return r, valid


def _member_mean(x, mask):
    """Row-wise equal-weighted mean of x over cells in mask (NaN when the row has none)."""
    cnt = mask.sum(axis=1)
    tot = np.where(mask, x, 0.0).sum(axis=1)
    out = np.full(x.shape[0], np.nan)
    has = cnt > 0
    out[has] = tot[has] / cnt[has]
    return out


def reaction_z(r, valid, uni, ea):
    """Reaction z on every release cell (step 1 of the card); NaN elsewhere.

    Returns (z, ev): ev marks releases that count (ea == 1, LIQ-500 member at t, two-session
    abnormal return a and scale s both computable, s > 0); z is defined where ev."""
    T, N = r.shape
    # daily abnormal return: stock minus equal-weighted mean of the members with a valid return
    mkt1 = _member_mean(r, valid & uni)
    d = r - mkt1[:, None]                        # NaN where r is invalid or no member mean

    # two-session compounded return over t-1 and t
    r2 = np.full((T, N), np.nan)
    v2 = np.zeros((T, N), dtype=bool)
    if T > 1:
        v2[1:] = valid[1:] & valid[:-1]
        r2[1:] = (1.0 + np.nan_to_num(r[1:])) * (1.0 + np.nan_to_num(r[:-1])) - 1.0
    r2 = np.where(v2, r2, np.nan)
    mkt2 = _member_mean(r2, v2 & uni)
    a = r2 - mkt2[:, None]

    rel = (ea == 1.0) & uni                      # NaN flags (before the first filing) are not releases
    tt, jj = np.nonzero(rel)
    z = np.full((T, N), np.nan)
    ev = np.zeros((T, N), dtype=bool)
    if tt.size == 0:
        return z, ev
    # window of VOL_WINDOW sessions ending at t-2: rows t-61 .. t-2
    rows = tt[:, None] - (VOL_WINDOW + 1) + np.arange(VOL_WINDOW)[None, :]
    inside = rows >= 0
    win = d[np.clip(rows, 0, None), jj[:, None]]
    win = np.where(inside, win, np.nan)
    n = np.isfinite(win).sum(axis=1)
    mu = np.where(n > 0, np.nansum(win, axis=1) / np.maximum(n, 1), np.nan)
    dev = np.where(np.isfinite(win), win - mu[:, None], 0.0)
    var = (dev * dev).sum(axis=1) / np.maximum(n - 1, 1)
    s = np.sqrt(var)
    ae = a[tt, jj]
    ok = (n >= VOL_MIN_VALID) & np.isfinite(ae) & (s > 0)
    z[tt[ok], jj[ok]] = ae[ok] / (s[ok] * np.sqrt(2.0))
    ev[tt[ok], jj[ok]] = True
    return z, ev


def build_book(z, ev, uni, listed, thr):
    """Steps (2)-(4): enter on z >= thr, stay while each next release confirms, exit on a
    failing release, after MAX_HOLD sessions without a release, or when leaving the universe."""
    T, N = z.shape
    w = np.zeros((T, N))
    held = np.zeros(N, dtype=bool)
    age = np.zeros(N, dtype=np.int64)
    for t in range(T):
        age[held] += 1
        e = ev[t]
        good = e & (z[t] >= thr)
        bad = e & ~good
        held &= ~bad                     # next release not strong enough -> sold at t+1 open
        age[good] = 0                    # new entry or confirmation: clock (re)starts at t
        held |= good
        held &= age < MAX_HOLD           # 80 sessions held after entry without a release
        held &= uni[t] & listed[t]       # left LIQ-500 (or delisted) -> sold
        k = int(held.sum())
        if k:
            w[t, held] = 1.0 / max(k, MIN_SLOTS)
    return w


def target_weights(data, params):
    thr = float(params["z_threshold"])
    uni = data.universe
    if uni is None:
        raise ValueError("H-0136 needs the point-in-time LIQ-500 universe")
    uni = np.asarray(uni, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    r, valid = daily_returns(np.asarray(data.ret_co), np.asarray(data.ret_oc), listed,
                             np.asarray(data.close, dtype=np.float64))
    z, ev = reaction_z(r, valid, uni, ea)
    return build_book(z, ev, uni, listed, thr)    # every session is a decision row
