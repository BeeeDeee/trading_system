"""H-0077 Muni closed-end fund discount-shock reversal (MUB-hedged event slots, hold 5 sessions).

Decision after the close of row t from rows 0..t only; the engine fills at the open of t+1.
"""
import numpy as np

PARAMS = {"formation_days": 5, "z_entry": 2.0}

# card universe: national muni CEFs (events) and the open-end muni ETF MUB (anchor and hedge only)
CEFS = (
    "NUV", "NEA", "NVG", "NAD", "NZF", "NXP", "NMI", "MYI", "MQY", "MYD", "MVF", "MVT", "MHD",
    "BFK", "BLE", "BKN", "BYM", "MUI", "EVN", "EIM", "PML", "PMF", "PMX", "VKQ", "VMO", "VGM",
    "IQI", "KTF", "MMU", "MFM", "LEO", "DSM",
)
ANCHOR = "MUB"

SLOT_WEIGHT = 0.07        # long weight per slot
MAX_SLOTS = 5             # at most 5 open slots
HOLD = 5                  # slot held 5 sessions (decision rows t..t+4 hold, row t+5 closes)
BETA_LOOKBACK = 250       # sessions for the beta regression
BETA_STEP = 5             # non-overlapping 5-session returns -> 50 observations
SIGMA_LOOKBACK = 250      # overlapping k-session gaps ending t-1 .. t-250
LIQ_WINDOW = 63           # median dollar volume window
MIN_DOLLAR_VOLUME = 1.0e6
BETA_MIN = 0.5
BETA_MAX = 1.75


def _shift(x, s):
    """Value of row t-s at row t (s >= 0), NaN for the first s rows."""
    if s == 0:
        return x.copy()
    out = np.full(x.shape, np.nan)
    if s < x.shape[0]:
        out[s:] = x[:-s]
    return out


def _rolling_sum(x, w):
    """Sum of rows t-w+1..t; NaN if the window is incomplete or contains a NaN. Prefix-stable (cumsum)."""
    t_len = x.shape[0]
    out = np.full(x.shape, np.nan)
    if t_len < w:
        return out
    valid = np.isfinite(x)
    z = np.where(valid, x, 0.0)
    pad = np.zeros((1,) + x.shape[1:])
    c = np.concatenate([pad, np.cumsum(z, axis=0)], axis=0)
    n_bad = np.concatenate([pad, np.cumsum((~valid).astype(float), axis=0)], axis=0)
    s = c[w:] - c[:-w]
    bad = (n_bad[w:] - n_bad[:-w]) > 0
    out[w - 1:] = np.where(bad, np.nan, s)
    return out


def _strided_sum(x, n_obs, step):
    """Sum of x[t], x[t-step], ..., x[t-(n_obs-1)*step]; NaN if any term is missing."""
    acc = np.zeros(x.shape)
    for j in range(n_obs):
        acc = acc + _shift(x, j * step)
    return acc


def _rolling_median(x, w):
    """Median of rows t-w+1..t per column; NaN if the window is incomplete or holds a NaN."""
    t_len, n = x.shape
    out = np.full(x.shape, np.nan)
    for j in range(n):
        lagged = np.stack([_shift(x[:, j], s) for s in range(w)], axis=1)
        ok = np.isfinite(lagged).all(axis=1)
        med = np.full(t_len, np.nan)
        if ok.any():
            med[ok] = np.median(lagged[ok], axis=1)
        out[:, j] = med
    return out


def compute_signal(data, params):
    """Returns dict with fund_cols, mub col, beta (T,F), z (T,F), eligible (T,F), event (T,F)."""
    k = int(params["formation_days"])
    z_entry = float(params["z_entry"])

    fund_cols = np.array([data.col(name) for name in CEFS], dtype=int)   # raises if a fund is missing
    m = data.col(ANCHOR)                                                  # raises if MUB is missing
    cols = np.concatenate([fund_cols, np.array([m])])

    listed = np.asarray(data.listed, dtype=bool)[:, cols]
    tradable = np.asarray(data.tradable, dtype=bool)[:, cols]
    ret_co = np.asarray(data.ret_co, dtype=float)[:, cols]
    ret_oc = np.asarray(data.ret_oc, dtype=float)[:, cols]
    dv = np.asarray(data.dollar_volume, dtype=float)[:, cols]

    # missing return data on a listed row must raise (card), never be skipped
    missing = listed & ~(np.isfinite(ret_co) & np.isfinite(ret_oc))
    if missing.any():
        rows, cc = np.nonzero(missing)
        raise ValueError(
            "missing return data for listed instrument %s on row %d" % (data.instruments[cols[cc[0]]], rows[0])
        )

    # daily total return (ret_co and ret_oc compounded), as log return; NaN where not jointly listed with MUB
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log1p(r)
    lr_f = lr[:, :-1]
    lr_m = lr[:, -1]
    lst_f = listed[:, :-1]
    lst_m = listed[:, -1]
    joint = lst_f & lst_m[:, None]
    lr_f = np.where(joint, lr_f, np.nan)
    lr_mj = np.where(joint, lr_m[:, None], np.nan)       # MUB return on the fund's joint rows (T, F)

    # (1) beta: OLS slope of non-overlapping 5-session returns over the last 250 sessions (50 obs), clipped
    r5_f = np.expm1(_rolling_sum(lr_f, BETA_STEP))
    r5_m = np.expm1(_rolling_sum(lr_mj, BETA_STEP))
    n_b = BETA_LOOKBACK // BETA_STEP
    sx = _strided_sum(r5_m, n_b, BETA_STEP)
    sy = _strided_sum(r5_f, n_b, BETA_STEP)
    sxx = _strided_sum(r5_m * r5_m, n_b, BETA_STEP)
    sxy = _strided_sum(r5_m * r5_f, n_b, BETA_STEP)
    var_x = sxx / n_b - (sx / n_b) ** 2
    cov_xy = sxy / n_b - (sx / n_b) * (sy / n_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_raw = np.where(var_x > 0, cov_xy / var_x, np.nan)
    beta = np.clip(beta_raw, BETA_MIN, BETA_MAX)

    # (2) gap: k-session compounded returns, fund minus beta * MUB over the same k sessions
    rk_f = np.expm1(_rolling_sum(lr_f, k))
    rk_m = np.expm1(_rolling_sum(lr_mj, k))
    gap = rk_f - beta * rk_m

    # (3) scale: std of the same k-session gap (current beta) over overlapping windows ending t-1..t-250
    n_s = SIGMA_LOOKBACK
    a = _shift(_rolling_sum(rk_f, n_s), 1)
    b = _shift(_rolling_sum(rk_m, n_s), 1)
    c = _shift(_rolling_sum(rk_f * rk_f, n_s), 1)
    d = _shift(_rolling_sum(rk_m * rk_m, n_s), 1)
    e = _shift(_rolling_sum(rk_f * rk_m, n_s), 1)
    mean_g = (a - beta * b) / n_s
    var_g = (c - 2.0 * beta * e + beta * beta * d) / n_s - mean_g ** 2
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(np.maximum(var_g, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, gap / sigma, np.nan)

    # eligibility: listed and tradable today, full history (finite z), 63-session median dollar volume >= 1e6
    dv_f = dv[:, :-1]
    dv_f = np.where(lst_f, np.where(np.isfinite(dv_f), dv_f, 0.0), np.nan)   # listed without a bar = 0 traded
    med_dv = _rolling_median(dv_f, LIQ_WINDOW)
    eligible = lst_f & tradable[:, :-1] & np.isfinite(z) & np.isfinite(beta)
    with np.errstate(invalid="ignore"):
        eligible &= np.where(np.isfinite(med_dv), med_dv, -1.0) >= MIN_DOLLAR_VOLUME
    if data.universe is not None:
        eligible &= np.asarray(data.universe, dtype=bool)[:, fund_cols]

    with np.errstate(invalid="ignore"):
        event = eligible & (np.where(np.isfinite(z), z, 0.0) <= -z_entry)

    return {
        "fund_cols": fund_cols,
        "mub": m,
        "beta": beta,
        "z": z,
        "eligible": eligible,
        "event": event,
        "listed_f": lst_f,
        "tradable_f": tradable[:, :-1],
    }


def target_weights(data, params):
    sig = compute_signal(data, params)
    fund_cols = sig["fund_cols"]
    m = sig["mub"]
    beta = sig["beta"]
    z = sig["z"]
    event = sig["event"]
    ok_f = sig["listed_f"] & sig["tradable_f"]

    t_len = len(data.dates)
    n_f = len(fund_cols)
    w = np.zeros((t_len, len(data.instruments)))

    active = np.zeros(n_f, dtype=bool)
    start = np.zeros(n_f, dtype=int)
    slot_beta = np.zeros(n_f)
    for t in range(t_len):
        # (5) slots opened at decision s are held through the close of s+5: close at decision s+5
        active &= (t - start) < HOLD
        # a fund that is not listed or not tradable during its slot is closed (fills at the first tradable
        # open, or is paid out on delisting); its hedge is removed at the same decision
        active &= ok_f[t]
        # (4) events on funds without an open slot; most negative z first while slots are free
        cand = event[t] & ~active
        free = MAX_SLOTS - int(active.sum())
        if free > 0 and cand.any():
            idx = np.flatnonzero(cand)
            order = np.argsort(z[t, idx], kind="stable")
            take = idx[order[:free]]
            active[take] = True
            start[take] = t
            slot_beta[take] = beta[t, take]
        if active.any():
            w[t, fund_cols[active]] = SLOT_WEIGHT
            w[t, m] = -SLOT_WEIGHT * float(slot_beta[active].sum())
    return w
