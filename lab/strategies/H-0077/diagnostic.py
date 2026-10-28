"""H-0077 mechanism event: a liquid national muni CEF whose k-session gap to beta * MUB has z <= -z_entry,
not already in an open event (an event at row s blocks the same fund on rows s+1..s+4). Side +1.

Same signal definition as strategy.py (duplicated here so the file only imports numpy).
"""
import numpy as np

CEFS = (
    "NUV", "NEA", "NVG", "NAD", "NZF", "NXP", "NMI", "MYI", "MQY", "MYD", "MVF", "MVT", "MHD",
    "BFK", "BLE", "BKN", "BYM", "MUI", "EVN", "EIM", "PML", "PMF", "PMX", "VKQ", "VMO", "VGM",
    "IQI", "KTF", "MMU", "MFM", "LEO", "DSM",
)
ANCHOR = "MUB"

HOLD = 5
BETA_LOOKBACK = 250
BETA_STEP = 5
SIGMA_LOOKBACK = 250
LIQ_WINDOW = 63
MIN_DOLLAR_VOLUME = 1.0e6
BETA_MIN = 0.5
BETA_MAX = 1.75


def _shift(x, s):
    if s == 0:
        return x.copy()
    out = np.full(x.shape, np.nan)
    if s < x.shape[0]:
        out[s:] = x[:-s]
    return out


def _rolling_sum(x, w):
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
    acc = np.zeros(x.shape)
    for j in range(n_obs):
        acc = acc + _shift(x, j * step)
    return acc


def _rolling_median(x, w):
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


def _signal(data, params):
    k = int(params["formation_days"])
    z_entry = float(params["z_entry"])

    fund_cols = np.array([data.col(name) for name in CEFS], dtype=int)
    m = data.col(ANCHOR)
    cols = np.concatenate([fund_cols, np.array([m])])

    listed = np.asarray(data.listed, dtype=bool)[:, cols]
    tradable = np.asarray(data.tradable, dtype=bool)[:, cols]
    ret_co = np.asarray(data.ret_co, dtype=float)[:, cols]
    ret_oc = np.asarray(data.ret_oc, dtype=float)[:, cols]
    dv = np.asarray(data.dollar_volume, dtype=float)[:, cols]

    missing = listed & ~(np.isfinite(ret_co) & np.isfinite(ret_oc))
    if missing.any():
        rows, cc = np.nonzero(missing)
        raise ValueError(
            "missing return data for listed instrument %s on row %d" % (data.instruments[cols[cc[0]]], rows[0])
        )

    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log1p(r)
    lst_f = listed[:, :-1]
    lst_m = listed[:, -1]
    joint = lst_f & lst_m[:, None]
    lr_f = np.where(joint, lr[:, :-1], np.nan)
    lr_mj = np.where(joint, lr[:, -1][:, None], np.nan)

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

    rk_f = np.expm1(_rolling_sum(lr_f, k))
    rk_m = np.expm1(_rolling_sum(lr_mj, k))
    gap = rk_f - beta * rk_m

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

    dv_f = dv[:, :-1]
    dv_f = np.where(lst_f, np.where(np.isfinite(dv_f), dv_f, 0.0), np.nan)
    med_dv = _rolling_median(dv_f, LIQ_WINDOW)
    eligible = lst_f & tradable[:, :-1] & np.isfinite(z) & np.isfinite(beta)
    eligible &= np.where(np.isfinite(med_dv), med_dv, -1.0) >= MIN_DOLLAR_VOLUME
    if data.universe is not None:
        eligible &= np.asarray(data.universe, dtype=bool)[:, fund_cols]
    event = eligible & (np.where(np.isfinite(z), z, 0.0) <= -z_entry)
    return fund_cols, event


def events(data, params):
    fund_cols, event = _signal(data, params)
    t_len = len(data.dates)
    n_f = len(fund_cols)
    mask_f = np.zeros((t_len, n_f), dtype=bool)
    last = np.full(n_f, -HOLD - 1)
    for t in range(t_len):
        fire = event[t] & ((t - last) >= HOLD)      # not already in an open event
        if fire.any():
            mask_f[t] = fire
            last[fire] = t
    mask = np.zeros((t_len, len(data.instruments)), dtype=bool)
    mask[:, fund_cols] = mask_f                      # MUB is never an event
    return mask, None
