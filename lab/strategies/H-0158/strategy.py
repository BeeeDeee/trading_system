"""H-0158 - credit-market news through debt rollover need, learned.

LIQ-500 ex banks, dollar-neutral weekly book from a LightGBM score on
net rollover need, short-term share of debt, book leverage, 63-session
volatility and a date-level credit-specific Baa-Aaa shock z.
"""
import numpy as np

from lab.framework.api import only_on, total_return_index
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"n_estimators": 100, "top": 0.2}

# fixed by the card (not grid parameters)
HORIZON = 5               # label horizon (ml.horizon_days), also the framework purge
RETRAIN_EVERY = 126       # ml.retrain_every_days
SHOCK_SESSIONS = 5        # change of the spread / market return window
OLS_WINDOW = 252          # trailing sessions strictly before d for the OLS
VOL_WINDOW = 63           # realized volatility window
VOL_MIN_DAYS = 50         # minimum valid daily returns in the window
WINSOR = 3.0              # cross-sectional z-scores clipped at +/- 3
MIN_NAMES = 100           # at least 100 scored members, else flat
G_FLOOR = 0.5             # |z| below this: flat
GBM_FIXED = {
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 200,
    "subsample": 0.8,
    "colsample": 0.8,
}

SPREAD_HI = "fred_moody_yields_1d.DBAA"
SPREAD_LO = "fred_moody_yields_1d.DAAA"


# ---------------------------------------------------------------- calendar
def weekday(dates):
    """Monday = 0 ... Sunday = 6, from the date of each row alone."""
    days = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    return (days + 3) % 7          # 1970-01-01 was a Thursday


def weekly_decision_days(dates):
    """Last session of the calendar week, approximated by weekday arithmetic:
    a row is a decision row when the next weekday (np.busday_offset, which
    ignores exchange holidays) lies in a later week, i.e. the row is a Friday.
    A week whose Friday is an exchange holiday has no decision row."""
    return weekday(dates) == 4


# ---------------------------------------------------------------- credit shock
def market_return(idx, listed, members, k=SHOCK_SESSIONS):
    """Equal-weight mean k-session total return of the members on each row."""
    T, N = idx.shape
    m = np.full(T, np.nan)
    if T <= k:
        return m
    now = idx[k:].astype(np.float64)
    then = idx[:-k].astype(np.float64)
    ok = (np.isfinite(now) & np.isfinite(then) & (then > 0)
          & listed[k:] & listed[:-k] & members[k:])
    r = np.where(ok, now / np.where(ok, then, 1.0) - 1.0, 0.0)
    cnt = ok.sum(axis=1)
    tot = r.sum(axis=1)
    m[k:] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return m


def spread_change(s, k=SHOCK_SESSIONS):
    ds = np.full(len(s), np.nan)
    if len(s) > k:
        ds[k:] = s[k:] - s[:-k]
    return ds


def credit_shock_z(ds, m, window=OLS_WINDOW):
    """z_d = (ds_d - a - b m_d) / std(residuals), OLS of ds on m over the
    `window` sessions strictly before d; all of them must have values."""
    ds = np.asarray(ds, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    T = len(ds)
    z = np.full(T, np.nan)
    valid = np.isfinite(ds) & np.isfinite(m)
    run = np.concatenate([[0], np.cumsum(valid)])
    for t in range(window, T):
        if not valid[t] or run[t] - run[t - window] < window:
            continue
        x = m[t - window:t]
        y = ds[t - window:t]
        xm = x.mean()
        ym = y.mean()
        xc = x - xm
        sxx = float(xc @ xc)
        if not sxx > 0:
            continue
        b = float(xc @ (y - ym)) / sxx
        a = ym - b * xm
        sd = float(np.std(y - a - b * x))
        if not sd > 0:
            continue
        z[t] = (ds[t] - a - b * m[t]) / sd
    return z


def shock_from_data(data):
    s = (np.asarray(data.series[SPREAD_HI], dtype=np.float64).reshape(-1)
         - np.asarray(data.series[SPREAD_LO], dtype=np.float64).reshape(-1))
    uni = universe_of(data)
    idx = total_return_index(data)
    m = market_return(idx, np.asarray(data.listed, dtype=bool), uni)
    return credit_shock_z(spread_change(s), m)


def size_factor(z):
    """g = min(1, |z|) if |z| >= 0.5, else 0; NaN z -> 0 (flat)."""
    az = np.abs(np.asarray(z, dtype=np.float64))
    ok = np.isfinite(az) & (az >= G_FLOOR)
    return np.where(ok, np.minimum(1.0, np.where(ok, az, 0.0)), 0.0)


# ---------------------------------------------------------------- stock features
def bank_mask(deposits):
    """Banks / deposit takers: latest usable SF1 quarter has deposits > 0."""
    d = np.asarray(deposits, dtype=np.float64)
    return np.isfinite(d) & (d > 0)


def balance_sheet_features(assets, debt, debtc, cash):
    """(a) (debtc - cashneq) / assets, (b) debtc / debt (0 when debt = 0),
    (c) debt / assets. NaN where any input is missing or assets <= 0."""
    assets = np.asarray(assets, dtype=np.float64)
    debt = np.asarray(debt, dtype=np.float64)
    debtc = np.asarray(debtc, dtype=np.float64)
    cash = np.asarray(cash, dtype=np.float64)
    ok = (np.isfinite(assets) & np.isfinite(debt) & np.isfinite(debtc)
          & np.isfinite(cash) & (assets > 0))
    a_safe = np.where(ok, assets, 1.0)
    net = np.where(ok, (debtc - cash) / a_safe, np.nan)
    nz = ok & (debt != 0)
    share = np.where(nz, debtc / np.where(nz, debt, 1.0), 0.0)
    share = np.where(ok, share, np.nan)
    lev = np.where(ok, debt / a_safe, np.nan)
    return net, share, lev


def daily_total_return(data):
    """Daily total return close-to-close; NaN where the stock has no price
    on the row or on the previous row."""
    c = np.asarray(data.close, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    ok = listed & np.isfinite(c)
    valid = np.zeros_like(ok)
    valid[1:] = ok[1:] & ok[:-1]
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    return np.where(valid & np.isfinite(r), r, np.nan)


def realized_vol(r, window=VOL_WINDOW, min_days=VOL_MIN_DAYS):
    """Rolling std (ddof 1) of the valid daily returns over the last `window`
    rows; NaN with fewer than `min_days` valid returns."""
    r = np.asarray(r, dtype=np.float64)
    T = r.shape[0]
    ok = np.isfinite(r)
    x = np.where(ok, r, 0.0)
    lo = np.maximum(np.arange(T) + 1 - window, 0)
    c = np.zeros((T + 1,) + r.shape[1:])
    np.cumsum(ok, axis=0, out=c[1:])
    n = c[1:] - c[lo]
    np.cumsum(x, axis=0, out=c[1:])
    s1 = c[1:] - c[lo]
    np.cumsum(x * x, axis=0, out=c[1:])
    s2 = c[1:] - c[lo]
    del c, x
    good = n >= min_days
    nn = np.where(good, n, 2.0)
    var = (s2 - s1 * s1 / nn) / (nn - 1.0)
    return np.where(good, np.sqrt(np.maximum(var, 0.0)), np.nan)


def cs_feature(x, members):
    z = cs_zscore(np.asarray(x, dtype=np.float32), members)
    return np.clip(z, -WINSOR, WINSOR)


def universe_of(data):
    if data.universe is None:
        raise ValueError("H-0158 needs the LIQ-500 point-in-time universe")
    return np.asarray(data.universe, dtype=bool)


def eligible(data):
    """LIQ-500 point-in-time members minus banks (per row)."""
    return universe_of(data) & ~bank_mask(data.extras["sf1_arq_deposits"])


def model_score(data, params, members, z):
    """Walk-forward gbm score on all columns; non-members masked per row."""
    ex = data.extras
    net, share, lev = balance_sheet_features(
        ex["sf1_arq_assets"], ex["sf1_arq_debt"], ex["sf1_arq_debtc"], ex["sf1_arq_cashneq"])
    f_net = cs_feature(net, members)
    del net
    f_share = cs_feature(share, members)
    del share
    f_lev = cs_feature(lev, members)
    del lev
    f_vol = cs_feature(realized_vol(daily_total_return(data)), members)
    T, N = f_net.shape
    f_z = np.broadcast_to(np.asarray(z, dtype=np.float32)[:, None], (T, N))
    X = stack(f_z, f_net, f_share, f_lev, f_vol)
    del f_net, f_share, f_lev, f_vol
    y = cs_rank(forward_return(data, HORIZON), members)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    return walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                        retrain_every=RETRAIN_EVERY, universe=members)


def leg_book(score, members, top):
    """Long the top `top` fraction, short the bottom `top` fraction, 0.5 / 0.5,
    flat with fewer than MIN_NAMES scored members."""
    w = score_weights(score, members, top=float(top), bottom=float(top), min_names=MIN_NAMES)
    w = np.asarray(w, dtype=np.float64)
    return np.where(np.isfinite(w), w, 0.0)


# ---------------------------------------------------------------- book
def apply_universe_exits(w, uni):
    """w: decision rows filled, other rows NaN. A held stock that leaves the
    universe is closed (weight 0) at the first row outside the universe."""
    out = np.array(w, dtype=np.float64, copy=True)
    T, N = out.shape
    held = np.zeros(N)
    for t in range(T):
        row = out[t]
        fin = np.isfinite(row)
        if fin.any():
            held = np.where(fin, row, held)
        leave = (held != 0) & ~uni[t]
        if leave.any():
            out[t, leave] = 0.0
            held[leave] = 0.0
    return out


def target_weights(data, params):
    uni = universe_of(data)
    members = eligible(data)
    z = shock_from_data(data)
    score = model_score(data, params, members, z)
    w = leg_book(score, members, params["top"]) * size_factor(z)[:, None]
    w = only_on(weekly_decision_days(data.dates), w)
    return apply_universe_exits(w, uni)
