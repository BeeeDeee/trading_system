"""H-0162: Treasury-curve shocks into funding structure, learned (LightGBM), LIQ-500, beta-balanced legs.

Rules of the card and where they live:
- weekly decision after the close of the last trading day of the week -> week_end_mask (Friday rule)
- z_curve / z_2y: 5-observation change / std of such changes over trailing 252 observations -> rate_z
- shock week: |z_curve| >= shock_z or |z_2y| >= shock_z -> shock_rows
- features 1-4 from SF1 -> balance_features; feature 5 -> relative_return; all cs z-scored, winsorized +-4
- z_curve, z_2y date-level, NaN outside shock decision rows -> build_features
- label: forward_return(data, 5) ranked across members; gbm walk-forward, refit 126 rows -> model_scores
- book: top/bottom 10 % of scored members, >= 50 scored, legs scaled to zero trailing 252-session beta to
  the equal-weight universe, each leg capped at 0.7 gross, gross 1 -> beta_balanced_row
- 5-session hold, flat in non-shock weeks, new cohort each shock week -> schedule
"""
import numpy as np
from lab.framework.api import total_return_index, trailing_return
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"n_estimators": 100, "shock_z": 1.0}

HORIZON = 5          # label horizon and holding period (sessions)
CHG_K = 5            # change over the previous 5 observations
Z_WIN = 252          # trailing observations for the std of 5-observation changes
RET_K = 5            # own 5-session total return
TOP = 0.10           # fixed by the card, not a grid parameter
MIN_NAMES = 50       # fewer scored members on a date: flat
BETA_WIN = 252       # trailing sessions for the leg betas
LEG_CAP = 0.7        # max gross of one leg
WINSOR = 4.0
RETRAIN = 126
GBM_FIXED = {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 200}


# ----------------------------------------------------------------------------------------------- calendar
def week_end_mask(dates):
    """True on Fridays: the last weekday of the week, computed from the date of the row alone.

    Day 0 of the epoch was a Thursday, so (days + 3) % 7 gives Monday = 0 ... Friday = 4. Exchange holidays are
    not knowable point-in-time, so a week whose Friday is a holiday has no decision date.
    """
    days = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    return (days + 3) % 7 == 4


# ----------------------------------------------------------------------------------------------- rates
def rate_z(x, k=CHG_K, win=Z_WIN):
    """Change over the previous k available observations / std (ddof 1) of such changes over the trailing
    `win` observations (current one included). NaN until win changes exist. Observations are the rows with
    a finite value (the aligned series carries the last known value)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    out = np.full(x.shape, np.nan)
    pos = np.flatnonzero(np.isfinite(x))
    v = x[pos]
    if v.size < k + win:
        return out
    chg = v[k:] - v[:-k]                       # chg[j] belongs to observation j + k
    c1 = np.concatenate([[0.0], np.cumsum(chg)])
    c2 = np.concatenate([[0.0], np.cumsum(chg * chg)])
    s1 = c1[win:] - c1[:-win]                  # window ending at chg index win-1 ...
    s2 = c2[win:] - c2[:-win]
    var = (s2 - s1 * s1 / win) / (win - 1)
    var = np.where(var > 1e-14, var, np.nan)
    z = chg[win - 1:] / np.sqrt(var)
    out[pos[k + win - 1:]] = z
    return out


def shock_rows(zc, z2, week_end, shock_z):
    """Decision rows (week end, both z finite) on which either |z| reaches shock_z."""
    ok = week_end & np.isfinite(zc) & np.isfinite(z2)
    with np.errstate(invalid="ignore"):
        big = (np.abs(zc) >= shock_z) | (np.abs(z2) >= shock_z)
    return ok & big


# ----------------------------------------------------------------------------------------------- features
def balance_features(deposits, assets, debtc, debt, intexp, revenue):
    """Deposit share (missing or zero deposits read as 0 when assets are present), short-term debt share,
    total debt share, interest burden (intexp / revenue clipped to [0, 2], missing if revenue <= 0).
    Assets missing or <= 0: all four missing."""
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(np.isfinite(assets) & (assets > 0), assets, np.nan)
        dep = np.where(np.isfinite(deposits), deposits, 0.0) / a
        sdebt = debtc / a
        tdebt = debt / a
        ib = np.where(np.isfinite(revenue) & (revenue > 0), np.clip(intexp / revenue, 0.0, 2.0), np.nan)
        ib = np.where(np.isfinite(a), ib, np.nan)
    return dep, sdebt, tdebt, ib


def relative_return(r5, uni):
    """Own 5-session return minus the mean over the point-in-time members with a finite return."""
    valid = uni & np.isfinite(r5)
    s = np.where(valid, r5, 0.0).sum(axis=1)
    n = valid.sum(axis=1)
    mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return r5 - mean[:, None]


def _cs(x, uni):
    z = np.asarray(cs_zscore(x, uni), dtype=np.float32)
    return np.clip(z, -WINSOR, WINSOR)


def build_features(data, params):
    """(T, N, 7) features and the shock-row mask. Non-shock rows have NaN date-level inputs."""
    uni = np.asarray(data.universe, dtype=bool)
    T, N = uni.shape
    zc = rate_z(data.series["fred_macro.T10Y2Y"])
    z2 = rate_z(data.series["fred_macro.DGS2"])
    if zc.shape[0] != T or z2.shape[0] != T:
        raise ValueError("rate series not aligned with the view")
    shock = shock_rows(zc, z2, week_end_mask(data.dates), float(params["shock_z"]))

    ex = data.extras
    dep, sdebt, tdebt, ib = balance_features(
        np.asarray(ex["sf1_arq_deposits"], dtype=np.float64),
        np.asarray(ex["sf1_arq_assets"], dtype=np.float64),
        np.asarray(ex["sf1_arq_debtc"], dtype=np.float64),
        np.asarray(ex["sf1_arq_debt"], dtype=np.float64),
        np.asarray(ex["sf1_art_intexp"], dtype=np.float64),
        np.asarray(ex["sf1_art_revenue"], dtype=np.float64),
    )
    f_dep = _cs(dep, uni); del dep
    f_sd = _cs(sdebt, uni); del sdebt
    f_td = _cs(tdebt, uni); del tdebt
    f_ib = _cs(ib, uni); del ib

    idx = total_return_index(data)
    r5 = np.asarray(trailing_return(idx, RET_K), dtype=np.float64)
    del idx
    f_r5 = _cs(relative_return(r5, uni), uni)
    del r5

    zc_s = np.where(shock, zc, np.nan).astype(np.float32)
    z2_s = np.where(shock, z2, np.nan).astype(np.float32)
    f_zc = np.where(uni, zc_s[:, None], np.float32(np.nan)).astype(np.float32)
    f_z2 = np.where(uni, z2_s[:, None], np.float32(np.nan)).astype(np.float32)
    X = stack(f_dep, f_sd, f_td, f_ib, f_r5, f_zc, f_z2)
    return X, shock


def model_scores(data, params):
    """Walk-forward gbm scores (T, N) and the shock-row mask."""
    uni = np.asarray(data.universe, dtype=bool)
    X, shock = build_features(data, params)
    y = cs_rank(forward_return(data, HORIZON), uni)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN, universe=uni)
    return np.asarray(score, dtype=np.float64), shock


def leg_books(score, uni, shock):
    """Equal-weight top / bottom 10 % legs among scored members on shock rows (sign only matters here);
    rows with fewer than MIN_NAMES scored members are flat. Returns (T, N) with +1 / -1 / 0 on shock rows."""
    base = np.asarray(score_weights(score, uni, top=TOP, bottom=TOP, min_names=MIN_NAMES), dtype=np.float64)
    base = np.nan_to_num(base, nan=0.0)
    scored = (np.isfinite(score) & uni).sum(axis=1)
    sides = np.sign(base) * uni
    sides[~shock] = 0.0
    sides[scored < MIN_NAMES] = 0.0
    return sides


# ----------------------------------------------------------------------------------------------- book
def daily_returns(data):
    """Close-to-close total return per row and the equal-weight member mean."""
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    listed = np.asarray(data.listed, dtype=bool)
    r = np.where(listed & np.isfinite(r), r, np.nan)
    uni = np.asarray(data.universe, dtype=bool)
    valid = uni & np.isfinite(r)
    n = valid.sum(axis=1)
    m = np.where(n > 0, np.where(valid, r, 0.0).sum(axis=1) / np.maximum(n, 1), np.nan)
    return r.astype(np.float32), m


def stock_betas(r, m, t, cols, win=BETA_WIN):
    """OLS beta of each column in `cols` to m over rows t-win+1 .. t (rows with both returns finite).
    NaN if fewer than win rows exist before t or the market variance is zero."""
    cols = np.asarray(cols, dtype=np.int64)
    out = np.full(cols.size, np.nan)
    if t - win + 1 < 1 or cols.size == 0:
        return out
    R = np.asarray(r[t - win + 1:t + 1][:, cols], dtype=np.float64)
    M = np.asarray(m[t - win + 1:t + 1], dtype=np.float64)[:, None]
    v = np.isfinite(R) & np.isfinite(M)
    n = v.sum(axis=0)
    Rz = np.where(v, R, 0.0)
    Mz = np.where(v, M, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mr = Rz.sum(axis=0) / n
        mm = Mz.sum(axis=0) / n
        cov = (np.where(v, (R - mr) * (M - mm), 0.0)).sum(axis=0) / n
        var = (np.where(v, (M - mm) ** 2, 0.0)).sum(axis=0) / n
        beta = np.where((n >= 2) & (var > 0), cov / var, np.nan)
    return beta


def long_leg_share(b_long, b_short, cap=LEG_CAP):
    """Share of gross in the long leg so that w_L * b_L - (1 - w_L) * b_S = 0, clipped to [1-cap, cap]
    (clipping the root minimizes the residual beta of this linear function). 0.5 if a beta is unknown."""
    if not (np.isfinite(b_long) and np.isfinite(b_short)):
        return 0.5
    den = b_long + b_short
    if abs(den) < 1e-12:
        return 0.5
    return float(np.clip(b_short / den, 1.0 - cap, cap))


def beta_balanced_row(side_row, r, m, t):
    """Weights of one decision row: equal weight inside each leg, legs scaled to zero beta, gross 1."""
    w = np.zeros(side_row.shape[0])
    L = np.flatnonzero(side_row > 0)
    S = np.flatnonzero(side_row < 0)
    if L.size == 0 or S.size == 0:
        return w
    bl = stock_betas(r, m, t, L)
    bs = stock_betas(r, m, t, S)
    b_long = float(np.nanmean(bl)) if np.isfinite(bl).any() else np.nan
    b_short = float(np.nanmean(bs)) if np.isfinite(bs).any() else np.nan
    wl = long_leg_share(b_long, b_short)
    w[L] = wl / L.size
    w[S] = -(1.0 - wl) / S.size
    return w


def schedule(shock, rows, uni, hold=HORIZON):
    """Weight matrix from decision books.

    shock[t]: a shock decision row, rows[t] its target weights (zeros = flat that week).
    Each cohort is held `hold` sessions: zeros on row e + hold (executed at the open hold sessions after
    entry) unless a new shock decision falls there; a new shock decision replaces the cohort. A held name
    that leaves the universe is closed on that row (NaN elsewhere keeps the other positions).
    """
    T, N = uni.shape
    W = np.full((T, N), np.nan)
    entry = -1
    held = np.zeros(N, dtype=bool)
    for t in range(T):
        if shock[t]:
            W[t] = rows[t]
            held = rows[t] != 0
            entry = t if held.any() else -1
        elif entry >= 0 and t == entry + hold:
            W[t] = 0.0
            entry = -1
            held[:] = False
        elif entry >= 0:
            leave = held & ~uni[t]
            if leave.any():
                W[t, leave] = 0.0
                held &= ~leave
    return W


def target_weights(data, params):
    uni = np.asarray(data.universe, dtype=bool)
    score, shock = model_scores(data, params)
    sides = leg_books(score, uni, shock)
    del score
    r, m = daily_returns(data)
    T, N = uni.shape
    rows = np.zeros((T, N))
    for t in np.flatnonzero(shock):
        rows[t] = beta_balanced_row(sides[t], r, m, t)
    return schedule(shock, rows, uni, HORIZON)
