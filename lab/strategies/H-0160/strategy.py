"""H-0160 Aggregate earnings news into Treasuries.

Signal (signal-only stocks, never traded): LIQ-500 members' release-session reactions net of the
non-announcer equal-weight market, standardized by the stock's trailing 63-session residual vol,
averaged over the calendar week; z-scored against previous qualifying weeks. Book: TLT/IEF short
(long) 0.5/0.5 after strongly positive (negative) weeks, held to the next weekly decision.
"""
import numpy as np
from lab.framework.api import only_on

PARAMS = {"norm_weeks": 40, "z_entry": 0.5}

LIQ_N = 500          # LIQ-500 membership (card: liq_rank <= 500)
SIGMA_WIN = 63       # sessions ending at d-1
SIGMA_MIN = 40       # minimum valid values for sigma_j
WINSOR = 5.0         # s_jd clipped to [-5, +5]
MIN_ANNOUNCERS = 25  # N_t >= 25: earnings-season week
MIN_HIST = 20        # at least 20 previous qualifying weeks


def _day_numbers(dates):
    return dates.astype("datetime64[D]").astype(np.int64)


def week_id(dates):
    """Monday-anchored calendar-week index (day 0 of the epoch was a Thursday, hence +3)."""
    return (_day_numbers(dates) + 3) // 7


def weekly_decision_rows(dates):
    """Row t is the weekly decision if the next weekday falls in another calendar week.

    Uses only the date of row t (np.busday_offset ignores exchange holidays): a Friday session.
    When the Friday is an exchange holiday the week has no decision (approximation).
    """
    d = dates.astype("datetime64[D]")
    nxt = np.busday_offset(d, 1, roll="forward")
    return week_id(nxt) != week_id(d)


def stock_columns(data):
    return np.array([i for i, a in enumerate(data.asset_class) if a == "us_equity"], dtype=np.int64)


def liq_members(data, cols, listed):
    """Point-in-time LIQ-500 membership of the stock columns.

    1. extras["liq_rank"] <= 500 (the card's definition) when the loader delivers it;
    2. else the stock columns of data.universe, if a membership mask is given;
    3. else every delivered listed stock, but ONLY if the loader delivered at most 500 listed stocks
       on every row (i.e. the delivered set is itself the PIT LIQ-500); otherwise fail loudly.
    """
    if data.extras is not None and "liq_rank" in data.extras:
        rank = np.asarray(data.extras["liq_rank"][:, cols], dtype=np.float64)
        return np.isfinite(rank) & (rank <= LIQ_N)
    if data.universe is not None:
        uni = np.asarray(data.universe[:, cols], dtype=bool) & listed
        if uni.any():
            return uni
    if listed.sum(axis=1).max(initial=0) > LIQ_N:
        raise ValueError("liq_rank not delivered and more than 500 listed stocks on a row: "
                         "cannot form the point-in-time LIQ-500 the card requires")
    return listed.copy()


def announcer_reactions(data):
    """Return (row_sum, row_count) of winsorized announcer reactions s_jd for each row d."""
    cols = stock_columns(data)
    if cols.size == 0:
        raise ValueError("no us_equity (sharadar_sep) columns in the view")
    T = len(data.dates)
    rco = np.asarray(data.ret_co[:, cols], dtype=np.float64)
    roc = np.asarray(data.ret_oc[:, cols], dtype=np.float64)
    close = np.asarray(data.close[:, cols], dtype=np.float64)
    listed = np.asarray(data.listed[:, cols], dtype=bool)
    R = (1.0 + rco) * (1.0 + roc) - 1.0
    valid = listed & np.isfinite(close) & np.isfinite(R)
    R = np.where(valid, R, np.nan)

    member = liq_members(data, cols, listed)
    if T > 0 and not member.any():
        raise ValueError("no LIQ-500 member in the view: the signal input is missing")

    ea = np.asarray(data.extras["ea_reaction_day"][:, cols], dtype=np.float64)
    ea = np.where(np.isfinite(ea), ea, 0.0) > 0.5          # NaN treated as 0

    # market M_d: equal-weight mean of members without a release, valid return
    mk = member & valid & ~ea
    cnt = mk.sum(axis=1)
    sm = np.where(mk, R, 0.0).sum(axis=1)
    M = np.full(T, np.nan)
    ok = cnt > 0
    M[ok] = sm[ok] / cnt[ok]

    # residual D_jd = R_jd - M_d (any valid stock-day, membership not required for sigma history)
    D = R - M[:, None]
    dv = np.isfinite(D)
    D0 = np.where(dv, D, 0.0)
    n_s = D0.shape[1]
    c1 = np.zeros((T + 1, n_s))
    c2 = np.zeros((T + 1, n_s))
    cn = np.zeros((T + 1, n_s))
    np.cumsum(D0, axis=0, out=c1[1:])
    np.cumsum(D0 * D0, axis=0, out=c2[1:])
    np.cumsum(dv.astype(np.float64), axis=0, out=cn[1:])
    idx = np.arange(T)
    lo = np.maximum(idx - SIGMA_WIN, 0)
    # rows lo..d-1 for row d: c[d] - c[lo]
    s1 = c1[idx] - c1[lo]
    s2 = c2[idx] - c2[lo]
    nn = cn[idx] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / nn) / (nn - 1.0)
    var = np.where(nn >= SIGMA_MIN, np.maximum(var, 0.0), np.nan)
    sigma = np.sqrt(var)

    ann = member & ea & dv & np.isfinite(sigma) & (sigma > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.clip(D / sigma, -WINSOR, WINSOR)
    s = np.where(ann, s, 0.0)
    return s.sum(axis=1), ann.sum(axis=1).astype(np.float64)


def weekly_z(data, params):
    """z_t on weekly decision rows (NaN where no signal), plus the decision-row mask."""
    norm_weeks = int(params["norm_weeks"])
    dates = data.dates
    T = len(dates)
    row_sum, row_cnt = announcer_reactions(data)
    dec = weekly_decision_rows(dates)
    wk = week_id(dates)
    cs = np.concatenate([[0.0], np.cumsum(row_sum)])
    cc = np.concatenate([[0.0], np.cumsum(row_cnt)])
    # first row of each row's calendar week
    first = np.zeros(T, dtype=np.int64)
    if T:
        start = np.r_[True, wk[1:] != wk[:-1]]
        first = np.maximum.accumulate(np.where(start, np.arange(T), 0))
    z = np.full(T, np.nan)
    hist = []
    for t in np.flatnonzero(dec):
        n_t = cc[t + 1] - cc[first[t]]
        if n_t < MIN_ANNOUNCERS:
            continue
        a_t = (cs[t + 1] - cs[first[t]]) / n_t
        if len(hist) >= MIN_HIST:
            h = np.asarray(hist[-norm_weeks:])
            sd = h.std(ddof=1)
            if sd > 0:
                z[t] = (a_t - h.mean()) / sd
        hist.append(a_t)
    return z, dec


def etf_bar(data, name):
    j = data.col(name)
    return np.asarray(data.listed[:, j], dtype=bool) & np.isfinite(np.asarray(data.close[:, j], dtype=np.float64))


def target_weights(data, params):
    z_entry = float(params["z_entry"])
    T, N = len(data.dates), len(data.instruments)
    jt, ji = data.col("TLT"), data.col("IEF")
    z, dec = weekly_z(data, params)
    bar = etf_bar(data, "TLT") & etf_bar(data, "IEF")
    side = np.zeros(T)
    side[np.isfinite(z) & (z >= z_entry)] = -1.0
    side[np.isfinite(z) & (z <= -z_entry)] = 1.0
    side[~bar] = 0.0
    w = np.zeros((T, N))
    w[:, jt] = 0.5 * side
    w[:, ji] = 0.5 * side
    return only_on(dec, w)
