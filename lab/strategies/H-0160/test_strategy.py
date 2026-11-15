import math

import numpy as np
import pytest
from lab.framework.data import DataView

from strategy import PARAMS, target_weights, announcer_reactions, weekly_decision_rows, weekly_z

ETFS = ("TLT", "IEF", "SHY", "SHV", "BIL")


def bdays(n_weeks):
    # day 4 of the epoch is a Monday; keep Monday..Friday
    d = np.arange(4, 4 + 7 * n_weeks)
    d = d[((d + 3) % 7) < 5]
    return d.astype("datetime64[D]")


def make_view(n_weeks=48, n_stocks=60, n_ann=30, ann_from_week=14, seed=1, last_week_shock=0.0):
    rng = np.random.default_rng(seed)
    dates = bdays(n_weeks)
    T = len(dates)
    names = ETFS + tuple("E%d" % (100 + k) for k in range(n_stocks))
    N = len(names)
    ac = ("us_etf",) * len(ETFS) + ("us_equity",) * n_stocks
    ret_co = rng.normal(0, 0.005, (T, N))
    ret_oc = rng.normal(0, 0.01, (T, N))
    ea = np.zeros((T, N))
    ea[:, :len(ETFS)] = np.nan
    wk = (dates.astype(np.int64) + 3) // 7
    wd = (dates.astype(np.int64) + 3) % 7
    uw = np.unique(wk)
    for i, w in enumerate(uw):
        if i < ann_from_week:
            continue
        rows = np.flatnonzero((wk == w) & (wd == 2))
        if rows.size == 0:
            continue
        start = (i * 7) % n_stocks
        js = [len(ETFS) + (start + k) % n_stocks for k in range(n_ann)]
        for j in js:
            ea[rows[0], j] = 1.0
            if i == len(uw) - 1 and last_week_shock != 0.0:
                ret_oc[rows[0], j] = last_week_shock
    liq = np.full((T, N), np.nan)
    liq[:, len(ETFS):] = np.arange(1, n_stocks + 1)[None, :]
    close = np.full((T, N), 50.0)
    return DataView(
        dates=dates, instruments=names, asset_class=ac,
        ret_co=ret_co, ret_oc=ret_oc,
        tradable=np.ones((T, N), dtype=bool), listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool), close=close,
        dollar_volume=np.full((T, N), 1e8), universe=None,
        extras={"ea_reaction_day": ea, "liq_rank": liq}, series={}, cash_ret=np.zeros(T))


def truncate(v, t):
    ex = {k: a[: t + 1] for k, a in v.extras.items()}
    return DataView(
        dates=v.dates[: t + 1], instruments=v.instruments, asset_class=v.asset_class,
        ret_co=v.ret_co[: t + 1], ret_oc=v.ret_oc[: t + 1], tradable=v.tradable[: t + 1],
        listed=v.listed[: t + 1], delisting=v.delisting[: t + 1], close=v.close[: t + 1],
        dollar_volume=v.dollar_volume[: t + 1], universe=None, extras=ex, series={},
        cash_ret=v.cash_ret[: t + 1])


# ---------- slow reference implementation of the card ----------
def reference_z(v, norm_weeks):
    cols = [j for j, a in enumerate(v.asset_class) if a == "us_equity"]
    T = len(v.dates)
    R = (1 + v.ret_co) * (1 + v.ret_oc) - 1
    rank = v.extras["liq_rank"]
    ea = np.nan_to_num(v.extras["ea_reaction_day"], nan=0.0)
    M = np.full(T, np.nan)
    for d in range(T):
        xs = [R[d, j] for j in cols if v.listed[d, j] and np.isfinite(v.close[d, j])
              and rank[d, j] <= 500 and ea[d, j] == 0]
        if xs:
            M[d] = sum(xs) / len(xs)
    s_sum, s_n = np.zeros(T), np.zeros(T)
    for d in range(T):
        for j in cols:
            if not (ea[d, j] == 1 and rank[d, j] <= 500 and v.listed[d, j] and np.isfinite(v.close[d, j])
                    and np.isfinite(M[d])):
                continue
            hist = [R[k, j] - M[k] for k in range(max(0, d - 63), d)
                    if v.listed[k, j] and np.isfinite(v.close[k, j]) and np.isfinite(M[k])]
            if len(hist) < 40:
                continue
            sd = float(np.std(hist, ddof=1))
            if sd <= 0:
                continue
            s = min(5.0, max(-5.0, (R[d, j] - M[d]) / sd))
            s_sum[d] += s
            s_n[d] += 1
    days = v.dates.astype(np.int64)
    wk = (days + 3) // 7
    wd = (days + 3) % 7
    z = np.full(T, np.nan)
    hist = []
    for t in range(T):
        if wd[t] != 4:
            continue
        rows = [k for k in range(t + 1) if wk[k] == wk[t]]
        n = s_n[rows].sum()
        if n < 25:
            continue
        a = s_sum[rows].sum() / n
        if len(hist) >= 20:
            h = hist[-norm_weeks:]
            mu = sum(h) / len(h)
            sd = math.sqrt(sum((x - mu) ** 2 for x in h) / (len(h) - 1))
            z[t] = (a - mu) / sd
        hist.append(a)
    return z


def test_decision_rows_are_fridays():
    d = bdays(3)
    dec = weekly_decision_rows(d)
    wd = (d.astype(np.int64) + 3) % 7
    assert np.array_equal(dec, wd == 4)
    assert dec.sum() == 3


@pytest.mark.parametrize("nw", [26, 40, 60])
def test_matches_reference(nw):
    v = make_view(n_weeks=60)
    z, dec = weekly_z(v, {"norm_weeks": nw, "z_entry": 0.5})
    zr = reference_z(v, nw)
    assert np.array_equal(np.isfinite(z), np.isfinite(zr))
    assert np.isfinite(z).sum() >= 10
    np.testing.assert_allclose(z[np.isfinite(z)], zr[np.isfinite(zr)], rtol=1e-8, atol=1e-10)


def test_weights_follow_z_and_shape():
    v = make_view(n_weeks=60)
    p = dict(PARAMS)
    w = target_weights(v, p)
    z, dec = weekly_z(v, p)
    assert w.shape == (len(v.dates), len(v.instruments))
    assert np.all(np.isnan(w[~dec]))
    wd = w[dec]
    assert np.all(np.abs(wd).sum(axis=1) <= 1.0 + 1e-12)
    jt, ji = v.col("TLT"), v.col("IEF")
    for t in np.flatnonzero(dec):
        exp = 0.0
        if np.isfinite(z[t]) and z[t] >= p["z_entry"]:
            exp = -0.5
        elif np.isfinite(z[t]) and z[t] <= -p["z_entry"]:
            exp = 0.5
        assert w[t, jt] == exp and w[t, ji] == exp
        others = [j for j in range(len(v.instruments)) if j not in (jt, ji)]
        assert np.all(w[t, others] == 0.0)
    assert (w[dec, jt] != 0).sum() > 0


def test_positive_shock_shorts_negative_longs():
    vp = make_view(n_weeks=50, last_week_shock=0.15)
    vn = make_view(n_weeks=50, last_week_shock=-0.15)
    wp = target_weights(vp, PARAMS)
    wn = target_weights(vn, PARAMS)
    t = len(vp.dates) - 1
    assert weekly_decision_rows(vp.dates)[t]
    assert wp[t, vp.col("TLT")] == -0.5 and wp[t, vp.col("IEF")] == -0.5
    assert wn[t, vn.col("TLT")] == 0.5 and wn[t, vn.col("IEF")] == 0.5


def test_first_twenty_qualifying_weeks_no_signal():
    v = make_view(n_weeks=60)
    z, dec = weekly_z(v, PARAMS)
    _, n = announcer_reactions(v)
    wk = (v.dates.astype(np.int64) + 3) // 7
    qual = [t for t in np.flatnonzero(dec) if n[wk == wk[t]].sum() >= 25]
    assert len(qual) > 25
    assert np.all(np.isnan(z[qual[:20]]))
    assert np.isfinite(z[qual[20]])
    nonq = [t for t in np.flatnonzero(dec) if t not in qual]
    assert np.all(np.isnan(z[nonq]))


def test_24_announcers_do_not_qualify():
    v = make_view(n_weeks=60, n_ann=24)
    z, _ = weekly_z(v, PARAMS)
    assert np.all(np.isnan(z))
    w = target_weights(v, PARAMS)
    assert np.nansum(np.abs(w)) == 0.0


def test_non_members_excluded_and_nan_ea_is_zero():
    v = make_view(n_weeks=60)
    base, _ = weekly_z(v, PARAMS)
    # NaN in ea for non-announcers changes nothing
    ea = v.extras["ea_reaction_day"].copy()
    ea[ea == 0] = np.nan
    v2 = make_view(n_weeks=60)
    v2.extras["ea_reaction_day"] = ea
    z2, _ = weekly_z(v2, PARAMS)
    np.testing.assert_array_equal(np.isfinite(base), np.isfinite(z2))
    np.testing.assert_allclose(base[np.isfinite(base)], z2[np.isfinite(z2)])
    # ranks above 500 drop out of the signal
    v3 = make_view(n_weeks=60)
    v3.extras["liq_rank"][:, len(ETFS) + 50:] = 600.0
    _, n3 = announcer_reactions(v3)
    _, n0 = announcer_reactions(v)
    assert n3.sum() < n0.sum()


def test_sigma_needs_40_values():
    v = make_view(n_weeks=60, ann_from_week=0)
    _, n = announcer_reactions(v)
    assert n[:40].sum() == 0          # fewer than 40 past values: announcers skipped
    assert n[41:].sum() > 0


def test_no_etf_bar_is_flat():
    v = make_view(n_weeks=50, last_week_shock=0.15)
    t = len(v.dates) - 1
    v.close[t, v.col("IEF")] = np.nan
    w = target_weights(v, PARAMS)
    assert w[t, v.col("TLT")] == 0.0 and w[t, v.col("IEF")] == 0.0


def test_params_change_output():
    v = make_view(n_weeks=70, seed=3)
    w0 = target_weights(v, {"norm_weeks": 40, "z_entry": 0.5})
    w1 = target_weights(v, {"norm_weeks": 40, "z_entry": 0.25})
    z26, _ = weekly_z(v, {"norm_weeks": 26, "z_entry": 0.5})
    z40, _ = weekly_z(v, {"norm_weeks": 40, "z_entry": 0.5})
    assert np.nansum(np.abs(w1)) > np.nansum(np.abs(w0))
    f = np.isfinite(z26)
    assert not np.allclose(z26[f], z40[f])


def test_point_in_time_truncation():
    v = make_view(n_weeks=60)
    w = target_weights(v, PARAMS)
    for t in (100, 180, 230, 275):
        wt = target_weights(truncate(v, t), PARAMS)
        np.testing.assert_array_equal(np.isnan(w[: t + 1]), np.isnan(wt))
        np.testing.assert_allclose(np.nan_to_num(w[: t + 1]), np.nan_to_num(wt))


def test_missing_liq_rank_with_many_stocks_fails_loudly():
    v = make_view(n_weeks=20, n_stocks=501, n_ann=30)
    del v.extras["liq_rank"]
    with pytest.raises(ValueError):
        target_weights(v, PARAMS)
