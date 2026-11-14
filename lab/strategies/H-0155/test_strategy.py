import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS
from strategy import formation_rows, vix_innovations, exposures, long_leg, tie_ranks, market_factor
from strategy import formation_long_leg, prepare

EPOCH_2018 = 17532   # days since 1970-01-01 of Monday 2018-01-01 (no date strings)


def weekdays(n_cal_days, start=EPOCH_2018):
    d = np.arange(start, start + n_cal_days).astype("datetime64[D]")
    return d[np.is_busday(d)]


def make_view(T_cal=760, N=30, seed=0, uni=None, close=None):
    rng = np.random.default_rng(seed)
    dates = weekdays(T_cal)
    T = len(dates)
    mkt = rng.normal(0, 0.01, T)
    vix = 20 + np.cumsum(rng.normal(0, 1, T))
    beta = np.linspace(0.5, 1.5, N)
    gvix = np.linspace(-0.002, 0.002, N)[::-1]
    dv = np.r_[np.nan, np.diff(vix)]
    r = mkt[:, None] * beta[None, :] + np.nan_to_num(dv)[:, None] * gvix[None, :] + rng.normal(0, 0.01, (T, N))
    ret_co = (r / 2).astype(np.float32)
    ret_oc = ((1 + r) / (1 + ret_co) - 1).astype(np.float32)
    if close is None:
        close = np.full((T, N), 50.0, dtype=np.float32)
    if uni is None:
        uni = np.ones((T, N), dtype=bool)
    listed = np.isfinite(close)
    aligned = np.r_[np.nan, vix[:-1]]        # next_morning clock: row t holds VIX of t-1
    view = DataView(
        dates=dates,
        instruments=tuple("E%d" % (100 + j) for j in range(N)),
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=np.where(listed, ret_co, 0).astype(np.float32),
        ret_oc=np.where(listed, ret_oc, 0).astype(np.float32),
        tradable=listed.copy(),
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.where(listed, 1e8, np.nan).astype(np.float32),
        universe=uni,
        extras={},
        series={"fred_macro.VIXCLS": aligned},
        cash_ret=np.zeros(T),
    )
    return view


# ------------------------------------------------------------------------------------------ calendar
def test_formation_rows_last_weekday_each_month():
    d = weekdays(400)
    f = formation_rows(d)
    months = d.astype("datetime64[M]")
    for m in np.unique(months)[:-1]:
        rows = np.nonzero(months == m)[0]
        assert f[rows].sum() == 1
        assert f[rows[-1]]
    # point-in-time: flags are unchanged when the data is truncated
    for cut in (17, 40, 123, 250):
        assert np.array_equal(formation_rows(d[:cut]), f[:cut])


def test_formation_fallback_when_last_weekday_is_holiday():
    d = weekdays(120)
    f = formation_rows(d)
    months = d.astype("datetime64[M]")
    march = np.unique(months)[2]
    last_march = np.nonzero(months == march)[0][-1]
    d2 = np.delete(d, last_march)               # the last weekday of March is a holiday
    f2 = formation_rows(d2)
    assert not f2[last_march - 1]               # the session before is not recognised as month end
    assert f2[last_march]                       # first April session is the (one session late) formation
    assert f2.sum() == f.sum()


# ------------------------------------------------------------------------------------------ inputs
def test_vix_alignment():
    s = np.array([np.nan, 20.0, 21.0, np.nan, 19.0, 18.5])
    dv = vix_innovations(s)
    assert np.isnan(dv[0])                  # VIX of day -1 unknown
    assert dv[1] == 1.0                      # VIX_1 - VIX_0 = S[2] - S[1]
    assert np.isnan(dv[2]) and np.isnan(dv[3])
    assert dv[4] == -0.5
    assert np.isnan(dv[5])                   # VIX of the last day not yet known


def test_market_factor_members_only():
    r = np.array([[0.01, 0.03, 0.5], [0.02, 0.0, -0.5]], dtype=np.float32)
    valid = np.array([[True, True, True], [True, False, True]])
    uni = np.array([[True, True, False], [True, True, False]])
    m = market_factor(r, valid, uni)
    assert m[0] == pytest.approx(0.02)
    assert m[1] == pytest.approx(0.02)


def test_exposures_recover_betas_and_min_valid():
    rng = np.random.default_rng(1)
    T = 300
    x1 = rng.normal(0, 0.01, T)
    x2 = rng.normal(0, 1.0, T)
    y = np.stack([0.001 + 1.5 * x1 - 0.003 * x2, -0.5 * x1 + 0.002 * x2 + rng.normal(0, 0.001, T)], axis=1)
    valid = np.ones((T, 2), dtype=bool)
    valid[: T - 150, 1] = False             # stock 1 has only 150 valid days in the last 252 -> < 80 %
    b1, b2, iv, el = exposures(y.astype(np.float32), valid, x1, x2, np.array([0, 1]), T, 252)
    assert b1[0] == pytest.approx(1.5, abs=1e-3)
    assert b2[0] == pytest.approx(-0.003, abs=1e-5)
    assert iv[0] < 1e-5
    assert el[0] and not el[1]
    # the window ends at t-1: row t itself is never used
    y2 = y.copy()
    y2[T - 1] = 99.0
    b1b, _, _, _ = exposures(y2.astype(np.float32), valid, x1, x2, np.array([0]), T - 1, 252)
    b1c, _, _, _ = exposures(y.astype(np.float32), valid, x1, x2, np.array([0]), T - 1, 252)
    assert b1b[0] == b1c[0]


# ------------------------------------------------------------------------------------------ cells
def test_long_leg_cells():
    M = 30
    b_mkt = np.arange(M, dtype=float)                   # terciles: 0-9, 10-19, 20-29
    ivol = np.tile(np.arange(10, dtype=float), 3)       # halves: ivol 0-4 / 5-9 in each tercile
    b_vix = -np.arange(M, dtype=float)                  # lowest b_vix = highest index in each cell
    tie = np.arange(M)
    leg = long_leg(b_mkt, ivol, b_vix, tie)
    assert list(leg) == [4, 9, 14, 19, 24, 29]          # 5 // 5 = 1 per cell
    # 60 stocks: cells of 10 -> 2 per cell
    b_mkt = np.arange(60, dtype=float)
    ivol = np.tile(np.arange(20, dtype=float), 3)
    b_vix = np.zeros(60)
    leg = long_leg(b_mkt, ivol, b_vix, np.arange(60))   # all tied -> lowest ticker rank
    assert list(leg) == [0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51]
    # tiny cells still give at least one name
    leg = long_leg(np.arange(6.0), np.zeros(6), np.zeros(6), np.arange(6))
    assert len(leg) == 6


def test_tie_ranks_numeric_ids():
    rk = tie_ranks(("E45", "E123", "E7"))
    assert list(rk) == [1, 2, 0]


# ------------------------------------------------------------------------------------------ book
def test_book_startup_and_cohorts():
    v = make_view()
    w = target_weights(v, PARAMS)
    T, N = w.shape
    dec = np.nonzero(~np.all(np.isnan(w), axis=1))[0]
    assert len(dec) > 5
    first = dec[0]
    assert first >= int(0.8 * PARAMS["est_days"])
    f = formation_rows(v.dates)
    assert np.all(f[dec])
    # first formation: one cohort of 6 names (30 stocks, 6 cells of 5) at 1/6 each
    assert np.sum(w[first] > 0) == 6
    assert np.allclose(w[first][w[first] > 0], 1 / 6)
    for t in dec:
        row = w[t]
        assert np.all(row >= 0)
        assert row.sum() == pytest.approx(1.0)
    # second formation: two cohorts at 1/2 each
    s = w[dec[1]]
    assert s.sum() == pytest.approx(1.0)
    # rows between formations: no decision
    between = np.setdiff1d(np.arange(first, T), dec)
    assert np.all(np.isnan(w[between]))


def test_hold_months_changes_output():
    v = make_view(T_cal=1100)
    w6 = target_weights(v, PARAMS)
    w3 = target_weights(v, dict(PARAMS, hold_months=3))
    dec = np.nonzero(~np.all(np.isnan(w6), axis=1))[0]
    assert np.allclose(np.nan_to_num(w6[dec[:3]]), np.nan_to_num(w3[dec[:3]]))   # start-up identical
    assert not np.allclose(np.nan_to_num(w6[dec[4:]]), np.nan_to_num(w3[dec[4:]]))


def test_est_days_changes_output():
    v = make_view(T_cal=1100)
    w1 = target_weights(v, PARAMS)
    w2 = target_weights(v, dict(PARAMS, est_days=126))
    d1 = np.nonzero(~np.all(np.isnan(w1), axis=1))[0]
    d2 = np.nonzero(~np.all(np.isnan(w2), axis=1))[0]
    assert d2[0] < d1[0]


def test_exit_from_universe_zeroes_and_reweights():
    v0 = make_view()
    w0 = target_weights(v0, PARAMS)
    dec = np.nonzero(~np.all(np.isnan(w0), axis=1))[0]
    t0 = dec[0]
    j = int(np.nonzero(w0[t0] > 0)[0][0])
    uni = np.ones_like(v0.universe)
    uni[t0 + 3:, j] = False
    v = make_view(uni=uni)
    w = target_weights(v, PARAMS)
    assert np.allclose(w[t0], w0[t0])
    assert w[t0 + 3, j] == 0.0                       # zero on the first non-member row
    others = np.delete(np.arange(w.shape[1]), j)
    assert np.all(np.isnan(w[t0 + 3, others]))       # everybody else held as is
    assert np.all(w[t0 + 1:t0 + 3].ravel() != w[t0 + 1:t0 + 3].ravel())   # NaN before
    t1 = dec[1]
    assert w[t1, j] == 0.0
    # gross <= 1 always, and the row stays fully invested when the cohort keeps other members
    assert np.nansum(w[t1]) <= 1.0 + 1e-12


def test_late_listing_not_eligible_until_enough_history():
    v0 = make_view()
    T = len(v0.dates)
    close = np.full((T, 30), 50.0, dtype=np.float32)
    close[: T - 150, 0] = np.nan                     # listed only for the last 150 sessions
    v = make_view(close=close)
    w = target_weights(v, PARAMS)
    assert np.all(np.nan_to_num(w[:, 0]) == 0.0)


def test_point_in_time_truncation():
    v = make_view(T_cal=900)
    w = target_weights(v, PARAMS)
    T = len(v.dates)
    for cut in (300, 417, 500, T - 5):
        vt = make_view(T_cal=900)
        sl = slice(0, cut)
        vt = DataView(
            dates=v.dates[sl], instruments=v.instruments, asset_class=v.asset_class,
            ret_co=v.ret_co[sl], ret_oc=v.ret_oc[sl], tradable=v.tradable[sl], listed=v.listed[sl],
            delisting=v.delisting[sl], close=v.close[sl], dollar_volume=v.dollar_volume[sl],
            universe=v.universe[sl], extras={}, series={"fred_macro.VIXCLS": v.series["fred_macro.VIXCLS"][sl]},
            cash_ret=v.cash_ret[sl])
        wt = target_weights(vt, PARAMS)
        assert np.array_equal(np.nan_to_num(wt, nan=-1), np.nan_to_num(w[:cut], nan=-1))


def test_missing_vix_series_raises():
    v = make_view(T_cal=100)
    bad = DataView(
        dates=v.dates, instruments=v.instruments, asset_class=v.asset_class, ret_co=v.ret_co, ret_oc=v.ret_oc,
        tradable=v.tradable, listed=v.listed, delisting=v.delisting, close=v.close,
        dollar_volume=v.dollar_volume, universe=v.universe, extras={}, series={}, cash_ret=v.cash_ret)
    with pytest.raises(KeyError):
        target_weights(bad, PARAMS)


# ------------------------------------------------------------------------------------------ diagnostic
def test_new_cohort_is_the_formation_long_leg():
    # (diagnostic.py cannot be imported here; it copies formation_long_leg, whose output is the new cohort)
    v = make_view(T_cal=1100)
    w = target_weights(v, PARAMS)
    uni, r, valid, mkt, dvix, tie, form = prepare(v)
    dec = np.nonzero(~np.all(np.isnan(w), axis=1))[0]
    for t in dec[:4]:
        leg = formation_long_leg(r, valid, mkt, dvix, uni, tie, t, PARAMS["est_days"])
        assert len(leg) == 6
        assert np.all(w[t, leg] > 0)
