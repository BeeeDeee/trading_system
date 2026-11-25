import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS
from strategy import _calendar, _third_friday, _legs

N = 130
DAY0 = 10957          # days since 1970-01-01 (a Saturday at the start of 2000)


def _dates(n_cal=220, drop=()):
    d = np.arange(DAY0, DAY0 + n_cal).astype("datetime64[D]")
    d = d[np.is_busday(d)]
    if drop:
        keep = np.ones(d.shape[0], dtype=bool)
        for x in drop:
            keep &= d != x
        d = d[keep]
    return d


def _june_friday(dates):
    m = dates.astype("datetime64[M]").astype(np.int64)
    jun = np.unique(m[(m % 12) == 5])[0]
    return _third_friday(np.array([jun]))[0]


def _march_friday(dates):
    m = dates.astype("datetime64[M]").astype(np.int64)
    mar = np.unique(m[(m % 12) == 2])[0]
    return _third_friday(np.array([mar]))[0]


def build(drop=(), g_short_count=15, leave=None):
    dates = _dates(drop=drop)
    T = dates.shape[0]
    F = _june_friday(dates)
    Fp = _march_friday(dates)
    row_fp = int(np.searchsorted(dates, Fp, side="right")) - 1
    ref = row_fp - 5
    # filing A usable from row 10, filing B usable from the row 10 sessions after ref
    uA, uB = 10, ref + 10
    g = np.zeros(N)
    g[0:60] = 0.003 + 0.0005 * np.arange(60)
    g[60:75] = -0.003 - 0.001 * np.arange(15)
    if g_short_count < 15:
        g[60 + g_short_count:75] = 0.0
    g[75] = 0.001        # under the noise floor
    g[76] = 0.06         # >= 5 %: off-cycle change, excluded
    g[78] = 0.01         # split between filings, excluded
    s_ref = 1000.0 + np.arange(N)
    s_new = s_ref * np.exp(g)
    mc = np.full((T, N), np.nan)
    pr = np.full((T, N), np.nan)
    mc[uA:] = s_ref * 10.0
    pr[uA:] = 10.0
    mc[uB:] = s_new * 11.0
    pr[uB:] = 11.0
    mc[uB:, 77] = mc[uA, 77]      # no new filing
    pr[uB:, 77] = pr[uA, 77]
    mc[:, 81] = np.nan            # no fundamentals at all
    close = np.full((T, N), 10.0)
    close[uB - 1:, 78] = 5.0      # 2:1 split on the new filing session, no total-return change
    ret = np.zeros((T, N))
    uni = np.ones((T, N), dtype=bool)
    trad = np.ones((T, N), dtype=bool)
    fn, fprev, cnt, dec = _calendar(dates, PARAMS["window_sessions"])
    d = int(dec[dates[dec] > Fp][0])
    uni[d, 79] = False
    trad[d, 80] = False
    if leave is not None:
        uni[d + leave[0]:d + leave[1], leave[2]] = False
    arrays = dict(
        dates=dates,
        ret_co=ret.copy(), ret_oc=ret.copy(), tradable=trad, listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool), close=close, dollar_volume=np.full((T, N), 1e8),
        universe=uni,
        extras={"sf1_arq_marketcap": mc.astype(np.float32), "sf1_arq_price": pr.astype(np.float32)},
        cash_ret=np.zeros(T),
    )
    return arrays, d, F


def view(arrays, upto=None):
    T = arrays["dates"].shape[0] if upto is None else upto
    sl = slice(0, T)
    return DataView(
        dates=arrays["dates"][sl],
        instruments=tuple("E%d" % (1000 + j) for j in range(N)),
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=arrays["ret_co"][sl], ret_oc=arrays["ret_oc"][sl],
        tradable=arrays["tradable"][sl], listed=arrays["listed"][sl], delisting=arrays["delisting"][sl],
        close=arrays["close"][sl], dollar_volume=arrays["dollar_volume"][sl],
        universe=arrays["universe"][sl],
        extras={k: v[sl] for k, v in arrays["extras"].items()},
        series={}, cash_ret=arrays["cash_ret"][sl],
    )


def test_third_friday_and_decision_date():
    dates = _dates()
    F = _june_friday(dates)
    assert F.astype("datetime64[D]").astype(np.int64) % 7 == 1        # 1970-01-02 was a Friday -> Friday = 1 mod 7
    first = F.astype("datetime64[M]").astype("datetime64[D]")
    assert 14 <= int((F - first).astype(np.int64)) <= 20              # day 15..21 of the month
    for W in (15, 20, 25):
        fn, fprev, cnt, dec = _calendar(dates, W)
        dj = [int(x) for x in dec if fn[x] == F]
        assert len(dj) == 1
        assert np.busday_count(dates[dj[0]], F) == W                 # R is the W-th weekday after d
    # row 0 is never a decision row and the March quarter (inside the data) gets one
    fn, fprev, cnt, dec = _calendar(dates, 20)
    assert 0 not in dec


def test_holiday_on_decision_day_moves_to_next_session():
    dates = _dates()
    F = _june_friday(dates)
    target = np.busday_offset(F, -20)
    d2 = _dates(drop=(target,))
    fn, fprev, cnt, dec = _calendar(d2, 20)
    dj = [int(x) for x in dec if fn[x] == F]
    assert len(dj) == 1 and np.busday_count(d2[dj[0]], F) == 19


def test_legs_and_weights():
    arrays, d, F = build()
    w = target_weights(view(arrays), PARAMS)
    row = w[d]
    longs = np.flatnonzero(row > 0)
    shorts = np.flatnonzero(row < 0)
    assert list(longs) == list(range(10, 60))                         # top 50 of 60 increases
    assert list(shorts) == list(range(60, 75))                        # all 15 decreases
    assert np.allclose(row[longs], 0.5 / 50)
    assert np.allclose(row[shorts], -0.5 / 15)
    for j in (75, 76, 77, 78, 79, 80, 81):                            # floor, cap, same filing, split, member, tradable, NaN
        assert row[j] == 0.0
    assert abs(row.sum()) < 1e-12
    assert np.abs(w).sum(axis=1).max() <= 1.0 + 1e-12


def test_holding_window_and_exit():
    arrays, d, F = build()
    w = target_weights(view(arrays), PARAMS)
    dates = arrays["dates"]
    r_row = int(np.flatnonzero(dates == F)[0])
    assert r_row - d == 20
    assert np.all(w[:d] == 0.0)
    for t in range(d, r_row):
        assert np.array_equal(w[t], w[d])                            # unchanged through the close of R
    assert np.all(w[r_row:] == 0.0)                                   # flat after the close of R


def test_name_leaving_universe_is_closed_not_redistributed():
    arrays, d, F = build(leave=(3, 5, 70))
    w = target_weights(view(arrays), PARAMS)
    dates = arrays["dates"]
    r_row = int(np.flatnonzero(dates == F)[0])
    assert w[d + 2, 70] < 0
    assert np.all(w[d + 3:r_row, 70] == 0.0)                          # closed, and stays closed after re-entry
    others = np.ones(N, dtype=bool)
    others[70] = False
    assert np.array_equal(w[d + 3, others], w[d, others])


def test_skip_quarter_with_fewer_than_10_on_a_side():
    arrays, d, F = build(g_short_count=9)
    w = target_weights(view(arrays), PARAMS)
    assert np.all(w == 0.0)
    arrays, d, F = build(g_short_count=10)
    w = target_weights(view(arrays), PARAMS)
    assert (w[d] < 0).sum() == 10


def test_window_param_moves_decision_date():
    arrays, d, F = build()
    v = view(arrays)
    dates = arrays["dates"]
    r_row = int(np.flatnonzero(dates == F)[0])
    for W in (15, 25):
        w = target_weights(v, {"window_sessions": W})
        first = int(np.flatnonzero(np.abs(w).sum(axis=1) > 0)[0])
        assert r_row - first == W


def test_point_in_time_truncation():
    arrays, d, F = build()
    full = target_weights(view(arrays), PARAMS)
    for cut in (d - 1, d, d + 1, d + 7, d + 20, d + 21):
        part = target_weights(view(arrays, upto=cut + 1), PARAMS)
        assert np.array_equal(part, full[:cut + 1])


def test_no_universe_raises():
    arrays, d, F = build()
    v = view(arrays)
    v2 = DataView(
        dates=v.dates, instruments=v.instruments, asset_class=v.asset_class, ret_co=v.ret_co, ret_oc=v.ret_oc,
        tradable=v.tradable, listed=v.listed, delisting=v.delisting, close=v.close,
        dollar_volume=v.dollar_volume, universe=None, extras=v.extras, series={}, cash_ret=v.cash_ret)
    with pytest.raises(Exception):
        target_weights(v2, PARAMS)


def test_legs_match_weights():
    # diagnostic.py copies _legs verbatim; the event set is exactly the leg entries at d
    arrays, d, F = build()
    v = view(arrays)
    legs = _legs(v, PARAMS)
    assert len(legs) == 1
    dd, e, lc, sc = legs[0]
    assert dd == d and lc.size == 50 and sc.size == 15
    w = target_weights(v, PARAMS)
    assert set(np.flatnonzero(w[d] > 0)) == set(lc)
    assert set(np.flatnonzero(w[d] < 0)) == set(sc)
