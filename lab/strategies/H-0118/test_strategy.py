import numpy as np
import pytest

from strategy import (PARAMS, build_book, deadline_clocks, in_tax_window, leg_weights,
                      stock_features, stock_features_blocked, target_weights, _holidays)


def ymd(y, m, d):
    base = np.array([y - 1970]).astype("datetime64[Y]").astype("datetime64[M]")
    base = base + np.array(m - 1).astype("timedelta64[M]")
    return (base.astype("datetime64[D]") + np.array(d - 1).astype("timedelta64[D]"))[0]


def sessions(y0, y1):
    """Weekdays of years y0..y1 minus the rule holidays (a test XNYS-like calendar)."""
    start = ymd(y0, 1, 1)
    end = ymd(y1 + 1, 1, 1)
    days = np.arange(start, end, dtype="datetime64[D]")
    wd = (days.astype(np.int64) + 3) % 7
    days = days[wd < 5]
    hol = _holidays(days)
    return days[~np.isin(days, hol)]


def row(dates, y, m, d):
    hit = np.flatnonzero(dates == ymd(y, m, d))
    assert hit.size == 1
    return int(hit[0])


def test_params_match_card():
    assert PARAMS == {"leg": 0.1, "n_estimators": 100}


def test_holiday_rules():
    hol = _holidays(sessions(2011, 2012))
    assert ymd(2012, 9, 3) in hol      # Labor Day 2012
    assert ymd(2012, 11, 22) in hol    # Thanksgiving 2012
    assert ymd(2012, 12, 25) in hol    # Christmas 2012 (Tuesday)
    assert ymd(2011, 12, 26) in hol    # Christmas 2011 Sunday -> Monday
    assert ymd(2011, 11, 24) in hol    # Thanksgiving 2011


def test_clocks_hand_derived():
    d = sessions(2011, 2013)
    ct, cs, lf, li = deadline_clocks(d)
    # 2012-10-31 (Wednesday) is the October deadline
    assert ct[row(d, 2012, 10, 31)] == 0 and cs[row(d, 2012, 10, 31)] == 0
    assert ct[row(d, 2012, 10, 30)] == 1
    assert ct[row(d, 2012, 10, 26)] == 3
    assert cs[row(d, 2012, 11, 1)] == 1
    assert cs[row(d, 2012, 11, 21)] == 15
    # December 2012: Christmas Tuesday, deadline Monday Dec 31
    assert ct[row(d, 2012, 12, 24)] == 4
    assert ct[row(d, 2012, 11, 21)] == 26
    assert ct[row(d, 2012, 12, 31)] == 0
    # December 2011: Dec 31 is a Saturday -> deadline Friday Dec 30; Dec 26 holiday
    assert ct[row(d, 2011, 12, 23)] == 4
    assert ct[row(d, 2011, 12, 30)] == 0
    assert cs[row(d, 2012, 1, 3)] == 2       # Jan 2 2012 is a session in this test calendar
    # caps and unknown
    assert ct[row(d, 2012, 3, 1)] == 63
    assert cs[row(d, 2012, 6, 1)] == 63
    assert np.isnan(cs[row(d, 2011, 6, 1)])   # no deadline in view yet
    # references strictly before t
    assert li[row(d, 2012, 3, 1)] == row(d, 2011, 12, 30)
    assert lf[row(d, 2012, 3, 1)] == row(d, 2011, 10, 31)
    assert li[row(d, 2012, 12, 31)] == row(d, 2011, 12, 30)
    assert li[row(d, 2013, 1, 2)] == row(d, 2012, 12, 31)
    assert lf[row(d, 2012, 10, 31)] == row(d, 2011, 10, 31)
    assert lf[row(d, 2012, 11, 1)] == row(d, 2012, 10, 31)


def test_window():
    d = sessions(2011, 2013)
    ct, cs, _, _ = deadline_clocks(d)
    w = in_tax_window(ct, cs)
    assert w[row(d, 2012, 10, 31)]           # deadline day: pressure phase
    assert w[row(d, 2012, 11, 21)]           # 15 sessions after
    assert not w[row(d, 2012, 11, 23)]       # 16 sessions after
    assert not w[row(d, 2012, 6, 1)]
    i = row(d, 2012, 10, 31)
    assert w[i - 15] and not w[i - 16]


def test_clocks_point_in_time():
    d = sessions(2011, 2013)
    full = deadline_clocks(d)
    for cut in (200, 455, 456, 457, 500, 640):
        part = deadline_clocks(d[:cut])
        for a, b in zip(full, part):
            np.testing.assert_array_equal(a[:cut], b)


def test_stock_features_by_hand():
    rng = np.random.default_rng(0)
    T, N = 320, 3
    r = rng.normal(0, 0.01, (T, N))
    tri = np.cumprod(1 + r, axis=0)
    dv = rng.uniform(1e6, 2e6, (T, N))
    listed = np.ones((T, N), bool)
    listed[:100, 2] = False                   # late listing
    tri[:100, 2] = np.nan
    lf = np.full(T, -1)
    li = np.full(T, -1)
    li[260:] = 255
    lf[260:] = 200
    f = stock_features(tri, dv, listed, lf, li)
    t = 300
    assert np.isnan(f[0][250, 0])             # fewer than 252 sessions
    w = slice(t - 251, t + 1)
    ref = np.sum(dv[w, 0] * tri[w, 0]) / np.sum(dv[w, 0])
    assert f[2][t, 0] == pytest.approx(np.log(tri[t, 0] / ref), rel=1e-9)
    assert f[3][t, 1] == pytest.approx(tri[t, 1] / tri[t - 21, 1] - 1, rel=1e-9)
    dr = tri[t - 62:t + 1, 1] / tri[t - 63:t, 1] - 1
    assert f[4][t, 1] == pytest.approx(np.std(dr, ddof=1), rel=1e-7)
    assert f[0][t, 0] == pytest.approx(tri[t, 0] / tri[255, 0] - 1, rel=1e-9)
    assert f[1][t, 0] == pytest.approx(tri[t, 0] / tri[200, 0] - 1, rel=1e-9)
    assert np.isnan(f[0][259, 0])             # no December reference -> not scored
    # late listing: 252 listed sessions only from row 351 -> NaN everywhere here
    assert np.all(np.isnan(f[2][:, 2]))
    # column blocks give the same features (float32 storage)
    fb = stock_features_blocked(tri, dv, listed, lf, li, block=2)
    for a, b in zip(f, fb):
        np.testing.assert_allclose(b, a.astype(np.float32), rtol=1e-6, equal_nan=True)
    # point-in-time: truncating the future leaves earlier rows unchanged
    ft = stock_features(tri[:290], dv[:290], listed[:290], lf[:290], li[:290])
    for a, b in zip(f, ft):
        np.testing.assert_array_equal(a[:290], b)


def test_leg_weights_and_floor():
    T, N = 3, 120
    score = np.tile(np.arange(N, dtype=float), (T, 1))
    uni = np.ones((T, N), bool)
    score[1, 40:] = np.nan                    # only 40 scored -> flat
    dec = np.array([True, True, False])
    idx, sw = leg_weights(score, uni, dec, 0.1)
    assert list(idx) == [0, 1]
    assert np.abs(sw[0]).sum() == pytest.approx(1.0)
    assert sw[0][sw[0] > 0].sum() == pytest.approx(0.5)
    assert sw[0][sw[0] < 0].sum() == pytest.approx(-0.5)
    assert np.all(sw[0][-12:] > 0) and np.all(sw[0][:12] < 0)
    assert np.all(sw[1] == 0)
    _, sw2 = leg_weights(score, uni, dec, 0.2)
    assert (sw2[0] > 0).sum() > (sw[0] > 0).sum()


def test_book_hold_close_and_universe_exit():
    T, N = 20, 4
    uni = np.ones((T, N), bool)
    uni[4:, 3] = False                        # col 3 leaves the universe on row 4
    idx = np.array([2, 7])
    sw = np.array([[0.25, -0.25, 0.25, -0.25], [-0.25, 0.25, 0.0, 0.0]])
    out = build_book(T, N, idx, sw, uni)
    assert np.all(np.isnan(out[:2]))
    np.testing.assert_array_equal(out[2], sw[0])
    assert np.all(np.isnan(out[3]))
    assert out[4, 3] == 0 and np.all(np.isnan(out[4, :3]))
    np.testing.assert_array_equal(out[7], sw[1])
    assert np.all(out[12] == 0)               # closed 5 sessions after the last decision
    assert np.all(np.isnan(out[8:12])) and np.all(np.isnan(out[13:]))
    assert np.nanmax(np.abs(out).sum(axis=1)) <= 1.0 + 1e-12


def test_book_consecutive_weeks_no_close():
    T, N = 20, 2
    uni = np.ones((T, N), bool)
    idx = np.array([2, 7, 12])
    sw = np.array([[0.5, -0.5], [-0.5, 0.5], [0.5, -0.5]])
    out = build_book(T, N, idx, sw, uni)
    np.testing.assert_array_equal(out[7], sw[1])
    np.testing.assert_array_equal(out[12], sw[2])
    assert np.all(out[17] == 0)
