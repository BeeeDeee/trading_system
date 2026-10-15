import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS, events, top_n_universe

N = 10
NAMES = tuple("C%dUSDT.P" % j for j in range(N))
# funding per name per day; low-R half = names 0..4, high-R half = names 5..9
BASE_F = np.array([5, 1, 3, 2, 4, 9, 7, 8, 6, 10], dtype=float) * 1e-4


def make(T=40, fund=None, close=None, dv=None, uni=None, no_uni=False):
    dates = np.arange(T).astype("timedelta64[D]") + np.zeros(1, dtype="datetime64[D]")
    ret_oc = np.tile(0.001 * np.arange(N), (T, 1)).astype(np.float32)   # R ascending with the column
    ret_co = np.zeros((T, N), dtype=np.float32)
    if fund is None:
        fund = np.tile(BASE_F, (T, 1))
    if close is None:
        close = np.ones((T, N))
    if dv is None:
        dv = np.ones((T, N)) * 1e6
    if uni is None:
        uni = np.ones((T, N), dtype=bool)
    ok = ~np.isnan(close)
    return DataView(
        dates=dates,
        instruments=NAMES,
        asset_class=tuple("crypto_perp" for _ in range(N)),
        ret_co=np.where(ok, ret_co, 0).astype(np.float32),
        ret_oc=np.where(ok, ret_oc, 0).astype(np.float32),
        tradable=ok.copy(),
        listed=ok.copy(),
        delisting=np.zeros((T, N), dtype=bool),
        close=close.astype(np.float32),
        dollar_volume=dv.astype(np.float32),
        universe=None if no_uni else uni,
        extras={"funding": fund.astype(np.float32), "basis": np.zeros((T, N), dtype=np.float32)},
        series={},
        cash_ret=np.zeros(T),
    )


def expected_book():
    w = np.zeros(N)
    w[[1, 8]] = 0.25      # lowest F in low half (1) and high half (8)
    w[[0, 9]] = -0.25     # highest F in low half (0) and high half (9)
    return w


def test_legs_and_schedule():
    d = make()
    w = target_weights(d, PARAMS)
    # first row with >= 8 eligible: price on 7 rows -> row 6; rebalance every 5 rows from there
    assert np.all(np.isnan(w[:6]))
    for t in range(6, 40):
        if (t - 6) % 5 == 0:
            np.testing.assert_allclose(w[t], expected_book())
        else:
            assert np.all(np.isnan(w[t]))
    rows = w[~np.isnan(w).all(axis=1)]
    assert np.all(np.nansum(np.abs(rows), axis=1) <= 1.0 + 1e-12)
    np.testing.assert_allclose(np.nansum(rows, axis=1), 0.0, atol=1e-12)


def test_tie_break_by_liquidity():
    fund = np.tile(np.full(N, 3e-4), (40, 1))   # all funding equal
    dv = np.tile(np.arange(1, N + 1, dtype=float) * 1e6, (40, 1))
    dv[:, 2] = 1e9   # most liquid in the low half
    d = make(fund=fund, dv=dv)
    w = target_weights(d, PARAMS)[6]
    # all F tied: short = most liquid, long = next most liquid among the rest (never the same name)
    # low half 0..4: liquidity order 2, 4, 3, 1, 0 -> short 2, long 4
    # high half 5..9: liquidity order 9, 8, 7, 6, 5 -> short 9, long 8
    exp = np.zeros(N)
    exp[[2, 9]] = -0.25
    exp[[4, 8]] = 0.25
    np.testing.assert_allclose(w, exp)
    assert np.sum(w) == 0.0


def test_tie_break_distinct_legs():
    f = BASE_F.copy()
    f[[0, 4]] = 5e-4      # tie for the highest F in the low half
    f[[1, 3]] = 1e-4      # tie for the lowest F in the low half
    fund = np.tile(f, (40, 1))
    dv = np.ones((40, N)) * 1e6
    dv[:, 4] = 2e6        # 4 more liquid than 0
    dv[:, 3] = 2e6        # 3 more liquid than 1
    w = target_weights(make(fund=fund, dv=dv), PARAMS)[6]
    assert w[4] == -0.25 and w[0] == 0.0
    assert w[3] == 0.25 and w[1] == 0.0


def test_exit_when_leaving_universe_no_renormalisation():
    uni = np.ones((40, N), dtype=bool)
    uni[8:, 1] = False
    w = target_weights(make(uni=uni), PARAMS)
    assert w[8, 1] == 0.0
    assert np.all(np.isnan(np.delete(w[8], 1)))
    assert np.all(np.isnan(w[9]))
    # next rebalance (row 11): name 1 not eligible; 9 eligible -> low half 0,2,3,4 (k=1), high 5..9 (k=1)
    exp = np.zeros(N)
    exp[3] = 0.25   # lowest F among 0,2,3,4
    exp[0] = -0.25
    exp[8] = 0.25
    exp[9] = -0.25
    np.testing.assert_allclose(w[11], exp)


def test_exit_when_price_lost():
    close = np.ones((40, N))
    close[13, 9] = np.nan
    w = target_weights(make(close=close), PARAMS)
    assert w[13, 9] == 0.0
    assert np.all(np.isnan(np.delete(w[13], 9)))


def test_flat_when_fewer_than_eight_eligible():
    uni = np.ones((40, N), dtype=bool)
    uni[10:, :3] = False      # 7 eligible from row 10
    w = target_weights(make(uni=uni), PARAMS)
    np.testing.assert_allclose(w[11], np.zeros(N))
    np.testing.assert_allclose(w[16], np.zeros(N))


def test_late_listing_and_funding_window():
    close = np.ones((40, N))
    close[:20, 0] = np.nan          # name 0 listed at row 20
    fund = np.tile(BASE_F, (40, 1))
    fund[:20, 0] = np.nan
    w = target_weights(make(close=close, fund=fund), PARAMS)
    # 9 eligible from row 6: name 0 must not be held before it has 7 days of prices (row 26)
    for t in range(6, 26):
        if not np.isnan(w[t, 0]):
            assert w[t, 0] == 0.0
    np.testing.assert_allclose(w[26], expected_book())


def test_funding_days_changes_score():
    fund = np.tile(BASE_F, (40, 1))
    # name 2 has a funding spike two days before row 16: counts with funding_days 3, not with 1
    fund[14, 2] = 50e-4
    p1 = dict(PARAMS, funding_days=1)
    p3 = dict(PARAMS, funding_days=3)
    d = make(fund=fund)
    w1 = target_weights(d, p1)
    w3 = target_weights(d, p3)
    assert w1[16, 2] == 0.0 and w1[16, 0] == -0.25
    assert w3[16, 2] == -0.25 and w3[16, 0] == 0.0


def test_hold_days_changes_schedule():
    d = make()
    w3 = target_weights(d, dict(PARAMS, hold_days=3))
    w7 = target_weights(d, dict(PARAMS, hold_days=7))
    dec3 = np.flatnonzero(~np.isnan(w3).all(axis=1))
    dec7 = np.flatnonzero(~np.isnan(w7).all(axis=1))
    np.testing.assert_array_equal(dec3, np.arange(6, 40, 3))
    np.testing.assert_array_equal(dec7, np.arange(6, 40, 7))


def test_point_in_time_truncation():
    rng = np.random.default_rng(0)
    T = 60
    fund = rng.normal(1e-4, 3e-4, (T, N))
    d_full = make(T=T, fund=fund)
    w_full = target_weights(d_full, PARAMS)
    for cut in (15, 23, 37, 50):
        d_cut = make(T=cut, fund=fund[:cut])
        w_cut = target_weights(d_cut, PARAMS)
        np.testing.assert_array_equal(np.isnan(w_cut), np.isnan(w_full[:cut]))
        np.testing.assert_allclose(np.nan_to_num(w_cut), np.nan_to_num(w_full[:cut]))


def test_diagnostic_events_every_day():
    d = make()
    mask, side = events(d, PARAMS)
    assert not mask[:6].any()
    for t in range(6, 40):
        assert mask[t].sum() == 4
        np.testing.assert_allclose(side[t], np.sign(expected_book()))


def test_top_n_universe_by_trailing_30d_volume():
    T, M = 40, 25
    close = np.ones((T, M))
    dv = np.tile(np.arange(1, M + 1, dtype=float), (T, 1))   # column j has volume j + 1
    listed = np.ones((T, M), dtype=bool)
    u = top_n_universe(close, dv, listed)
    assert u.shape == (T, M)
    assert np.all(u.sum(axis=1) == 20)
    assert np.all(u[:, 5:]) and not u[:, :5].any()
    # name 24 loses its price at row 10 -> out at row 10, name 4 takes its place
    close[10, 24] = np.nan
    u = top_n_universe(close, dv, listed)
    assert not u[10, 24] and u[10, 4] and u[11, 24]
    # name 0 has a volume spike at row 20: 30-day sum 100 + 20*1 >> 25*30 -> member from row 20 for 30 rows
    dv2 = dv.copy()
    dv2[20, 0] = 1e4
    u2 = top_n_universe(np.ones((T, M)), dv2, listed)
    assert not u2[19, 0] and u2[20, 0] and u2[39, 0]
    # point-in-time: rows before the spike unchanged
    np.testing.assert_array_equal(u2[:20], top_n_universe(np.ones((T, M)), dv, listed)[:20])
    # a not-listed name is never a member
    listed[:, 24] = False
    assert not top_n_universe(np.ones((T, M)), dv, listed)[:, 24].any()


def test_no_universe_mask_builds_top_n():
    # with no gate mask the 10 names are all in the top 20 -> same book as with an all-true mask
    d = make()
    d0 = make(no_uni=True)
    assert d0.universe is None
    np.testing.assert_array_equal(np.isnan(target_weights(d0, PARAMS)), np.isnan(target_weights(d, PARAMS)))
    np.testing.assert_allclose(np.nan_to_num(target_weights(d0, PARAMS)), np.nan_to_num(target_weights(d, PARAMS)))
    m0, s0 = events(d0, PARAMS)
    m1, s1 = events(d, PARAMS)
    np.testing.assert_array_equal(m0, m1)
    np.testing.assert_allclose(s0, s1)


def test_no_universe_mask_holds_only_top_20():
    rng = np.random.default_rng(1)
    T, M = 60, 30
    names = tuple("X%dUSDT.P" % j for j in range(M))
    dates = np.arange(T).astype("timedelta64[D]") + np.zeros(1, dtype="datetime64[D]")
    close = np.ones((T, M))
    dv = rng.uniform(1e5, 1e7, (T, M))
    d = DataView(
        dates=dates, instruments=names, asset_class=tuple("crypto_perp" for _ in range(M)),
        ret_co=np.zeros((T, M), dtype=np.float32),
        ret_oc=rng.normal(0, 0.02, (T, M)).astype(np.float32),
        tradable=np.ones((T, M), dtype=bool), listed=np.ones((T, M), dtype=bool),
        delisting=np.zeros((T, M), dtype=bool), close=close.astype(np.float32),
        dollar_volume=dv.astype(np.float32), universe=None,
        extras={"funding": rng.normal(1e-4, 3e-4, (T, M)).astype(np.float32),
                "basis": np.zeros((T, M), dtype=np.float32)},
        series={}, cash_ret=np.zeros(T),
    )
    u = top_n_universe(close, np.asarray(d.dollar_volume, dtype=np.float64), np.ones((T, M), dtype=bool))
    w = target_weights(d, PARAMS)
    dec = ~np.isnan(w).all(axis=1)
    assert dec.any()
    for t in np.flatnonzero(dec):
        held = np.nan_to_num(w[t]) != 0.0
        assert np.all(u[t][held])
        assert np.sum(np.abs(np.nan_to_num(w[t]))) <= 1.0 + 1e-12
    m, _ = events(d, PARAMS)
    assert m.any() and not (m & ~u).any()


def test_missing_funding_input_raises():
    d = make()
    d.extras.pop("funding")
    with pytest.raises(KeyError):
        target_weights(d, PARAMS)
