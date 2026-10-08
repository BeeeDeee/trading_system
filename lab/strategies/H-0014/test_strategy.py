import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

MONDAY0 = 4 + 7 * 1500  # days since 1970-01-01 (a Thursday); day 4 was a Monday


def weekdays(n):
    days = np.arange(MONDAY0, MONDAY0 + 2 * n, dtype=np.int64)
    days = days[((days + 3) % 7) < 5][:n]
    return days.astype("datetime64[D]")


def make(ret_co, ret_oc, dv, tradable=None, universe=None):
    T, N = dv.shape
    names = tuple("S%02d" % j for j in range(N))
    if tradable is None:
        tradable = np.ones((T, N), dtype=bool)
    if universe is None:
        universe = np.ones((T, N), dtype=bool)
    return DataView(
        dates=weekdays(T),
        instruments=names,
        asset_class=tuple("us_equity" for _ in names),
        ret_co=np.asarray(ret_co, dtype=float),
        ret_oc=np.asarray(ret_oc, dtype=float),
        tradable=np.asarray(tradable, dtype=bool),
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=np.full((T, N), 10.0),
        dollar_volume=np.asarray(dv, dtype=float),
        universe=np.asarray(universe, dtype=bool),
        extras={},
        series={},
        cash_ret=np.zeros(T),
    )


def sliced(d, t):
    """The same market truncated after row t."""
    return make(d.ret_co[: t + 1], d.ret_oc[: t + 1], d.dollar_volume[: t + 1],
                d.tradable[: t + 1], d.universe[: t + 1])


P = {"leg_quantile": 0.2, "reference_days": 10}
T0 = 14  # first Friday row with a full window (4 + reference_days = 14)


def hand_market(T=30, N=10):
    """dv = 1 everywhere; on rows 10..14 odd stocks trade 2.0 and even ones 0.5.
    Row 14: ret_oc = 0.01 * j, so R5 rises with j and the R5 quintiles are {0,1}, {2,3}, ..."""
    ret_co = np.zeros((T, N))
    ret_oc = np.zeros((T, N))
    ret_oc[T0] = 0.01 * np.arange(N)
    dv = np.ones((T, N))
    for j in range(N):
        dv[T0 - 4:T0 + 1, j] = 2.0 if j % 2 else 0.5
    return ret_co, ret_oc, dv


def random_market(T=400, N=60, seed=0):
    rng = np.random.default_rng(seed)
    ret_co = rng.normal(0, 0.005, (T, N))
    ret_oc = rng.normal(0, 0.015, (T, N))
    dv = np.exp(rng.normal(15, 0.7, (T, N)))
    dv[rng.random((T, N)) < 0.05] = np.nan
    uni = rng.random((T, N)) < 0.85
    trad = rng.random((T, N)) < 0.97
    dv[:40, 5] = np.nan  # late listing
    ret_co[:40, 5] = 0.0
    ret_oc[:40, 5] = 0.0
    return make(ret_co, ret_oc, dv, trad, uni)


def test_params_match_card():
    assert PARAMS == {"leg_quantile": 0.2, "reference_days": 50}


def test_calendar_fridays_only():
    d = make(*hand_market())
    w = target_weights(d, P)
    wd = (d.dates.astype(np.int64) + 3) % 7
    fri = wd == 4
    assert np.isnan(w[~fri]).all()
    assert not np.isnan(w[fri]).any()
    # Fridays before the windows are full (rows 4, 9): flat
    assert (w[4] == 0).all() and (w[9] == 0).all()


def test_hand_example_legs():
    d = make(*hand_market())
    w = target_weights(d, P)
    odd = np.arange(10) % 2 == 1
    np.testing.assert_allclose(w[T0, odd], 0.1)
    np.testing.assert_allclose(w[T0, ~odd], -0.1)


def test_recent_window_coverage():
    ret_co, ret_oc, dv = hand_market()
    dv[T0 - 2:T0, 9] = np.nan  # 3 of 5 recent days: ineligible
    w = target_weights(make(ret_co, ret_oc, dv), P)[T0]
    # 9 eligible: quintiles {0,1},{2,3},{4,5},{6,7},{8} (the last has one name and is skipped)
    np.testing.assert_allclose(w[[1, 3, 5, 7]], 0.125)
    np.testing.assert_allclose(w[[0, 2, 4, 6]], -0.125)
    assert w[8] == 0 and w[9] == 0


def test_recent_window_80pct_is_enough():
    ret_co, ret_oc, dv = hand_market()
    dv[T0 - 1, 9] = np.nan  # 4 of 5 = 80 %: still eligible
    w = target_weights(make(ret_co, ret_oc, dv), P)[T0]
    np.testing.assert_allclose(w[9], 0.1)


def test_reference_window_coverage():
    ret_co, ret_oc, dv = hand_market()
    dv[0:3, 3] = np.nan  # 7 of 10 reference days < 80 %
    w = target_weights(make(ret_co, ret_oc, dv), P)[T0]
    assert w[3] == 0
    dv2 = hand_market()[2]
    dv2[0:2, 3] = np.nan  # 8 of 10: eligible
    w2 = target_weights(make(ret_co, ret_oc, dv2), P)[T0]
    np.testing.assert_allclose(w2[3], 0.1)


def test_missing_returns_not_eligible():
    ret_co, ret_oc, dv = hand_market()
    ret_co[T0 - 3, 7] = np.nan
    w = target_weights(make(ret_co, ret_oc, dv), P)[T0]
    assert w[7] == 0
    assert abs(np.nansum(w)) < 1e-12


def test_universe_and_tradable_filter():
    ret_co, ret_oc, dv = hand_market()
    uni = np.ones_like(dv, dtype=bool)
    trad = np.ones_like(dv, dtype=bool)
    uni[T0, 5] = False
    trad[T0, 2] = False
    w = target_weights(make(ret_co, ret_oc, dv, trad, uni), P)[T0]
    assert w[5] == 0 and w[2] == 0
    np.testing.assert_allclose(w[w > 0].sum(), 0.5)
    np.testing.assert_allclose(w[w < 0].sum(), -0.5)


def test_av_uses_ratio_to_own_history():
    # a stock with a high level but no change in volume is not "abnormal"
    ret_co, ret_oc, dv = hand_market()
    dv[:, 0] *= 1000.0  # scales both windows: AV unchanged, still the short side of its quintile
    w = target_weights(make(ret_co, ret_oc, dv), P)[T0]
    np.testing.assert_allclose(w[0], -0.1)


def test_r5_compounds_co_and_oc():
    # stock 0 jumps to the top R5 quintile via an overnight return on t-4
    ret_co, ret_oc, dv = hand_market()
    ret_co[T0 - 4, 0] = 0.5
    w = target_weights(make(ret_co, ret_oc, dv), P)[T0]
    # order by R5: 1,2,...,9,0 -> quintiles {1,2},{3,4},{5,6},{7,8},{9,0}
    np.testing.assert_allclose(w[[1, 3, 5, 7, 9]], 0.1)
    np.testing.assert_allclose(w[[2, 4, 6, 8, 0]], -0.1)


def test_random_market_invariants():
    d = random_market()
    w = target_weights(d, PARAMS)
    rows = ~np.isnan(w).all(axis=1)
    assert rows.sum() > 50
    for t in np.flatnonzero(rows):
        r = w[t]
        assert np.abs(r).sum() <= 1.0 + 1e-9
        if (r != 0).any():
            np.testing.assert_allclose(r[r > 0].sum(), 0.5)
            np.testing.assert_allclose(r[r < 0].sum(), -0.5)
            assert (r[d.universe[t] == 0] == 0).all()
            assert (r[d.tradable[t] == 0] == 0).all()
    # late listing (first dv on row 40): the reference window t-54..t-5 needs >= 40 present days,
    # so stock 5 cannot be eligible before row 84
    assert (np.nan_to_num(w[:84, 5]) == 0).all()
    assert (np.nan_to_num(w[84:, 5]) != 0).any()


def test_point_in_time_truncation():
    d = random_market(T=250, N=40, seed=1)
    full = target_weights(d, PARAMS)
    for t in (60, 104, 149, 200, 249):
        part = target_weights(sliced(d, t), PARAMS)
        np.testing.assert_array_equal(np.nan_to_num(part, nan=9.0), np.nan_to_num(full[: t + 1], nan=9.0))


def test_parameters_change_output():
    d = random_market(seed=2)
    base = target_weights(d, PARAMS)
    q = target_weights(d, dict(PARAMS, leg_quantile=0.3))
    r = target_weights(d, dict(PARAMS, reference_days=25))
    n_base = (np.nan_to_num(base) > 0).sum()
    n_q = (np.nan_to_num(q) > 0).sum()
    assert n_q > n_base
    assert not np.array_equal(np.nan_to_num(base), np.nan_to_num(r))


def test_leg_quantile_rounding():
    # 20 stocks -> 4 per quintile: 0.2 -> 1 per side (floor 0.8, min 1); 0.5 -> 2 per side
    T, N = 30, 20
    ret_co = np.zeros((T, N))
    ret_oc = np.zeros((T, N))
    ret_oc[T0] = 0.01 * np.arange(N)
    dv = np.ones((T, N))
    for j in range(N):
        dv[T0 - 4:T0 + 1, j] = 1.0 + 0.1 * (j % 4)
    d = make(ret_co, ret_oc, dv)
    w = target_weights(d, P)[T0]
    assert (w > 0).sum() == 5 and (w < 0).sum() == 5
    assert set(np.flatnonzero(w > 0) % 4) == {3}
    assert set(np.flatnonzero(w < 0) % 4) == {0}
    w5 = target_weights(d, dict(P, leg_quantile=0.5))[T0]
    assert (w5 > 0).sum() == 10 and (w5 < 0).sum() == 10
    np.testing.assert_allclose(w5[w5 > 0], 0.05)


def test_no_universe_fails_loudly():
    ret_co, ret_oc, dv = hand_market()
    d = make(ret_co, ret_oc, dv)
    d2 = DataView(dates=d.dates, instruments=d.instruments, asset_class=d.asset_class, ret_co=d.ret_co,
                  ret_oc=d.ret_oc, tradable=d.tradable, listed=d.listed, delisting=d.delisting, close=d.close,
                  dollar_volume=d.dollar_volume, universe=None, extras={}, series={}, cash_ret=d.cash_ret)
    with pytest.raises(ValueError):
        target_weights(d2, P)
