import math

import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import abnormal_returns, pre_event_sigma, reaction_z, event_sides, book_weights


def test_params_match_card():
    assert PARAMS == {"hold_days": 5, "z_threshold": 2.0}


def test_abnormal_returns_member_mean_and_validity():
    ret_co = np.array([[0.01, 0.00, 0.02], [0.0, 0.0, 0.0]])
    ret_oc = np.array([[0.01, 0.02, 0.00], [0.03, 0.01, 0.5]])
    valid = np.array([[True, True, True], [True, True, False]])
    universe = np.array([[True, True, False], [True, True, True]])
    a = abnormal_returns(ret_co, ret_oc, valid, universe)
    r0 = 1.01 * 1.01 - 1.0
    r1 = 0.02
    r2 = 0.02
    m = (r0 + r1) / 2.0                      # non-member stock 2 not in the mean
    assert a[0, 0] == pytest.approx(r0 - m)
    assert a[0, 2] == pytest.approx(r2 - m)  # abnormal return also defined for non-members
    m1 = (0.03 + 0.01) / 2.0                 # invalid stock 2 excluded from the mean
    assert a[1, 0] == pytest.approx(0.03 - m1)
    assert np.isnan(a[1, 2])


def test_sigma_window_ends_at_t_minus_2():
    rng = np.random.default_rng(0)
    T = 120
    a = rng.normal(0, 0.02, size=(T, 2))
    sig = pre_event_sigma(a)
    for t in (70, 90, 119):
        assert sig[t, 0] == pytest.approx(np.std(a[t - 64:t - 1, 0], ddof=1), rel=1e-9)
    # changing rows t-1 and t does not change sigma[t]
    b = a.copy()
    b[99:101] = 5.0
    sig_b = pre_event_sigma(b)
    assert sig_b[100, 0] == pytest.approx(sig[100, 0], rel=1e-12)
    # first rows: fewer than 40 values -> NaN; 40 values at t = 41
    assert np.all(np.isnan(sig[:41]))
    assert np.isfinite(sig[41, 0])
    assert sig[41, 0] == pytest.approx(np.std(a[0:40, 0], ddof=1), rel=1e-9)


def test_sigma_needs_40_valid_values():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.01, size=(100, 1))
    a[20:50, 0] = np.nan                     # window 80-63-1.. leaves < 40 valid
    sig = pre_event_sigma(a)
    t = 80                                   # window rows 16..78: 63 - 30 = 33 valid
    assert np.isnan(sig[t, 0])
    t = 99                                   # window rows 35..97: 63 - 15 = 48 valid
    w = a[35:98, 0]
    assert sig[t, 0] == pytest.approx(np.std(w[np.isfinite(w)], ddof=1), rel=1e-9)


def test_reaction_z_two_sessions():
    a = np.array([[0.0], [0.03], [0.05]])
    sigma = np.array([[0.01], [0.01], [0.01]])
    z = reaction_z(a, sigma)
    assert np.isnan(z[0, 0])
    assert z[2, 0] == pytest.approx((1.03 * 1.05 - 1.0) / (0.01 * math.sqrt(2.0)))


def test_event_sides_threshold_nan_membership():
    z = np.array([[2.2, -2.2, 2.2, 2.2, 1.9]])
    ea = np.array([[1.0, 1.0, np.nan, 1.0, 1.0]])
    uni = np.array([[True, True, True, False, True]])
    trd = np.ones((1, 5), dtype=bool)
    lo, sh = event_sides(ea, uni, trd, z, 2.0)
    assert lo.tolist() == [[True, False, False, False, False]]
    assert sh.tolist() == [[False, True, False, False, False]]
    lo, sh = event_sides(ea, uni, trd, z, 2.5)
    assert not lo.any() and not sh.any()
    lo, sh = event_sides(ea, uni, trd, z, 1.5)
    assert lo[0, 4]
    trd[0, 0] = False
    lo, sh = event_sides(ea, uni, trd, z, 2.0)
    assert not lo[0, 0]


def _empty(T, N):
    return np.zeros((T, N), dtype=bool), np.zeros((T, N), dtype=bool)


def test_hold_window_and_hedge():
    T, N = 10, 30
    lo, sh = _empty(T, N)
    lo[2, 0] = True
    uni = np.ones((T, N), dtype=bool)
    trd = np.ones((T, N), dtype=bool)
    W = book_weights(lo, sh, uni, trd, 5)
    assert np.all(W[:2] == 0)
    for t in range(2, 7):
        assert W[t, 0] == pytest.approx(0.05)
        assert np.allclose(W[t, 1:], -0.05 / 29)
        assert W[t].sum() == pytest.approx(0.0)
    assert np.all(W[7:] == 0)
    W3 = book_weights(lo, sh, uni, trd, 3)
    assert W3[4, 0] == pytest.approx(0.05) and np.all(W3[5] == 0)


def test_replacement_restarts_clock_and_drop():
    T, N = 12, 30
    lo, sh = _empty(T, N)
    lo[1, 0] = True
    sh[3, 0] = True
    uni = np.ones((T, N), dtype=bool)
    trd = np.ones((T, N), dtype=bool)
    W = book_weights(lo, sh, uni, trd, 5)
    assert W[2, 0] == pytest.approx(0.05)
    for t in range(3, 8):
        assert W[t, 0] == pytest.approx(-0.05)
    assert np.all(W[8:] == 0)
    # leaving the universe drops the stock for good
    uni2 = uni.copy()
    uni2[5, 0] = False
    W2 = book_weights(lo, sh, uni2, trd, 5)
    assert np.all(W2[5:, 0] == 0)
    assert np.all(W2[5:] == 0)
    # untradable at t drops it too
    trd2 = trd.copy()
    trd2[4, 0] = False
    W3 = book_weights(lo, sh, uni, trd2, 5)
    assert np.all(W3[4:, 0] == 0)


def test_divisor_floor_gross_and_neutrality():
    T, N = 3, 60
    lo, sh = _empty(T, N)
    lo[0, :25] = True
    sh[0, 25:28] = True
    uni = np.ones((T, N), dtype=bool)
    uni[:, 59] = False
    trd = np.ones((T, N), dtype=bool)
    trd[:, 58] = False
    W = book_weights(lo, sh, uni, trd, 5)
    assert W[0, 0] == pytest.approx(0.5 / 25)
    assert W[0, 25] == pytest.approx(-0.5 / 10)
    net = 0.5 - 0.15
    assert W[0, 30] == pytest.approx(-net / 30)    # hedge: 60 - 28 events - 1 non-member - 1 untradable
    assert W[0, 58] == 0 and W[0, 59] == 0
    assert W[0].sum() == pytest.approx(0.0)
    assert np.abs(W[0]).sum() <= 1.0 + 1e-12


def test_max_gross_one_leg_full():
    T, N = 2, 40
    lo, sh = _empty(T, N)
    lo[0, :20] = True
    uni = np.ones((T, N), dtype=bool)
    trd = np.ones((T, N), dtype=bool)
    W = book_weights(lo, sh, uni, trd, 5)
    assert np.abs(W[0]).sum() == pytest.approx(1.0)


def test_no_events_flat():
    lo, sh = _empty(5, 30)
    W = book_weights(lo, sh, np.ones((5, 30), bool), np.ones((5, 30), bool), 5)
    assert np.all(W == 0)


def _view(ret_co, ret_oc, ea, universe, tradable):
    from lab.framework.data import DataView, synthetic
    T, N = ret_co.shape
    dates = synthetic().dates[:T]
    names = tuple("E%d" % (100 + j) for j in range(N))
    return DataView(
        dates=dates, instruments=names, asset_class=tuple("us_equity" for _ in names),
        ret_co=ret_co, ret_oc=ret_oc, tradable=tradable, listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool), close=np.full((T, N), 50.0),
        dollar_volume=np.full((T, N), 1e8), universe=universe,
        extras={"ea_reaction_day": ea.astype(np.float32)}, series={}, cash_ret=np.zeros(T),
    )


def test_end_to_end_planted_events():
    rng = np.random.default_rng(7)
    T, N = 120, 30
    ret_co = np.zeros((T, N))
    ret_oc = rng.normal(0, 0.01, (T, N))
    ea = np.zeros((T, N))
    ea[:10, 3] = np.nan                       # before the first filing
    ret_oc[100, 0] = 0.15; ea[100, 0] = 1     # large positive reaction -> long
    ret_oc[100, 1] = -0.15; ea[100, 1] = 1    # large negative reaction -> short
    ret_oc[100, 2] = 0.001; ea[100, 2] = 1    # small reaction -> nothing
    ret_oc[30, 4] = 0.15; ea[30, 4] = 1       # sigma has < 40 values -> nothing
    uni = np.ones((T, N), dtype=bool)
    trd = np.ones((T, N), dtype=bool)
    W = target_weights(_view(ret_co, ret_oc, ea, uni, trd), PARAMS)
    assert W.shape == (T, N)
    assert np.all(W[:100] == 0)
    for t in range(100, 105):
        assert W[t, 0] == pytest.approx(0.05)
        assert W[t, 1] == pytest.approx(-0.05)
        assert W[t, 2] == pytest.approx(0.0)  # net 0 -> no hedge
    assert np.all(W[105:] == 0)
    W7 = target_weights(_view(ret_co, ret_oc, ea, uni, trd), {"hold_days": 7, "z_threshold": 2.0})
    assert W7[106, 0] == pytest.approx(0.05) and np.all(W7[107:] == 0)
    Whi = target_weights(_view(ret_co, ret_oc, ea, uni, trd), {"hold_days": 5, "z_threshold": 50.0})
    assert np.all(Whi == 0)
    # missing input fails loudly
    with pytest.raises(KeyError):
        v = _view(ret_co, ret_oc, ea, uni, trd)
        v.extras.pop("ea_reaction_day")
        target_weights(v, PARAMS)


def test_point_in_time_core():
    rng = np.random.default_rng(3)
    T, N = 200, 30
    ret_co = rng.normal(0, 0.005, (T, N))
    ret_oc = rng.normal(0, 0.015, (T, N))
    ea = (rng.random((T, N)) < 0.05).astype(float)
    ret_oc[ea == 1] *= 6
    uni = np.ones((T, N), dtype=bool)
    trd = np.ones((T, N), dtype=bool)

    def run(rc, ro, e):
        a = abnormal_returns(rc, ro, trd[: len(rc)], uni[: len(rc)])
        z = reaction_z(a, pre_event_sigma(a))
        lo, sh = event_sides(e, uni[: len(rc)], trd[: len(rc)], z, 2.0)
        return book_weights(lo, sh, uni[: len(rc)], trd[: len(rc)], 5)

    full = run(ret_co, ret_oc, ea)
    assert np.abs(full).sum() > 0
    for cut in (80, 120, 150):
        part = run(ret_co[: cut + 1], ret_oc[: cut + 1], ea[: cut + 1])
        assert np.allclose(part, full[: cut + 1])
