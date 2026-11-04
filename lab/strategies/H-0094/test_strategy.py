import math

import numpy as np
import pytest

from strategy import (PARAMS, cohort_book, daily_returns, membership_run, raw_features, residual_returns,
                      target_weights, trailing_std, trailing_sum)

NAN = np.nan


def test_params_match_card():
    assert PARAMS == {"leg_fraction": 0.1, "n_estimators": 100}


def test_trailing_sum_full_window_and_nan():
    x = np.array([[1.0], [2.0], [3.0], [NAN], [5.0], [6.0]])
    s = trailing_sum(x, 3)
    assert np.isnan(s[0, 0]) and np.isnan(s[1, 0])
    assert s[2, 0] == 6.0
    assert np.isnan(s[3, 0]) and np.isnan(s[4, 0]) and np.isnan(s[5, 0])   # window contains the NaN
    # window longer than the data: all NaN, no crash (G0 cuts can be short)
    assert np.isnan(trailing_sum(x[:2], 5)).all()


def test_trailing_std_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 2))
    s = trailing_std(x, 10)
    assert np.isnan(s[:9]).all()
    for t in range(9, 30):
        assert np.allclose(s[t], x[t - 9:t + 1].std(axis=0, ddof=1))
    assert np.isnan(trailing_std(x[:4], 10)).all()


def test_daily_returns_and_residual():
    ret_co = np.array([[0.01, 0.0], [0.0, 0.02]])
    ret_oc = np.array([[0.02, 0.0], [0.01, -0.01]])
    close = np.array([[10.0, NAN], [11.0, 20.0]])
    listed = np.array([[True, False], [True, True]])
    r = daily_returns(ret_co, ret_oc, close, listed)
    assert math.isclose(r[0, 0], 1.01 * 1.02 - 1)
    assert np.isnan(r[0, 1])
    assert math.isclose(r[1, 1], 1.02 * 0.99 - 1)
    uni = np.array([[True, True], [True, False]])
    e = residual_returns(r, uni)
    assert math.isclose(e[0, 0], 0.0, abs_tol=1e-15)          # only member with a return is itself
    assert math.isclose(e[1, 1], r[1, 1] - r[1, 0])           # non-member: minus the members' mean


def _panel(T=80, N=3, seed=1):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.02, size=(T, N))
    uni = np.ones((T, N), dtype=bool)
    flag = np.zeros((T, N))
    dv = rng.uniform(1e6, 2e6, size=(T, N))
    return r, uni, flag, dv


def test_features_by_hand():
    r, uni, flag, dv = _panel()
    flag[70, 0] = 1.0
    flag[72, 0] = 1.0
    flag[:10, 2] = NAN                                         # before the first filing: counts as 0
    z1, z5, ea5, ea_z, dv_ab, lvol = raw_features(r, uni, flag, dv)
    T = r.shape[0]
    e = r - r.mean(axis=1, keepdims=True)
    t = 75
    vol = r[t - 62:t + 1, 0].std(ddof=1)
    assert math.isclose(z1[t, 0], e[t, 0] / vol, rel_tol=1e-9)
    assert math.isclose(z5[t, 0], e[t - 4:t + 1, 0].sum() / (vol * math.sqrt(5)), rel_tol=1e-9)
    assert math.isclose(lvol[t, 0], math.log(vol), rel_tol=1e-9)
    want = math.log(dv[t - 4:t + 1, 0].mean() / dv[t - 67:t - 4, 0].mean())
    assert math.isclose(dv_ab[t, 0], want, rel_tol=1e-9)
    # release window t-4..t
    assert ea5[69, 0] == 0 and ea5[70, 0] == 1 and ea5[74, 0] == 1 and ea5[76, 0] == 1 and ea5[77, 0] == 0
    assert ea_z[71, 0] == z1[70, 0]                            # only release in window
    assert ea_z[74, 0] == z1[72, 0]                            # most recent release wins
    assert ea_z[77, 0] == 0.0 and ea_z[75, 1] == 0.0
    assert (ea5[:, 2] == 0).all()
    # history: z1/lvol need 63 sessions, dv_ab needs 68
    assert np.isnan(z1[61]).all() and np.isfinite(z1[62]).all()
    assert np.isnan(dv_ab[66]).all() and np.isfinite(dv_ab[67]).all()
    assert z1.shape == (T, 3)


def test_features_point_in_time():
    r, uni, flag, dv = _panel(T=90)
    flag[80, 1] = 1.0
    full = raw_features(r, uni, flag, dv)
    cut = 78
    r2, flag2, dv2 = r.copy(), flag.copy(), dv.copy()
    rng = np.random.default_rng(5)
    r2[cut + 1:] = rng.normal(0, 0.05, size=r2[cut + 1:].shape)
    flag2[cut + 1:] = 1.0
    dv2[cut + 1:] = 1.0
    pert = raw_features(r2, uni, flag2, dv2)
    for a, b in zip(full, pert):
        assert np.array_equal(np.isnan(a[:cut + 1]), np.isnan(b[:cut + 1]))
        assert np.allclose(np.nan_to_num(a[:cut + 1]), np.nan_to_num(b[:cut + 1]))


def test_membership_run():
    uni = np.array([[1, 0], [1, 1], [0, 1], [1, 1]], dtype=bool)
    assert membership_run(uni).tolist() == [[1, 0], [2, 1], [0, 2], [1, 3]]


def test_cohort_book_sum_hold_and_universe_exit():
    T, N = 10, 2
    c = np.zeros((T, N))
    c[2] = [0.5, -0.5]                                         # a single cohort on row 2
    uni = np.ones((T, N), dtype=bool)
    b = cohort_book(c, uni)
    assert np.allclose(b[:2], 0)
    assert np.allclose(b[2:7], [[0.1, -0.1]] * 5)              # held 5 sessions at 1/5
    assert np.allclose(b[7:], 0)                               # then dropped
    # stock 1 leaves the universe on row 4 and comes back on row 5: closed from row 4, not reopened
    uni2 = uni.copy()
    uni2[4, 1] = False
    b2 = cohort_book(c, uni2)
    assert np.allclose(b2[2:4, 1], -0.1) and np.allclose(b2[4:, 1], 0)
    assert np.allclose(b2[2:7, 0], 0.1)
    # five full cohorts: gross 1, dollar neutral; NaN cohort rows count as no new cohort
    c3 = np.tile([0.5, -0.5], (T, 1))
    c3[5] = NAN
    b3 = cohort_book(c3, uni)
    assert np.allclose(np.abs(b3[4]).sum(), 1.0) and np.allclose(b3[4].sum(), 0)
    assert np.allclose(np.abs(b3[6]).sum(), 0.8)
    assert (np.abs(b3).sum(axis=1) <= 1 + 1e-12).all()


def _synthetic_view():
    from lab.framework.data import synthetic
    return synthetic()


def test_end_to_end_gross_universe_and_flat_start():
    data = _synthetic_view()
    if data.universe is None:
        pytest.skip("synthetic market without a universe")
    w = target_weights(data, PARAMS)
    T, N = len(data.dates), len(data.instruments)
    assert w.shape == (T, N)
    assert np.isfinite(w).all()
    assert (np.abs(w).sum(axis=1) <= 1 + 1e-9).all()
    assert (w[~np.asarray(data.universe, dtype=bool)] == 0).all()
    assert np.abs(w[:68]).sum() == 0                           # no score before the history and first fit
