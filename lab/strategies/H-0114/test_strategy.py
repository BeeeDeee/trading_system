import numpy as np
import pytest

from strategy import PARAMS, target_weights
from strategy import (close_leavers, daily_returns, decision_book, oil_changes, oil_shock, raw_features,
                      roll_sum, shift_down, stock_regression)


def test_params_match_card():
    assert PARAMS == {"n_estimators": 100, "top": 0.1}
    assert callable(target_weights)


def test_roll_sum_and_shift():
    x = np.arange(1.0, 7.0)
    np.testing.assert_allclose(roll_sum(x, 3), [1, 3, 6, 9, 12, 15])
    np.testing.assert_allclose(roll_sum(x, 10), np.cumsum(x))      # shorter than the window
    s = shift_down(x, 1)
    assert np.isnan(s[0]) and s[1] == 1.0 and s[-1] == 5.0
    b = shift_down(np.array([True, False, True]), 1)
    assert b.tolist() == [False, True, False]


def test_daily_returns_needs_price_today_and_yesterday():
    co = np.array([[0.0], [0.01], [0.02], [0.0]])
    oc = np.array([[0.0], [0.02], [-0.01], [0.0]])
    close = np.array([[10.0], [11.0], [np.nan], [12.0]])
    r = daily_returns(co, oc, close)
    assert np.isnan(r[0, 0])                       # no previous close
    assert r[1, 0] == pytest.approx(1.01 * 1.02 - 1)
    assert np.isnan(r[2, 0]) and np.isnan(r[3, 0])  # no price on 2, so 2 and 3 invalid


def test_oil_changes_missing_nonpositive_and_clip():
    S = np.array([np.nan, 50.0, 55.0, -5.0, np.nan, 44.0, 44.0, 100.0])
    O = oil_changes(S)
    assert np.isnan(O[0]) and np.isnan(O[1])                    # no earlier valid value
    assert O[2] == pytest.approx(np.log(55 / 50))
    assert np.isnan(O[3]) and np.isnan(O[4])                    # non-positive and missing are missing
    assert O[5] == pytest.approx(-0.20)                          # log(44/55) = -0.223, clipped
    assert O[6] == 0.0
    assert O[7] == pytest.approx(0.20)                           # log(100/44) clipped


def _ols(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    s2 = resid @ resid / (len(y) - X1.shape[1])
    cov = s2 * np.linalg.inv(X1.T @ X1)
    return beta, beta / np.sqrt(np.diag(cov))


def test_stock_regression_matches_lstsq_and_min_pairs():
    rng = np.random.default_rng(1)
    T = 320
    M1 = rng.normal(0, 0.01, T)
    O = rng.normal(0, 0.02, T)
    O[[5, 100, 290]] = np.nan
    M1[200] = np.nan
    R1 = np.empty((T, 3))
    R1[:, 0] = 0.0005 + 1.1 * M1 + 0.4 * O + rng.normal(0, 0.01, T)
    R1[:, 2] = -0.3 * O + 0.8 * M1 + rng.normal(0, 0.02, T)
    R1[:, 1] = -0.3 * O + rng.normal(0, 0.02, T)
    R1[250, 0] = np.nan
    R1[:150, 1] = np.nan                                         # late listing: too few pairs for a while
    b_m, b_o, tst = stock_regression(R1, M1, O)
    t = T - 1
    for j in (0, 2):
        rows = np.arange(t - 251, t + 1)
        ok = np.isfinite(R1[rows, j]) & np.isfinite(M1[rows]) & np.isfinite(O[rows])
        beta, tt = _ols(R1[rows, j][ok], np.column_stack([M1[rows][ok], O[rows][ok]]))
        assert b_m[t, j] == pytest.approx(beta[1], rel=1e-7)
        assert b_o[t, j] == pytest.approx(beta[2], rel=1e-7)
        assert tst[t, j] == pytest.approx(tt[2], rel=1e-6)
    # stock 1 has valid pairs only from row 150: fewer than 200 before row ~350 -> NaN everywhere here
    assert np.all(np.isnan(b_o[:, 1]))
    # stock 0 needs 200 pairs: row 198 has at most 199
    assert np.isnan(b_o[198, 0]) and np.isfinite(b_o[220, 0])


def test_oil_shock_by_hand():
    rng = np.random.default_rng(2)
    T = 300
    M1 = rng.normal(0, 0.01, T)
    O = 0.3 * M1 + rng.normal(0, 0.02, T)
    O[[270, 285, 295]] = np.nan
    c, s_u, U10, U10lag, S10 = oil_shock(O, M1)
    t = T - 1
    rows = np.arange(t - 251, t + 1)
    ok = np.isfinite(O[rows])
    o, m = O[rows][ok], M1[rows][ok]
    cc = np.cov(o, m)[0, 1] / np.var(m, ddof=1)
    assert c[t] == pytest.approx(cc, rel=1e-8)
    u = o - cc * m
    assert s_u[t] == pytest.approx(np.std(u, ddof=1), rel=1e-8)
    last = np.arange(t - 9, t + 1)
    okl = np.isfinite(O[last])
    u10 = np.sum(O[last][okl] - cc * M1[last][okl])
    assert U10[t] == pytest.approx(u10, rel=1e-8)
    lag = np.arange(t - 19, t - 9)
    okg = np.isfinite(O[lag])
    assert U10lag[t] == pytest.approx(np.sum(O[lag][okg] - cc * M1[lag][okg]), rel=1e-8)
    assert S10[t] == pytest.approx(u10 / (s_u[t] * np.sqrt(10)), rel=1e-8)
    # fewer than 8 valid u values in the last 10 -> missing
    O2 = O.copy()
    O2[t - 2:t + 1] = np.nan
    _, _, U10b, _, S10b = oil_shock(O2, M1)
    assert np.isnan(U10b[t]) and np.isnan(S10b[t])
    # early rows: fewer than 200 sessions -> no shock
    assert np.all(np.isnan(S10[:199]))


def _market(T=330, N=4, seed=3):
    rng = np.random.default_rng(seed)
    co = rng.normal(0, 0.004, (T, N))
    oc = rng.normal(0, 0.01, (T, N))
    close = 50 * np.cumprod(1 + co + oc, axis=0)
    close[:40, 3] = np.nan                                        # late listing
    co[:40, 3] = 0.0
    oc[:40, 3] = 0.0
    uni = np.ones((T, N), dtype=bool)
    uni[:40, 3] = False
    S = 60 * np.exp(np.cumsum(rng.normal(0, 0.02, T)))
    S[:3] = np.nan
    return co, oc, close, uni, S


def test_raw_features_by_hand():
    co, oc, close, uni, S = _market()
    raw, S10 = raw_features(co, oc, close, uni, S)
    r = daily_returns(co, oc, close)
    ok = uni & np.isfinite(r)
    m = np.where(ok.any(axis=1), np.where(ok, r, 0.0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1), np.nan)
    O = oil_changes(S)
    M1 = shift_down(m, 1)
    b_m, b_o, tst = stock_regression(shift_down(r, 1), M1, O)
    _, _, U10, U10lag, S10x = oil_shock(O, M1)
    t = 320
    j = 1
    np.testing.assert_allclose(S10[t], S10x[t])
    assert raw["b_o"][t, j] == pytest.approx(b_o[t, j], rel=1e-5)
    assert raw["P10"][t, j] == pytest.approx(b_o[t, j] * U10[t], rel=1e-5)
    assert raw["P10lag"][t, j] == pytest.approx(b_o[t, j] * U10lag[t], rel=1e-5)
    res = np.sum(r[t - 4:t + 1, j] - b_m[t, j] * m[t - 4:t + 1])
    assert raw["resid5"][t, j] == pytest.approx(res, rel=1e-4, abs=1e-7)
    assert raw["vol63"][t, j] == pytest.approx(np.std(r[t - 62:t + 1, j], ddof=1), rel=1e-5)
    # late listing: no b_o until 200 pairs after its first return
    assert np.all(np.isnan(raw["b_o"][:240, 3]))
    assert np.all(np.isnan(raw["vol63"][:103, 3])) and np.isfinite(raw["vol63"][104, 3])


def test_raw_features_point_in_time():
    co, oc, close, uni, S = _market()
    raw, S10 = raw_features(co, oc, close, uni, S)
    rng = np.random.default_rng(9)
    for cut in (215, 260, 301):
        co2, oc2, close2, S2 = co.copy(), oc.copy(), close.copy(), S.copy()
        co2[cut + 1:] = rng.normal(0, 0.05, co2[cut + 1:].shape)
        close2[cut + 1:] = 99.0
        S2[cut + 1:] = -3.0
        raw2, S102 = raw_features(co2, oc2, close2, uni, S2)
        np.testing.assert_array_equal(S10[:cut + 1], S102[:cut + 1])
        for k in raw:
            np.testing.assert_array_equal(raw[k][:cut + 1], raw2[k][:cut + 1])
        rawt, S10t = raw_features(co[:cut + 1], oc[:cut + 1], close[:cut + 1], uni[:cut + 1], S[:cut + 1])
        np.testing.assert_array_equal(S10[:cut + 1], S10t)
        for k in raw:
            np.testing.assert_array_equal(raw[k][:cut + 1], rawt[k])


def _book_inputs(N=60, T=6):
    rng = np.random.default_rng(4)
    score = np.tile(np.arange(N, dtype=float), (T, 1)) + rng.normal(0, 0.01, (T, N))
    uni = np.ones((T, N), dtype=bool)
    weekly = np.array([True, False, True, False, True, True])
    S10 = np.array([1.5, 2.0, 0.5, 3.0, -1.2, np.nan])
    return score, S10, uni, weekly


def test_decision_book_active_inactive_and_legs():
    score, S10, uni, weekly = _book_inputs()
    w = decision_book(score, S10, uni, weekly, top=0.1)
    assert np.all(np.isnan(w[1])) and np.all(np.isnan(w[3]))    # not a decision row: hold
    assert np.all(w[2] == 0.0)                                   # |S10| < 1: flat
    assert np.all(w[5] == 0.0)                                   # S10 missing: flat
    for t in (0, 4):                                              # active, both signs of the shock
        row = w[t]
        assert row[row > 0].sum() == pytest.approx(0.5)
        assert row[row < 0].sum() == pytest.approx(-0.5)
        assert np.abs(row).sum() <= 1.0 + 1e-9
        k = int((row > 0).sum())
        assert 5 <= k <= 7 and int((row < 0).sum()) == k
        order = np.argsort(score[t])
        assert set(np.flatnonzero(row > 0)) == set(order[-k:])
        assert set(np.flatnonzero(row < 0)) == set(order[:k])
        assert np.allclose(row[row > 0], 0.5 / k)
    w2 = decision_book(score, S10, uni, weekly, top=0.2)
    assert int((w2[0] > 0).sum()) > int((w[0] > 0).sum())       # the `top` parameter matters


def test_decision_book_needs_50_scored_members_and_membership():
    score, S10, uni, weekly = _book_inputs()
    score[0, 11:] = np.nan                                        # 11 scored on row 0
    uni = uni.copy()
    uni[4, :15] = False                                           # 45 members on row 4
    w = decision_book(score, S10, uni, weekly, top=0.1)
    assert np.all(w[0] == 0.0) and np.all(w[4] == 0.0)
    score2, S102, uni2, weekly2 = _book_inputs()
    uni2[0, 59] = False                                          # best score but not a member
    w3 = decision_book(score2, S102, uni2, weekly2, top=0.1)
    assert w3[0, 59] == 0.0 and w3[0, 58] > 0


def test_close_leavers():
    nan = np.nan
    w = np.array([[0.5, -0.5, 0.0],
                  [nan, nan, nan],
                  [nan, nan, nan],
                  [nan, nan, nan],
                  [0.0, 0.0, 0.0]])
    uni = np.ones((5, 3), dtype=bool)
    uni[2, 1] = False
    uni[3, 1] = False
    uni[3, 2] = False                                             # not held: untouched
    out = close_leavers(w, uni)
    assert np.isnan(out[1]).all()
    assert out[2, 1] == 0.0 and np.isnan(out[2, 0]) and np.isnan(out[2, 2])
    assert np.isnan(out[3]).all()                                 # already closed
    assert np.all(out[4] == 0.0)
