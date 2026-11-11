import math

import numpy as np
import pytest

from strategy import PARAMS, target_weights
from strategy import regression_features, volume_shock, legs, leg_sizes, book, decision_rows, daily_returns


def _factor_data(T, N, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 0.01, size=(T, 5))
    B = rng.normal(0.8, 0.4, size=(N, 5))
    alpha = 0.0005
    eps = rng.normal(0.0, 0.002, size=(T, N))
    Y = alpha + X @ B.T + eps
    return X, B, Y, eps


def test_params_match_card():
    assert PARAMS == {"beta_days": 252, "top": 0.2}


def test_regression_recovers_betas_and_features():
    T, N, L = 300, 4, 126
    X, B, Y, eps = _factor_data(T, N)
    vs = np.ones((T, N), dtype=bool)
    ve = np.ones((T, 5), dtype=bool)
    dec = np.zeros(T, dtype=bool)
    dec[[50, 200, 299]] = True
    out = regression_features(Y, vs, X, ve, dec, L)
    # row 50: window not full yet -> not scored
    assert np.all(np.isnan(out["resid5"][50]))
    t = 299
    # exact OLS by hand for stock 0
    Xa = np.column_stack([np.ones(L), X[t - L + 1:t + 1]])
    coef = np.linalg.lstsq(Xa, Y[t - L + 1:t + 1, 0], rcond=None)[0]
    assert out["beta_spy"][t, 0] == pytest.approx(coef[1], abs=1e-9)
    assert abs(out["beta_spy"][t, 0] - B[0, 0]) < 0.1
    f5 = X[t - 4:t + 1].sum(axis=0)
    f21 = X[t - 20:t + 1].sum(axis=0)
    assert out["xasset5"][t, 0] == pytest.approx(coef[2:] @ f5[1:], abs=1e-9)
    assert out["resid5"][t, 0] == pytest.approx(Y[t - 4:t + 1, 0].sum() - coef[1:] @ f5, abs=1e-9)
    assert out["resid21"][t, 0] == pytest.approx(Y[t - 20:t + 1, 0].sum() - coef[1:] @ f21, abs=1e-9)
    res = Y[t - L + 1:t + 1, 0] - Xa @ coef
    assert out["ivol"][t, 0] == pytest.approx(np.std(res[-63:], ddof=1), abs=1e-9)
    # non-decision rows stay NaN
    assert np.all(np.isnan(out["resid5"][100]))


def test_coverage_rule_80_percent():
    T, N, L = 300, 3, 126
    X, B, Y, eps = _factor_data(T, N, seed=1)
    vs = np.ones((T, N), dtype=bool)
    ve = np.ones((T, 5), dtype=bool)
    t = 299
    s = t - L + 1
    need = int(math.ceil(0.8 * L))
    vs[s:s + (L - need), 1] = False          # exactly 80 % valid -> scored
    vs[s:s + (L - need) + 1, 2] = False      # one less -> not scored
    dec = np.zeros(T, dtype=bool)
    dec[t] = True
    out = regression_features(Y, vs, X, ve, dec, L)
    assert np.isfinite(out["resid5"][t, 1])
    assert np.isnan(out["resid5"][t, 2])
    # a factor day missing removes the row for every stock
    ve2 = ve.copy()
    ve2[s:s + (L - need) + 1, 3] = False
    out2 = regression_features(Y, vs, X, ve2, dec, L)
    assert np.all(np.isnan(out2["resid5"][t]))


def test_beta_days_changes_features():
    T, N = 400, 3
    X, B, Y, eps = _factor_data(T, N, seed=2)
    Y[:200] += np.random.default_rng(5).normal(0, 0.01, size=(200, N)) + X[:200, :1] * 1.5
    vs = np.ones((T, N), dtype=bool)
    ve = np.ones((T, 5), dtype=bool)
    dec = np.zeros(T, dtype=bool)
    dec[399] = True
    a = regression_features(Y, vs, X, ve, dec, 126)["beta_spy"][399]
    b = regression_features(Y, vs, X, ve, dec, 378)["beta_spy"][399]
    assert not np.allclose(a, b)


def test_point_in_time_regression():
    T, N, L = 300, 3, 126
    X, B, Y, eps = _factor_data(T, N, seed=3)
    vs = np.ones((T, N), dtype=bool)
    ve = np.ones((T, 5), dtype=bool)
    dec = np.zeros(T, dtype=bool)
    dec[[200, 250]] = True
    full = regression_features(Y, vs, X, ve, dec, L)
    Y2 = Y.copy()
    Y2[201:] = 9.0
    X2 = X.copy()
    X2[201:] = -3.0
    cut = regression_features(Y2, vs, X2, ve, dec, L)
    for k in full:
        np.testing.assert_allclose(full[k][:201], cut[k][:201])


def test_volume_shock():
    T, N = 80, 3
    dv = np.full((T, N), 1.0e6)
    dv[-5:, 1] = 2.0e6
    dv[-5:, 2] = np.nan
    dec = np.zeros(T, dtype=bool)
    dec[[60, T - 1]] = True
    v = volume_shock(dv, dec)
    assert np.isnan(v[60]).all()               # fewer than 68 rows of history
    assert v[T - 1, 0] == pytest.approx(0.0)
    assert v[T - 1, 1] == pytest.approx(math.log(2.0))
    assert np.isnan(v[T - 1, 2])               # no volume in the recent window


def test_legs_and_min_names():
    N = 30
    score = np.arange(N, dtype=float)
    member = np.ones(N, dtype=bool)
    beta = np.ones(N)
    lo, sh = legs(score, member, beta, 0.2)
    assert sorted(lo) == [24, 25, 26, 27, 28, 29]
    assert sorted(sh) == [0, 1, 2, 3, 4, 5]
    lo1, sh1 = legs(score, member, beta, 0.1)
    assert len(lo1) == 3 and len(sh1) == 3
    member2 = member.copy()
    member2[:11] = False                       # 19 scored members -> flat
    lo2, sh2 = legs(score, member2, beta, 0.2)
    assert lo2.size == 0 and sh2.size == 0
    score3 = score.copy()
    score3[29] = np.nan                        # unscored names are skipped
    lo3, _ = legs(score3, member, beta, 0.2)
    assert 29 not in lo3


def test_leg_sizes():
    assert leg_sizes(1.0, 1.0) == pytest.approx((0.5, 0.5))
    a, s = leg_sizes(1.5, 0.9)                 # a = 0.9 / 2.4 = 0.375
    assert a == pytest.approx(0.375) and s == pytest.approx(0.625)
    assert leg_sizes(3.0, 0.5)[0] == pytest.approx(0.3)     # clipped
    assert leg_sizes(0.5, 3.0)[0] == pytest.approx(0.7)
    assert leg_sizes(0.05, 1.0) == (0.5, 0.5)               # leg beta under 0.1
    assert leg_sizes(1.0, 0.09) == (0.5, 0.5)


def test_book_weights_exits_and_gross():
    T, N = 12, 30
    score = np.full((T, N), np.nan)
    beta = np.full((T, N), np.nan)
    dec = np.zeros(T, dtype=bool)
    dec[[0, 5, 10]] = True
    uni = np.ones((T, N), dtype=bool)
    score[5] = np.arange(N, dtype=float)
    beta[5] = 1.0
    beta[5, 24:] = 1.5                         # long leg mean beta 1.5, short 1.0 -> a = 0.4
    score[10] = np.arange(N, dtype=float)[::-1]
    beta[10] = 1.0
    uni[7, 27] = False                         # long name leaves mid-week
    W = book(score, beta, dec, uni, 0.2)
    assert np.all(W[0] == 0.0)                 # no model yet: flat
    assert np.all(np.isnan(W[1:5]))
    assert W[5, 25] == pytest.approx(0.4 / 6)
    assert W[5, 2] == pytest.approx(-0.6 / 6)
    assert np.nansum(np.abs(W[5])) == pytest.approx(1.0)
    assert W[7, 27] == 0.0
    assert np.isnan(W[7, 26]) and np.isnan(W[6, 27])
    assert np.isnan(W[8, 27])
    assert W[10, 0] == pytest.approx(0.5 / 6) and W[10, 29] == pytest.approx(-0.5 / 6)
    for t in range(T):
        assert np.nansum(np.abs(W[t])) <= 1.0 + 1e-12


def test_decision_rows_iso_weeks():
    # day 10957 = 2000-01-01 (Saturday); build business days and drop one Monday (holiday)
    days = np.arange(10957, 10957 + 28)
    dates = days.astype("datetime64[D]")
    wd = (days - 4) % 7                        # 0 = Monday (epoch day 0 was a Thursday)
    keep = wd < 5
    keep[np.flatnonzero(wd == 0)[1]] = False   # second Monday is a holiday
    dates = dates[keep]
    wdk = wd[keep]
    dec = decision_rows(dates)
    first = np.flatnonzero(dec)
    assert len(first) == 4
    assert wdk[first[0]] == 0 and wdk[first[1]] == 1 and wdk[first[2]] == 0


def test_daily_returns():
    class D:
        pass
    d = D()
    d.ret_co = np.array([[0.01, 0.0]])
    d.ret_oc = np.array([[0.02, 0.0]])
    d.listed = np.array([[True, False]])
    d.close = np.array([[10.0, np.nan]])
    r, v = daily_returns(d)
    assert r[0, 0] == pytest.approx(1.01 * 1.02 - 1.0)
    assert v.tolist() == [[True, False]]
