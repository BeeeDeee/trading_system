import numpy as np
import pytest

from strategy import PARAMS, target_weights
from strategy import (
    beta_ivol, cohort_weights, date_features, gbm_params, liq_63, market_return, overlap_book,
    release_feature, BETA_LOOKBACK, BETA_MIN_VALID,
)


def _days(start_ord, n):
    # consecutive calendar days from an integer day offset (no date literals)
    return (np.arange(n) + start_ord).astype("datetime64[D]")


def test_params_match_card():
    assert PARAMS == {"n_estimators": 100, "top": 0.2}


def test_date_features_hand_computed():
    # 1970-01-01 is a Thursday (day 0). January 1970: 22 weekdays; last weekday Fri Jan 30.
    d = _days(0, 33)                      # Jan 1 .. Feb 2 1970
    left, wd_in = date_features(d)
    assert wd_in[0] == 1 and left[0] == 21           # Thu Jan 1
    assert wd_in[1] == 2 and left[1] == 20           # Fri Jan 2
    assert wd_in[2] == 2 and left[2] == 20           # Sat Jan 3 (weekend: counts unchanged)
    assert wd_in[29] == 22 and left[29] == 0         # Fri Jan 30 = last weekday
    assert left[30] == 0 and wd_in[30] == 22         # Sat Jan 31
    assert wd_in[31] == 0 and left[31] == 20         # Sun Feb 1 (Feb 1970: 20 weekdays)
    assert wd_in[32] == 1 and left[32] == 19         # Mon Feb 2


def test_date_features_prefix_invariant():
    d = _days(10000, 400)
    a = date_features(d)
    b = date_features(d[:123])
    assert np.array_equal(a[0][:123], b[0]) and np.array_equal(a[1][:123], b[1])


def test_release_feature_clip_and_missing():
    s = np.array([np.nan, 1.0, 9.0, 10.0, 25.0])
    assert np.array_equal(release_feature(s), [10.0, 1.0, 9.0, 10.0, 10.0])


def test_market_return_members_only():
    r = np.array([[0.01, 0.03, 0.50], [np.nan, 0.02, 0.0]])
    uni = np.array([[True, True, False], [True, True, True]])
    m = market_return(r, uni)
    assert np.allclose(m, [0.02, 0.01])


def test_beta_ivol_exact_and_min_days():
    rng = np.random.default_rng(1)
    T = 300
    m = rng.normal(0, 0.01, T)
    eps = rng.normal(0, 0.005, T)
    y = 0.0002 + 1.5 * m + eps
    r = np.column_stack([y, y.copy()])
    r[: T - 200, 1] = np.nan           # second stock: valid from row 100 -> beta from row 288 on
    b, iv = beta_ivol(r, m)
    t = T - 1
    xs, ys = m[t - 251:t + 1], y[t - 251:t + 1]
    bb, aa = np.polyfit(xs, ys, 1)
    assert abs(b[t, 0] - bb) < 1e-4
    res = y[t - 62:t + 1] - aa - bb * m[t - 62:t + 1]
    assert abs(iv[t, 0] - np.sqrt(np.mean(res ** 2))) < 1e-5
    assert np.isnan(b[BETA_MIN_VALID - 2, 0]) and np.isfinite(b[BETA_MIN_VALID - 1, 0])
    valid_count = np.cumsum(np.isfinite(r[:, 1]))
    first = int(np.argmax(valid_count >= BETA_MIN_VALID))
    assert np.isnan(b[first - 1, 1]) and np.isfinite(b[first, 1])
    assert BETA_LOOKBACK == 252


def test_beta_point_in_time():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, (400, 3))
    m = r.mean(axis=1)
    b1, i1 = beta_ivol(r, m)
    r2 = r.copy()
    r2[300:] = rng.normal(0, 0.05, (100, 3))
    b2, i2 = beta_ivol(r2, m)
    assert np.allclose(b1[:300], b2[:300], equal_nan=True)
    assert np.allclose(i1[:300], i2[:300], equal_nan=True)


def test_liq_median_and_missing():
    T = 70
    dv = np.full((T, 2), 4.0)
    dv[:, 1] = np.arange(T) + 1.0
    dv[5, 1] = np.nan
    uni = np.ones((T, 2), dtype=bool)
    uni[10, 0] = False
    liq = liq_63(dv, uni)
    assert np.isclose(liq[0, 0], np.log(4.0))
    assert np.isnan(liq[10, 0])                       # non-member cell not computed
    assert np.isclose(liq[T - 1, 1], np.log(np.median(np.arange(T - 62, T + 1))))
    dz = np.zeros((T, 1))
    assert np.isnan(liq_63(dz, np.ones((T, 1), dtype=bool))[-1, 0])   # log(0) -> unknown


def test_overlap_book_cohorts_and_exit():
    T, N = 8, 3
    uni = np.ones((T, N), dtype=bool)
    c = np.zeros((T, N))
    c[1, 0] = 1.0                      # cohort decided at row 1: all in stock 0
    c[2, 1] = 1.0
    uni[4, 1] = False                  # stock 1 leaves at row 4, returns at row 5
    w = overlap_book(c, uni, hold=5)
    assert np.allclose(w[:, 0], [0, .2, .2, .2, .2, .2, 0, 0])
    assert np.allclose(w[:, 1], [0, 0, .2, .2, 0, 0, 0, 0])   # closed at row 4, not re-opened
    assert np.all(w.sum(axis=1) <= 1 + 1e-12) and np.all(w >= 0)


def test_overlap_book_full_gross():
    T, N = 10, 4
    uni = np.ones((T, N), dtype=bool)
    c = np.full((T, N), 0.25)
    w = overlap_book(c, uni, hold=5)
    assert np.allclose(w.sum(axis=1), [0.2, 0.4, 0.6, 0.8, 1, 1, 1, 1, 1, 1])


def test_cohort_top_param_and_min_names():
    rng = np.random.default_rng(3)
    T, N = 4, 120
    score = rng.normal(size=(T, N))
    uni = np.ones((T, N), dtype=bool)
    uni[3, 60:] = False                    # 60 members on row 3: still >= 50
    score[2, 40:] = np.nan                 # only 40 scored on row 2 -> cash
    c2 = cohort_weights(score, uni, 0.2)
    c1 = cohort_weights(score, uni, 0.1)
    assert np.isclose(c2[0].sum(), 1.0) and np.isclose(c1[0].sum(), 1.0)
    assert (c2[0] > 0).sum() > (c1[0] > 0).sum()
    assert c2[2].sum() == 0.0
    assert np.all(c2[3, 60:] == 0) and np.isclose(c2[3].sum(), 1.0)
    top_idx = np.argsort(-score[0])[: (c1[0] > 0).sum()]
    assert np.all(c1[0, top_idx] > 0)


def test_gbm_params_reads_n_estimators():
    p = gbm_params({"n_estimators": 50, "top": 0.2})
    assert p["n_estimators"] == 50 and p["num_leaves"] == 15 and p["min_child_samples"] == 2000


def test_target_weights_is_callable():
    assert callable(target_weights)
