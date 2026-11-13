import math

import numpy as np
import pytest

from strategy import PARAMS, target_weights  # noqa: F401
from strategy import _book, _cohort, _decision_rows, _features, _gbm_params, _n_cohorts, _top_k


# ----------------------------------------------------------------------------------------------- helpers
def _weekdays(first_day, last_day, drop=()):
    d = np.arange(first_day, last_day + 1).astype("datetime64[D]")
    d = d[np.is_busday(d)]
    keep = ~np.isin(d.astype(np.int64), np.array(drop, dtype=np.int64))
    return d[keep]


# day numbers since 1970-01-01: 2021-05-03 = 18750, 2021-05-28 (Fri) = 18775, 2021-05-31 (Mon) = 18778,
# 2021-06-01 = 18779, 2021-06-30 (Wed) = 18808, 2021-07-30 (Fri) = 18838
D_MAY3, D_MAY28, D_MAY31, D_JUN1, D_JUN30, D_JUL30 = 18750, 18775, 18778, 18779, 18808, 18838


def _ref_features(r, valid, dv, t, j):
    """Direct slicing reference of the card's formulas at row t, column j."""
    rr = np.where(valid[:, j], r[:, j], np.nan)
    win = rr[t - 251: t + 1]
    F = rr[t - 251: t - 20]
    assert F.shape[0] == 231
    sd = np.nanstd(win, ddof=1)
    mom = np.nansum(F) / (sd * math.sqrt(231))
    path = (np.sum(F > 0) - np.sum(F < 0)) / 231
    f2 = np.where(np.isfinite(F), F, 0.0) ** 2
    jump = np.sort(f2)[-5:].sum() / f2.sum()
    sd63 = np.nanstd(rr[t - 62: t + 1], ddof=1)
    recent = np.nansum(rr[t - 20: t + 1]) / (sd63 * math.sqrt(21))
    volratio = sd63 / sd
    d = dv[:, j]
    att = math.log(np.nanmean(d[t - 20: t + 1]) / np.nanmean(d[t - 251: t + 1]))
    return [mom, path, jump, recent, volratio, att]


def _random_inputs(T=400, N=3, seed=1):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.01, (T, N))
    r[rng.random((T, N)) < 0.02] *= 6.0
    valid = rng.random((T, N)) > 0.03
    dv = np.exp(rng.normal(15.0, 0.5, (T, N)))
    dv[~valid] = np.nan
    return r, valid, dv


# ----------------------------------------------------------------------------------------------- features
def test_features_match_reference():
    r, valid, dv = _random_inputs()
    feats, elig = _features(np.where(valid, r, np.nan), valid, dv)
    for t in (251, 300, 399):
        for j in range(3):
            assert elig[t, j]
            ref = _ref_features(r, valid, dv, t, j)
            for f, x in zip(feats, ref):
                assert f[t, j] == pytest.approx(x, rel=1e-9, abs=1e-12)


def test_features_nan_before_full_window():
    r, valid, dv = _random_inputs()
    feats, elig = _features(np.where(valid, r, np.nan), valid, dv)
    assert not elig[:251].any()
    for f in feats:
        assert np.isnan(f[:251]).all()


def test_path_and_jump_by_hand():
    T = 260
    r = np.full((T, 1), 0.001)
    r[100:105, 0] = 0.01                      # five larger days inside F of t = 259 (rows 8..238)
    r[250:260, 0] = 0.002                     # after F: gives the 63-row window a non-zero std
    valid = np.ones((T, 1), dtype=bool)
    dv = np.ones((T, 1))
    feats, elig = _features(r, valid, dv)
    t = 259
    assert elig[t, 0]
    assert feats[1][t, 0] == pytest.approx(1.0)                         # all 231 days up
    exp_jump = 5 * 1e-4 / (5 * 1e-4 + 226 * 1e-6)
    assert feats[2][t, 0] == pytest.approx(exp_jump)
    assert feats[5][t, 0] == pytest.approx(0.0)                         # flat dollar volume -> log(1) = 0
    r2 = -r
    feats2, _ = _features(r2, valid, dv)
    assert feats2[1][t, 0] == pytest.approx(-1.0)                       # signed, not absolute


def test_eligibility_counts():
    T = 300
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, (T, 3))
    valid = np.ones((T, 3), dtype=bool)
    dv = np.exp(rng.normal(15, 0.3, (T, 3)))
    t = 299
    valid[t - 251: t - 251 + 22, 0] = False          # 230 valid returns in the last 252 rows -> scored
    valid[t - 251: t - 251 + 23, 1] = False          # 229 -> not scored
    dv[t - 251: t - 251 + 53, 2] = np.nan            # 199 valid dollar volumes -> not scored
    feats, elig = _features(np.where(valid, r, np.nan), valid, dv)
    assert elig[t, 0]
    assert not elig[t, 1]
    assert not elig[t, 2]
    assert np.isnan(feats[0][t, 1]) and np.isnan(feats[5][t, 2])


def test_features_point_in_time():
    r, valid, dv = _random_inputs(T=420)
    full, _ = _features(np.where(valid, r, np.nan), valid, dv)
    cut = 330
    trunc, _ = _features(np.where(valid, r, np.nan)[: cut + 1], valid[: cut + 1], dv[: cut + 1])
    rng = np.random.default_rng(9)
    r2 = r.copy()
    r2[cut + 1:] = rng.normal(0, 0.05, r2[cut + 1:].shape)
    pert, _ = _features(np.where(valid, r2, np.nan), valid, dv)
    for a, b, c in zip(full, trunc, pert):
        np.testing.assert_array_equal(a[: cut + 1], b)
        np.testing.assert_array_equal(a[: cut + 1], c[: cut + 1])


# ----------------------------------------------------------------------------------------------- calendar
def test_decision_rows_month_end_and_holiday_fallback():
    d = _weekdays(D_MAY3, D_JUL30)
    dec = _decision_rows(d)
    days = set(d[dec].astype(np.int64).tolist())
    assert days == {D_MAY31, D_JUN30, D_JUL30}

    d2 = _weekdays(D_MAY3, D_JUL30, drop=(D_MAY31,))    # Memorial Day on the last weekday of May
    dec2 = _decision_rows(d2)
    days2 = set(d2[dec2].astype(np.int64).tolist())
    assert days2 == {D_JUN1, D_JUN30, D_JUL30}           # May falls back to the first June row
    assert D_MAY28 not in days2


def test_decision_rows_point_in_time():
    d = _weekdays(D_MAY3, D_JUL30, drop=(D_MAY31,))
    full = _decision_rows(d)
    for cut in range(1, d.shape[0]):
        np.testing.assert_array_equal(_decision_rows(d[:cut]), full[:cut])


# ----------------------------------------------------------------------------------------------- book
def _sleeves(n_risk=33, n_def=10):
    risk = np.arange(n_risk)
    dfn = np.arange(n_risk, n_risk + n_def)
    return [(risk, 0.60, 0), (dfn, 0.40, n_risk + 1)], n_risk + n_def


def test_top_k():
    assert [_top_k(n) for n in (2, 5, 10, 11, 15, 16, 33)] == [2, 2, 2, 3, 3, 4, 7]


def test_cohort_top_fifth_per_sleeve():
    sleeves, N = _sleeves()
    s = np.full(N, np.nan)
    s[0:15] = np.arange(15, dtype=float)              # 15 scored RISK -> top 3 = cols 14, 13, 12
    s[33:43] = np.arange(10, dtype=float)[::-1]       # 10 scored DEFENSIVE -> top 2 = cols 33, 34
    w = _cohort(s, sleeves, N)
    assert w[[12, 13, 14]] == pytest.approx([0.2, 0.2, 0.2])
    assert w[[33, 34]] == pytest.approx([0.2, 0.2])
    assert np.count_nonzero(w) == 5
    assert w.sum() == pytest.approx(1.0)


def test_cohort_benchmark_fallbacks():
    sleeves, N = _sleeves()
    w = _cohort(np.full(N, np.nan), sleeves, N)       # before the first fit
    assert w[0] == pytest.approx(0.6) and w[34] == pytest.approx(0.4)
    assert np.count_nonzero(w) == 2
    s = np.full(N, np.nan)
    s[3:8] = [1.0, 5.0, 2.0, 4.0, 3.0]                # 5 RISK scored -> top 2 = cols 4, 6
    s[40] = 1.0                                       # 1 DEFENSIVE scored -> IEF stand-in (col 34)
    w = _cohort(s, sleeves, N)
    assert w[[4, 6]] == pytest.approx([0.3, 0.3])
    assert w[34] == pytest.approx(0.4) and w[40] == 0.0
    assert w.sum() == pytest.approx(1.0)


def test_book_averages_live_cohorts():
    sleeves, N = _sleeves()
    T = 6
    score = np.full((T, N), np.nan)
    score[1, 0:10] = np.arange(10)                   # cohort A: RISK cols 9, 8
    score[3, 0:10] = np.arange(10)[::-1]             # cohort B: RISK cols 0, 1
    score[5, 0:10] = np.arange(10)                   # cohort C: RISK cols 9, 8
    dec = np.array([False, True, False, True, False, True])
    W2 = _book(score, dec, sleeves, 2)
    assert np.isnan(W2[[0, 2, 4]]).all()
    assert W2[1, 9] == pytest.approx(0.3) and W2[1, 34] == pytest.approx(0.4)
    assert W2[3, 9] == pytest.approx(0.15) and W2[3, 0] == pytest.approx(0.15)
    assert W2[5, 9] == pytest.approx(0.15) and W2[5, 0] == pytest.approx(0.15)
    W1 = _book(score, dec, sleeves, 1)
    assert W1[3, 0] == pytest.approx(0.3) and W1[3, 9] == 0.0
    W3 = _book(score, dec, sleeves, 3)
    assert W3[5, 9] == pytest.approx(0.2) and W3[5, 0] == pytest.approx(0.1)
    for W in (W1, W2, W3):
        rows = W[dec]
        assert (rows >= 0).all()
        assert np.all(np.abs(rows).sum(axis=1) <= 1.0 + 1e-12)
        assert rows.sum(axis=1) == pytest.approx(np.ones(3))


def test_params_reach_the_book_and_model():
    assert PARAMS == {"horizon": 42, "n_estimators": 100}
    assert _n_cohorts({"horizon": 21}) == 1
    assert _n_cohorts({"horizon": 42}) == 2
    assert _n_cohorts({"horizon": 63}) == 3
    p = _gbm_params({"n_estimators": 200})
    assert p["n_estimators"] == 200 and p["num_leaves"] == 8 and p["min_child_samples"] == 200
    assert p["learning_rate"] == 0.05
