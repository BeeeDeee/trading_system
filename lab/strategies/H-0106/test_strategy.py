import numpy as np
import pytest

from strategy import PARAMS, target_weights
from strategy import (
    COHORTS, LEG_GROSS, _cohort_book, _gather, _lagged_median, _log_ratio, _model_params, _news_flag,
    _sector_assignment, _trailing_sum,
)
from strategy import _flow_sessions, _lagged_std, _new_entries
from lab.framework.ml import score_weights


def test_params_match_card():
    assert PARAMS == {"n_estimators": 100, "top": 0.1}
    assert callable(target_weights)


def test_model_params_read_n_estimators():
    p = _model_params({"n_estimators": 50, "top": 0.1})
    assert p == {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 500, "n_estimators": 50}
    assert _model_params({"n_estimators": 200, "top": 0.1})["n_estimators"] == 200


def test_trailing_sum_window_and_nan():
    x = np.arange(1.0, 9.0)[:, None]          # 1..8
    s = _trailing_sum(x, 5)
    assert np.all(np.isnan(s[:4, 0]))
    assert s[4, 0] == 15.0 and s[7, 0] == 4 + 5 + 6 + 7 + 8
    x2 = x.copy()
    x2[5, 0] = np.nan
    s2 = _trailing_sum(x2, 5)
    assert s2[4, 0] == 15.0
    assert np.all(np.isnan(s2[5:, 0]))       # rows 5..8 contain the gap within t-4..t
    assert np.all(np.isnan(_trailing_sum(x[:3], 5)))   # view shorter than the window


def test_lagged_median_excludes_today_and_nan():
    T = 70
    v = np.arange(1.0, T + 1)[:, None].repeat(2, axis=1)
    v[64, 1] = np.nan
    need = np.ones((T, 2), bool)
    need[66, 0] = False
    med = _lagged_median(v, need, window=63, chunk=4)
    assert np.all(np.isnan(med[:63]))
    # row 63: values of rows 0..62 = 1..63 -> median 32
    assert med[63, 0] == 32.0
    # row 65, column 1: rows 2..64 = 3..65 with row 64 (value 65) missing -> 3..64 (62 values) -> 33.5
    assert med[65, 1] == 33.5
    assert med[65, 0] == 34.0
    assert np.isnan(med[66, 0])               # not needed -> NaN
    # point-in-time: changing row t does not change the median at row t
    v2 = v.copy()
    v2[63:] = 1e9
    med2 = _lagged_median(v2, need, window=63, chunk=4)
    assert med2[63, 0] == med[63, 0]


def test_log_ratio_nan():
    out = _log_ratio(np.array([[np.e, np.nan, 1.0]]), np.array([[1.0, 1.0, np.nan]]))
    assert out[0, 0] == pytest.approx(1.0)
    assert np.isnan(out[0, 1]) and np.isnan(out[0, 2])


def _sector_market(T=300, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.01, size=(T, 9))
    e = np.empty((T, 4))
    e[:, 0] = 2.0 * x[:, 3] + rng.normal(0.0, 0.001, T)     # bound to ETF 3, beta 2
    e[:, 1] = -0.5 * x[:, 7] + 0.8 * x[:, 1] + rng.normal(0.0, 0.001, T)   # most correlated with ETF 1
    e[:, 2] = 1.0 * x[:, 5] + rng.normal(0.0, 0.001, T)
    e[:100, 2] = np.nan                                       # late listing
    e[:, 3] = 1.0 * x[:, 0] + rng.normal(0.0, 0.001, T)
    e[:150, 3] = np.nan                                       # only 150 valid rows by row 299
    return e, x


def test_sector_assignment_and_hold_through_month():
    e, x = _sector_market()
    T = e.shape[0]
    starts = np.zeros(T, bool)
    starts[[10, 260, 290]] = True
    k, b, r = _sector_assignment(e, x, starts)
    assert np.all(k[:260] == -1) and np.all(np.isnan(b[:260]))   # row 10: < 189 pairs; before row 10: none
    assert np.all(k[260:, 0] == 3)
    assert b[260, 0] == pytest.approx(2.0, abs=0.02)
    assert r[260, 0] == pytest.approx(1.0, abs=0.01)
    assert np.all(k[260:, 1] == 1)


def test_sector_assignment_min_pairs_and_point_in_time():
    e, x = _sector_market()
    T = e.shape[0]
    starts = np.zeros(T, bool)
    starts[[260, 290]] = True
    k, b, r = _sector_assignment(e, x, starts)
    assert np.all(k[260:290, 2] == -1)        # 161 valid pairs < 189
    assert np.all(k[290:, 2] == 5)            # 191 valid pairs
    assert np.all(k[:, 3] == -1)              # at most 150 valid pairs
    # held constant between month starts
    assert np.all(b[260:290, 0] == b[260, 0])
    # only rows <= the month start matter
    e2, x2 = e.copy(), x.copy()
    e2[261:] = 0.3
    x2[261:] = -0.2
    k2, b2, r2 = _sector_assignment(e2, x2, starts)
    assert np.array_equal(k2[:290], k[:290])
    assert np.allclose(b2[:290], b[:290], equal_nan=True)
    # window is 252 rows: data before row 260-251 = 9 does not matter for row 260
    e3 = e.copy()
    e3[:9, 0] = 5.0
    k3, b3, _ = _sector_assignment(e3, x, starts)
    assert b3[260, 0] == b[260, 0]


def test_gather():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    k = np.array([[2, -1], [0, 1]])
    g = _gather(a, k)
    assert g[0, 0] == 3.0 and np.isnan(g[0, 1]) and g[1, 0] == 4.0 and g[1, 1] == 5.0


def test_news_flag():
    T = 10
    ea = np.zeros((T, 3))
    ne = np.zeros((T, 3))
    ea[2, 0] = 1.0                 # earnings release on row 2
    ne[8, 1] = 1.0                 # non-earnings 8-K on row 8
    ea[:6, 2] = np.nan             # company's first filing at row 6
    f = _news_flag(ea, ne)
    assert np.all(np.isnan(f[:4]))                 # window not full
    assert np.all(f[4:7, 0] == 1.0) and np.all(f[7:, 0] == 0.0)
    assert np.all(f[4:8, 1] == 0.0) and np.all(f[8:, 1] == 1.0)
    assert np.all(np.isnan(f[4:10, 2]))            # unknown on part of t-4..t
    ea[9, 2] = 1.0
    assert _news_flag(ea, ne)[9, 2] == 1.0          # a known hit counts


def test_cohort_book_by_hand():
    T, N = 8, 6
    sw = np.zeros((T, N))
    sw[1, [0, 1]] = 0.25
    sw[1, [2, 3]] = -0.25
    sw[2, 0] = 0.5
    sw[2, 4] = -0.5
    uni = np.ones((T, N), bool)
    uni[4, 1] = False
    w = _cohort_book(sw, uni)
    assert np.all(w[0] == 0.0)
    assert w[1, 0] == pytest.approx(0.05) and w[1, 2] == pytest.approx(-0.05)
    assert w[2, 0] == pytest.approx(0.15) and w[2, 4] == pytest.approx(-0.1)
    assert w[5, 0] == pytest.approx(0.15)          # cohort 1 lives rows 1..5, cohort 2 rows 2..6
    assert w[6, 0] == pytest.approx(0.1) and w[6, 1] == 0.0
    assert np.all(w[7] == 0.0)
    assert w[4, 1] == 0.0 and w[3, 1] == pytest.approx(0.05) and w[5, 1] == pytest.approx(0.05)
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)
    net = w.sum(axis=1)
    assert np.all(np.abs(np.delete(net, 4)) < 1e-12)   # dollar-neutral while all names stay members
    assert net[4] == pytest.approx(-0.05)              # stock 1 closed early on leaving the universe


def test_cohort_book_gross_with_score_weights_and_top():
    rng = np.random.default_rng(3)
    T, N = 30, 200
    score = rng.normal(size=(T, N))
    score[:5] = np.nan                               # before the first walk-forward score
    uni = np.ones((T, N), bool)
    uni[:, 150:] = False
    w_out = {}
    for top in (0.05, 0.1, 0.2):
        sw = score_weights(score, uni, top=top, bottom=top, min_names=100)
        w = _cohort_book(sw, uni)
        assert np.all(w[:5] == 0.0)
        assert np.all(w[:, 150:] == 0.0)
        g = np.abs(w).sum(axis=1)
        assert np.all(g <= 1.0 + 1e-9)
        one = _cohort_book(sw, uni, n_cohorts=1)
        assert np.allclose(np.abs(one[5:]).sum(axis=1), 2 * LEG_GROSS)      # cohort gross 0.2
        assert np.allclose(one[5:].clip(min=0).sum(axis=1), LEG_GROSS)     # 0.1 per leg
        n_long = (one[5:] > 0).sum(axis=1)
        assert np.all(np.abs(n_long - top * 150) <= 1.0)
        assert np.all(np.abs(w.sum(axis=1)) < 1e-9)
        w_out[top] = w
    assert not np.allclose(w_out[0.05], w_out[0.1])
    assert not np.allclose(w_out[0.1], w_out[0.2])
    # fewer than 100 scored members -> no new cohort
    uni2 = np.zeros((T, N), bool)
    uni2[:, :99] = True
    sw2 = score_weights(score, uni2, top=0.1, bottom=0.1, min_names=100)
    assert np.all(_cohort_book(sw2, uni2) == 0.0)


def test_lagged_std_and_flow_sessions():
    T = 70
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 0.01, size=(T, 2))
    sd = _lagged_std(x)
    assert np.all(np.isnan(sd[:63]))
    assert sd[63, 0] == pytest.approx(np.std(x[:63, 0], ddof=1))
    assert sd[69, 1] == pytest.approx(np.std(x[6:69, 1], ddof=1))
    dv = np.full((T, 2), 100.0)
    med = np.full((T, 2), 100.0)
    x[65, 0] = 10 * sd[65, 0]
    dv[65, 0] = 250.0
    x[66, 1] = 10 * sd[66, 1]
    dv[66, 1] = 150.0                    # return condition only
    f = _flow_sessions(x, dv, med)
    assert f[65, 0] and not f[66, 1]
    assert not f[:63].any()


def test_new_entries():
    sw = np.array([[0.5, -0.5, 0.0], [0.5, 0.0, -0.5], [0.0, 0.5, -0.5], [-0.5, 0.5, 0.0]])
    nl, ns = _new_entries(sw)
    assert nl[0].tolist() == [True, False, False] and ns[0].tolist() == [False, True, False]
    assert nl[1].tolist() == [False, False, False] and ns[1].tolist() == [False, False, True]
    assert nl[2].tolist() == [False, True, False] and ns[2].tolist() == [False, False, False]
    assert nl[3].tolist() == [False, False, False] and ns[3].tolist() == [True, False, False]
