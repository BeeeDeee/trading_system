import numpy as np
import pytest

from strategy import (PARAMS, build_book, compute_features, weekday, _rolling_sum_count, _lag,
                      HORIZON, MIN_SCORED, target_weights)


def _days(start_int, n):
    return (np.arange(n) + start_int).astype("datetime64[D]")


def test_params_match_card():
    assert PARAMS == {"n_estimators": 100, "n_leg": 6}
    assert HORIZON == 21 and MIN_SCORED == 15
    assert callable(target_weights)


def test_weekday_sunday():
    # epoch day 0 is a Thursday; day 3 a Sunday; day 4 a Monday
    wd = weekday(_days(0, 8))
    assert list(wd) == [3, 4, 5, 6, 0, 1, 2, 3]


def test_lag_and_short_views():
    x = np.arange(5, dtype=float)[:, None]
    assert np.isnan(_lag(x, 7)).all()                 # view shorter than the lag
    out = _lag(x, 2)
    assert np.isnan(out[:2]).all() and out[2, 0] == 0 and out[4, 0] == 2
    s, c = _rolling_sum_count(np.array([[1.0], [np.nan], [3.0], [4.0]]), 90)
    assert list(s[:, 0]) == [1, 1, 4, 8] and list(c[:, 0]) == [1, 1, 2, 3]


def _market(T=120, N=3, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    qv = rng.uniform(1e6, 2e6, (T, N))
    basis = rng.normal(0, 0.001, (T, N))
    member = np.ones((T, N), dtype=bool)
    return close, qv, basis, member


def test_features_by_hand():
    close, qv, basis, member = _market()
    (mom28, ret7, basis7, volshock, rvol28), scored = compute_features(close, qv, basis, member)
    t = 100
    assert mom28[t, 0] == pytest.approx(np.log(close[t, 0] / close[t - 28, 0]))
    assert ret7[t, 1] == pytest.approx(np.log(close[t, 1] / close[t - 7, 1]))
    assert basis7[t, 2] == pytest.approx(basis[t - 6:t + 1, 2].mean())
    assert volshock[t, 0] == pytest.approx(np.log(qv[t - 6:t + 1, 0].mean() / qv[t - 89:t + 1, 0].mean()))
    r = np.diff(np.log(close[:, 0]))
    assert rvol28[t, 0] == pytest.approx(np.std(r[t - 28:t], ddof=1))   # returns ending at t-27..t
    # scored needs 80 valid closes in the last 90 rows: first scored row is index 79
    assert not scored[78].any() and scored[79].all()


def test_scoring_rules_missing_data_and_membership():
    close, qv, basis, member = _market()
    close[:, 0] = np.where(np.arange(120) < 50, np.nan, close[:, 0])     # late listing
    basis[95:100, 1] = np.nan                                            # 2 valid of last 7 at t=99
    member[110, 2] = False
    feats, scored = compute_features(close, qv, basis, member)
    assert not scored[:, 0].any()                                        # 80 valid closes only at row 129
    assert not scored[103, 1] and scored[104, 1]                         # basis rule: >= 5 of 7
    assert not scored[110, 2] and scored[111, 2]                         # member on t
    for f in feats:
        assert np.isnan(f[~scored]).all()


def test_book_cohorts_weights_and_hold():
    T, N, n_leg = 60, 20, 4
    score = np.tile(np.arange(N, dtype=float), (T, 1))       # col 19 best, col 0 worst
    member = np.ones((T, N), dtype=bool)
    decision = np.zeros(T, dtype=bool)
    decision[[3, 10, 17, 24]] = True
    W = build_book(score, member, decision, n_leg)
    w = 1.0 / (6 * n_leg)
    assert np.allclose(W[:3], 0)                               # cash before first cohort
    assert np.allclose(W[3, 16:], w) and np.allclose(W[3, :4], -w) and np.allclose(W[3, 4:16], 0)
    assert np.allclose(W[17:24, 16:], 3 * w)                   # three cohorts overlap, weights add
    assert np.allclose(np.abs(W[17:24]).sum(axis=1), 1.0)      # gross 1, net 0
    assert np.allclose(W[17:24].sum(axis=1), 0.0)
    assert np.allclose(W[45:], 0)                              # last cohort (row 24) closes at 24+21
    assert np.allclose(W[44, 16:], w)
    assert (np.abs(W).sum(axis=1) <= 1.0 + 1e-12).all()


def test_book_universe_exit_stays_cash_and_min_scored():
    T, N, n_leg = 40, 20, 4
    score = np.tile(np.arange(N, dtype=float), (T, 1))
    member = np.ones((T, N), dtype=bool)
    member[8, 19] = False                                      # leaves for one row, comes back
    decision = np.zeros(T, dtype=bool)
    decision[3] = True
    W = build_book(score, member, decision, n_leg)
    w = 1.0 / (6 * n_leg)
    assert np.allclose(W[3:8, 19], w) and np.allclose(W[8:, 19], 0)    # slot stays in cash
    assert np.allclose(W[3:24, 18], w) and np.allclose(W[24:, 18], 0)
    # fewer than 15 scored members: no cohort
    score2 = score.copy()
    score2[3, 6:] = np.nan                                     # only 6 finite
    assert np.allclose(build_book(score2, member, decision, n_leg), 0)
    score3 = score.copy()
    score3[3, 15:] = np.nan                                    # exactly 15 finite: cohort formed
    W3 = build_book(score3, member, decision, n_leg)
    assert np.allclose(W3[3, 11:15], w) and np.allclose(W3[3, 15:], 0)


def test_n_leg_changes_output():
    T, N = 30, 20
    score = np.tile(np.arange(N, dtype=float), (T, 1))
    member = np.ones((T, N), dtype=bool)
    decision = np.zeros(T, dtype=bool)
    decision[0] = True
    a = build_book(score, member, decision, 4)
    b = build_book(score, member, decision, 6)
    assert not np.allclose(a, b)
    assert (np.abs(a[0]) > 0).sum() == 8 and (np.abs(b[0]) > 0).sum() == 12
    assert np.abs(b[0]).sum() == pytest.approx(1.0 / 3.0)
