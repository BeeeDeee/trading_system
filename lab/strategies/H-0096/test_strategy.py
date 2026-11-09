import numpy as np
import pytest

from lab.framework.ml import cs_rank, cs_zscore
from strategy import PARAMS, target_weights  # noqa: F401
from strategy import book_from_legs, cohort_legs, ear1_feature, lag, rolling_std_full, window_flag


def _weekdays(n_cal=900, start=12000):
    allc = np.arange(start, start + n_cal).astype("datetime64[D]")
    return allc[np.is_busday(allc)]


def _window_reference(ea, dates):
    T, N = ea.shape
    out = np.zeros((T, N), dtype=bool)
    pred = np.busday_offset((dates.astype(np.int64) + 364).astype(dates.dtype), 0, roll="forward")
    for t in range(T):
        tgt = np.busday_offset(dates[t], 3, roll="forward")
        for j in range(N):
            if not np.isfinite(ea[t, j]):
                continue
            if np.any(ea[max(0, t - 19): t + 1, j] == 1):
                continue
            for s in range(t + 1):
                if ea[s, j] == 1 and pred[s] == tgt:
                    out[t, j] = True
                    break
    return out


def test_params_match_card():
    assert set(PARAMS) == {"leg_fraction", "num_leaves"}
    assert PARAMS["leg_fraction"] == 0.33 and PARAMS["num_leaves"] == 8


def test_window_flag_hand_case():
    dates = _weekdays()
    T = dates.size
    ea = np.zeros((T, 3))
    s = 10
    ea[s, :] = 1.0
    target = (dates[s : s + 1].astype(np.int64) + 364).astype(dates.dtype)[0]   # same weekday a year later
    p_row = int(np.searchsorted(dates, target))
    assert dates[p_row] == target
    assert np.busday_count(dates[p_row - 3], target) == 3
    t = p_row - 3                                   # no holidays in this calendar
    ea[t - 5, 1] = 1.0                              # stock 1 reported inside the last 20 sessions
    ea[t, 2] = np.nan                               # stock 2 unknown on t
    win = window_flag(ea, dates)
    assert win[t, 0] and win[:, 0].sum() == 1
    assert not win[t, 1]
    assert not win[t, 2]


def test_window_flag_matches_bruteforce_and_is_point_in_time():
    rng = np.random.default_rng(1)
    dates = _weekdays(700)
    T = dates.size
    ea = (rng.random((T, 4)) < 0.03).astype(float)
    ea[:30, 3] = np.nan
    win = window_flag(ea, dates)
    ref = _window_reference(ea, dates)
    assert np.array_equal(win, ref)
    assert win.sum() > 0
    for k in (50, 300, 400, T - 1):
        assert np.array_equal(window_flag(ea[:k], dates[:k]), win[:k])


def test_lag_and_short_views():
    x = np.arange(10.0).reshape(5, 2)
    y = lag(x, 2)
    assert np.all(np.isnan(y[:2])) and np.array_equal(y[2:], x[:3])
    assert np.all(np.isnan(lag(x, 7)))


def test_rolling_std_full():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 3))
    x[50, 1] = np.nan
    s = rolling_std_full(x, 63)
    assert np.all(np.isnan(s[:62]))
    assert s[70, 0] == pytest.approx(np.std(x[8:71, 0], ddof=1), rel=1e-9)
    assert np.isnan(s[70, 1]) and np.isfinite(s[99, 2])
    assert np.all(np.isnan(rolling_std_full(x[:20], 63)))


def test_ear1_feature():
    T, N = 6, 3
    r = np.array([[0.01, 0.03, 0.02]] * T)
    r[2] = [0.05, 0.01, 0.03]
    vol = np.full((T, N), 0.02)
    uni = np.ones((T, N), dtype=bool)
    ea = np.zeros((T, N))
    ea[2, 0] = 1.0
    out = ear1_feature(r, ea, vol, uni)
    assert np.all(np.isnan(out[:2, 0])) and np.all(np.isnan(out[:, 1:]))
    assert out[2, 0] == pytest.approx((0.05 - 0.03) / 0.02)
    assert np.allclose(out[2:, 0], out[2, 0])     # carried forward until the next release


def test_cohort_legs_percentiles_and_leg_fraction():
    N = 12
    score = np.full((2, N), np.nan)
    score[0, :10] = np.arange(10.0)
    score[0, 10] = 100.0                           # not a member: ignored in the percentile
    uni = np.ones((2, N), dtype=bool)
    uni[:, 10] = False
    win = np.zeros((2, N), dtype=bool)
    win[0, [0, 5, 9, 10]] = True
    legs = cohort_legs(score, win, uni, 0.33)
    assert legs[0, 9] == 1 and legs[0, 0] == -1 and legs[0, 5] == 0 and legs[0, 10] == 0
    legs5 = cohort_legs(score, win, uni, 0.5)
    assert legs5[0, 5] == 1 and legs5[0, 9] == 1 and legs5[0, 0] == -1
    # only one side present: no cohort
    win2 = np.zeros((2, N), dtype=bool)
    win2[0, [8, 9]] = True
    assert not cohort_legs(score, win2, uni, 0.33).any()


def test_book_weights_hold_sizing_and_exit():
    T, N = 12, 20
    uni = np.ones((T, N), dtype=bool)
    legs = np.zeros((T, N), dtype=np.int8)
    legs[0, [0, 1]] = 1
    legs[0, 2] = -1
    W = book_from_legs(legs, uni)
    for t in range(5):
        assert W[t, 0] == pytest.approx(0.015) and W[t, 2] == pytest.approx(-0.03)
    assert np.all(W[5:] == 0)
    # leaving the universe closes the position for the rest of the cohort
    uni2 = uni.copy()
    uni2[2, 1] = False
    W2 = book_from_legs(legs, uni2)
    assert W2[1, 1] == pytest.approx(0.015) and np.all(W2[2:, 1] == 0)
    # cap: 20 marked names every day, 5 overlapping cohorts -> gross exactly 1
    legs3 = np.zeros((T, N), dtype=np.int8)
    legs3[:, :10] = 1
    legs3[:, 10:] = -1
    W3 = book_from_legs(legs3, uni)
    gross = np.abs(W3).sum(axis=1)
    assert np.all(gross <= 1.0 + 1e-12) and gross[6] == pytest.approx(1.0)
    assert np.allclose(W3[6].sum(), 0.0)


def test_cs_helpers_keep_nan():
    x = np.array([[1.0, np.nan, 3.0, 2.0], [4.0, 1.0, 2.0, 3.0]])
    uni = np.ones((2, 4), dtype=bool)
    uni[1, 0] = False
    z = np.asarray(cs_zscore(cs_rank(x, uni), uni), dtype=float)
    assert np.isnan(z[0, 1]) and np.isfinite(z[0, [0, 2, 3]]).all()
    assert z[0, 2] > z[0, 3] > z[0, 0]
