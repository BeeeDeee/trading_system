import numpy as np
import pytest

from strategy import (PARAMS, _roll_mean_std, _roll_median_prev, _shift, build_books, shock_dates,
                      top_n_universe, vix_features)


def test_params_match_card():
    assert PARAMS == {"n_estimators": 100, "shock_abs": 0.1}


def test_shift_short_views():
    x = np.arange(6, dtype=float).reshape(3, 2)
    s = _shift(x, 1)
    assert np.isnan(s[0]).all() and (s[1:] == x[:-1]).all()
    assert np.isnan(_shift(x, 5)).all()            # window longer than the view


def test_rolling_mean_std_trailing_only():
    x = np.arange(10, dtype=float)[:, None]
    m, s = _roll_mean_std(x, 3)
    assert np.isnan(m[0, 0])                       # 1 of 3 < 2/3 of the window
    assert m[1, 0] == pytest.approx(0.5)           # 2 of 3 values
    assert m[5, 0] == pytest.approx(4.0)           # rows 3, 4, 5
    assert s[5, 0] == pytest.approx(1.0)           # sample std of 3, 4, 5
    y = x.copy()
    y[7:] = 1e6                                    # future change does not touch earlier rows
    m2, _ = _roll_mean_std(y, 3)
    assert (m2[:7] == m[:7])[np.isfinite(m[:7])].all()


def test_median_of_previous_rows_excludes_today():
    x = np.array([1, 5, 3, 100, 2, 4], dtype=float)[:, None]
    med = _roll_median_prev(x, 3)
    assert np.isnan(med[0, 0]) and np.isnan(med[1, 0])   # 0 and 1 previous values
    assert med[2, 0] == pytest.approx(3.0)               # median(1, 5): 2 of 3 values
    assert med[3, 0] == pytest.approx(3.0)               # median(1, 5, 3), not including 100
    assert med[4, 0] == pytest.approx(5.0)               # median(5, 3, 100)
    x2 = x.copy()
    x2[2, 0] = np.nan
    assert _roll_median_prev(x2, 3)[4, 0] == pytest.approx(52.5)   # median of the finite (5, 100)


def test_vix_new_observations_and_shock():
    # forward-filled series: weekend rows repeat the last value
    v = np.array([np.nan, 20.0, 20.0, 22.0, 22.0, 22.0, 19.0, 19.0])
    new, shock, state = vix_features(v)
    assert new.tolist() == [False, True, False, True, False, False, True, False]
    assert np.isnan(shock[1]) and np.isnan(shock[2])
    assert shock[3] == pytest.approx(np.log(22 / 20)) and shock[5] == pytest.approx(np.log(22 / 20))
    assert shock[7] == pytest.approx(np.log(19 / 22))
    assert np.isnan(state).all()                         # fewer than 252 previous sessions
    sd = shock_dates(new, shock, 0.1)
    assert sd.tolist() == [False, False, False, False, False, False, True, False]   # |log(19/22)| = 0.147
    sd2 = shock_dates(new, shock, 0.07)
    assert sd2[3] and sd2[6] and not sd2[4]              # only rows with a NEW observation
    assert not shock_dates(new, shock, 0.15).any()


def test_vix_state_uses_previous_252_sessions():
    obs = np.concatenate([np.full(252, 20.0), [30.0]])
    obs[::2] += 0.5                                      # make every value a new observation
    new, shock, state = vix_features(obs)
    assert new.all()
    assert np.isnan(state[251])
    assert state[252] == pytest.approx(np.log(obs[252] / np.median(obs[:252])))
    # truncation: earlier rows unchanged
    _, s2, st2 = vix_features(obs[:200])
    assert np.array_equal(s2, shock[:200], equal_nan=True) and np.array_equal(st2, state[:200], equal_nan=True)


def test_build_books_legs_hold_and_replace():
    T, N = 14, 12
    score = np.full((T, N), np.nan)
    score[1] = np.arange(N, dtype=float)                 # 12 scored: k = round(2.4) = 2
    score[8] = -np.arange(N, dtype=float)
    rebuild = np.zeros(T, bool)
    rebuild[1] = True
    member = np.ones((T, N), bool)
    w, e = build_books(score, rebuild, member, 5)
    assert (w[0] == 0).all()
    assert w[1, 11] == pytest.approx(0.25) and w[1, 10] == pytest.approx(0.25)
    assert w[1, 0] == pytest.approx(-0.25) and w[1, 1] == pytest.approx(-0.25)
    assert np.abs(w[1]).sum() == pytest.approx(1.0) and w[1].sum() == pytest.approx(0.0)
    assert all((w[t] == w[1]).all() for t in range(1, 6))   # held rows 1..5
    assert (w[6:] == 0).all()                               # flat afterwards
    assert e[1, 11] == 1 and e[1, 0] == -1 and (e[2:] == 0).all()
    # a newer shock inside the window replaces the book and restarts the clock
    rebuild[4] = True
    score[4] = -np.arange(N, dtype=float)
    w2, _ = build_books(score, rebuild, member, 5)
    assert w2[4, 0] == pytest.approx(0.25) and w2[4, 11] == pytest.approx(-0.25)
    assert (w2[8] == w2[4]).all() and (w2[9] == 0).all()


def test_build_books_k_rounding_and_universe_exit():
    T, N = 6, 20
    score = np.full((T, N), np.nan)
    score[0, :17] = np.arange(17, dtype=float)           # 17 scored: k = round(3.4) = 3
    rebuild = np.zeros(T, bool)
    rebuild[0] = True
    member = np.ones((T, N), bool)
    member[2, 16] = False                                # top coin leaves the universe on row 2
    w, _ = build_books(score, rebuild, member, 5)
    assert (w[0] > 0).sum() == 3 and (w[0] < 0).sum() == 3
    assert w[0, 16] == pytest.approx(0.5 / 3)
    assert w[1, 16] == pytest.approx(0.5 / 3) and w[2, 16] == 0 and w[3, 16] == 0   # closed, stays closed
    assert w[3, 15] == pytest.approx(0.5 / 3)            # others unchanged, no renormalisation
    assert (np.abs(w).sum(axis=1) <= 1 + 1e-12).all()
    assert (w[17 - 17:, 17:] == 0).all()                 # unscored columns never held


def test_build_books_minimum_k_is_two():
    T, N = 3, 10
    score = np.full((T, N), np.nan)
    score[0] = np.arange(N, dtype=float)                 # 10 scored: k = round(2.0) = 2
    rebuild = np.array([True, False, False])
    w, _ = build_books(score, rebuild, np.ones((T, N), bool), 5)
    assert (w[0] > 0).sum() == 2 and (w[0] < 0).sum() == 2


def test_fallback_universe_top_n_trailing_volume():
    T, N = 5, 4
    dv = np.array([[1, 2, 3, 4]] * T, dtype=float)
    dv[3:, 0] = 100.0
    listed = np.ones((T, N), bool)
    listed[:, 3] = False                                 # not listed: never a member
    u = top_n_universe(dv, listed, 2, 2)
    assert u[0].tolist() == [False, True, True, False]
    assert u[3].tolist() == [True, False, True, False]   # coin 0 jumps in on its own row's volume
    u_trunc = top_n_universe(dv[:3], listed[:3], 2, 30)  # window longer than the view
    assert u_trunc.shape == (3, N)
