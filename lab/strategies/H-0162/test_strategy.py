import numpy as np
import pytest

from strategy import (
    PARAMS, balance_features, beta_balanced_row, leg_books, long_leg_share, rate_z, relative_return, schedule,
    shock_rows, stock_betas, week_end_mask,
)


def _dates(start_days, n):
    # consecutive calendar days counted from epoch day 0 (a Thursday)
    return (np.arange(n) + start_days).astype("datetime64[D]")


def test_params_match_card():
    assert PARAMS == {"n_estimators": 100, "shock_z": 1.0}


def test_week_end_is_friday():
    d = _dates(0, 14)                       # Thu 1970-01-01 .. Wed 1970-01-14
    m = week_end_mask(d)
    assert np.flatnonzero(m).tolist() == [1, 8]   # Fri 01-02, Fri 01-09


def test_rate_z_hand_computed():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=400))
    x[:3] = np.nan                          # leading NaN before the series starts
    z = rate_z(x, k=5, win=252)
    v = x[3:]
    first = 3 + 5 + 252 - 1
    assert np.all(np.isnan(z[:first]))
    assert np.isfinite(z[first])
    for t in (first, 350, 399):
        j = t - 3
        chg = v[5:] - v[:-5]
        window = chg[j - 5 - 251:j - 5 + 1]
        expect = (v[j] - v[j - 5]) / np.std(window, ddof=1)
        assert z[t] == pytest.approx(expect, rel=1e-9)


def test_rate_z_point_in_time():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=500))
    full = rate_z(x)
    cut = rate_z(x[:320])
    assert np.allclose(full[:320], cut, equal_nan=True)


def test_shock_rows_threshold_and_nan():
    zc = np.array([0.5, 1.2, np.nan, -1.0, 0.2, 0.9])
    z2 = np.array([0.5, 0.0, 3.0, 0.0, -1.5, 0.9])
    we = np.array([True, True, True, True, False, True])
    assert shock_rows(zc, z2, we, 1.0).tolist() == [False, True, False, True, False, False]
    assert shock_rows(zc, z2, we, 0.75).tolist() == [False, True, False, True, False, True]
    assert shock_rows(zc, z2, we, 1.25).tolist() == [False, False, False, False, False, False]


def test_balance_features():
    nan = np.nan
    dep = np.array([[50.0, nan, 0.0, 10.0, 5.0]])
    assets = np.array([[100.0, 200.0, 100.0, nan, 100.0]])
    debtc = np.array([[10.0, 20.0, nan, 1.0, 1.0]])
    debt = np.array([[30.0, 40.0, 50.0, 1.0, 1.0]])
    intexp = np.array([[5.0, 300.0, 1.0, 1.0, 1.0]])
    rev = np.array([[50.0, 100.0, 0.0, 10.0, nan]])
    d, sd, td, ib = balance_features(dep, assets, debtc, debt, intexp, rev)
    assert d[0, :3].tolist() == [0.5, 0.0, 0.0]          # missing / zero deposits -> 0 with assets
    assert np.isnan(d[0, 3])                             # no assets -> missing
    assert sd[0, 0] == 0.1 and sd[0, 1] == 0.1 and np.isnan(sd[0, 2])
    assert td[0, 0] == 0.3 and td[0, 1] == 0.2
    assert ib[0, 0] == 0.1 and ib[0, 1] == 2.0          # clipped at 2
    assert np.isnan(ib[0, 2]) and np.isnan(ib[0, 3]) and np.isnan(ib[0, 4])   # rev <= 0, no assets, rev NaN


def test_relative_return_members_only():
    r5 = np.array([[0.1, 0.3, 5.0, np.nan]])
    uni = np.array([[True, True, False, True]])
    rel = relative_return(r5, uni)
    assert rel[0, 0] == pytest.approx(-0.1) and rel[0, 1] == pytest.approx(0.1)
    assert rel[0, 2] == pytest.approx(4.8) and np.isnan(rel[0, 3])


def test_leg_books_floor_and_shock_rows():
    T, N = 4, 80
    uni = np.ones((T, N), dtype=bool)
    score = np.tile(np.arange(N, dtype=float), (T, 1))
    score[1, 40:] = np.nan                  # only 40 scored members on row 1
    uni[2, 70:] = False                     # non-members are never held
    shock = np.array([True, True, True, False])
    sides = leg_books(score, uni, shock)
    assert (sides[0] > 0).sum() > 0 and (sides[0] < 0).sum() > 0
    assert sides[0, -1] == 1.0 and sides[0, 0] == -1.0         # top score long, bottom short
    assert (sides[0] != 0).sum() <= 0.2 * N + 2
    assert (sides[1] == 0).all()                               # fewer than 50 scored: flat
    assert (sides[2, 70:] == 0).all() and sides[2, 69] == 1.0
    assert (sides[3] == 0).all()                               # not a shock week


def test_long_leg_share():
    assert long_leg_share(1.0, 1.0) == pytest.approx(0.5)
    assert long_leg_share(1.5, 1.0) == pytest.approx(0.4)      # 0.4 * 1.5 = 0.6 * 1.0
    assert long_leg_share(3.0, 1.0) == pytest.approx(0.3)      # root 0.25 capped: short leg at 0.7
    assert long_leg_share(0.2, 2.0) == pytest.approx(0.7)
    assert long_leg_share(np.nan, 1.0) == 0.5


def test_stock_betas_and_balanced_row():
    rng = np.random.default_rng(2)
    T = 300
    m = rng.normal(0, 0.01, T)
    r = np.column_stack([2.0 * m, 1.0 * m, 0.5 * m, m + 0.001]).astype(np.float32)
    b = stock_betas(r, m, T - 1, [0, 1, 2])
    assert b == pytest.approx([2.0, 1.0, 0.5], rel=1e-4)
    assert np.all(np.isnan(stock_betas(r, m, 100, [0])))       # window not full yet
    side = np.array([1.0, 0.0, -1.0, 0.0])                     # long beta 2, short beta 0.5
    w = beta_balanced_row(side, r, m, T - 1)
    assert w[0] == pytest.approx(0.3) and w[2] == pytest.approx(-0.7)
    assert np.abs(w).sum() == pytest.approx(1.0)
    side = np.array([1.0, -1.0, 0.0, 0.0])                     # long beta 2, short beta 1 -> 1/3, 2/3
    w = beta_balanced_row(side, r, m, T - 1)
    assert w[0] == pytest.approx(1 / 3, rel=1e-3) and w[1] == pytest.approx(-2 / 3, rel=1e-3)
    assert w[0] * 2.0 + w[1] * 1.0 == pytest.approx(0.0, abs=1e-3)


def test_schedule_hold_replace_and_leavers():
    T, N = 16, 3
    uni = np.ones((T, N), dtype=bool)
    uni[3, 1] = False                       # name 1 leaves during the first cohort
    shock = np.zeros(T, dtype=bool)
    shock[[1, 9, 14]] = True
    rows = np.zeros((T, N))
    rows[1] = [0.5, -0.5, 0.0]
    rows[9] = [0.0, 0.4, -0.6]
    rows[14] = [0.0, 0.0, 0.0]              # fewer than 50 scored: flat that week
    W = schedule(shock, rows, uni, hold=5)
    assert W[1].tolist() == [0.5, -0.5, 0.0]
    assert np.isnan(W[2]).all()
    assert np.isnan(W[3, 0]) and W[3, 1] == 0.0 and np.isnan(W[3, 2])
    assert (W[6] == 0).all()                # closed 5 sessions after the decision
    assert np.isnan(W[7]).all() and np.isnan(W[8]).all()
    assert W[9].tolist() == [0.0, 0.4, -0.6]
    assert (W[14] == 0).all()               # a flat shock week replaces the cohort
    assert np.isnan(W[15]).all()


def test_schedule_consecutive_shock_weeks():
    T, N = 12, 2
    uni = np.ones((T, N), dtype=bool)
    shock = np.zeros(T, dtype=bool)
    shock[[0, 5]] = True
    rows = np.zeros((T, N))
    rows[0] = [0.5, -0.5]
    rows[5] = [-0.5, 0.5]
    W = schedule(shock, rows, uni, hold=5)
    assert W[5].tolist() == [-0.5, 0.5]     # new cohort at the open where the old one closes
    assert (W[10] == 0).all()
    gross = np.nansum(np.abs(W), axis=1)
    assert (gross <= 1.0 + 1e-12).all()
