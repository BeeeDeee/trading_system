import numpy as np
import pytest

from strategy import (
    PARAMS,
    apply_universe_exits,
    balance_sheet_features,
    bank_mask,
    credit_shock_z,
    leg_book,
    market_return,
    realized_vol,
    size_factor,
    spread_change,
    weekly_decision_days,
    weekday,
)


def _days(start_offset, n):
    # 2001-01-01 was a Monday: 11323 days after 1970-01-01
    return (np.arange(n) + 11323 + start_offset).astype("datetime64[D]")


def test_params_match_card():
    assert PARAMS == {"n_estimators": 100, "top": 0.2}


def test_weekday_and_friday_decisions():
    d = _days(0, 14)
    assert list(weekday(d)[:7]) == [0, 1, 2, 3, 4, 5, 6]
    dec = weekly_decision_days(d)
    assert list(np.flatnonzero(dec)) == [4, 11]


def test_decision_depends_only_on_own_date():
    d = _days(0, 30)
    full = weekly_decision_days(d)
    for cut in (3, 10, 17):
        assert np.array_equal(weekly_decision_days(d[:cut]), full[:cut])


def test_spread_change():
    s = np.arange(10, dtype=float) ** 2
    ds = spread_change(s, 5)
    assert np.all(np.isnan(ds[:5]))
    assert ds[5] == 25.0 - 0.0
    assert ds[9] == 81.0 - 16.0


def test_market_return_members_only():
    T, N = 8, 3
    idx = np.ones((T, N))
    idx[:, 0] = 1.1 ** np.arange(T)
    idx[:, 1] = 1.0
    idx[:, 2] = 2.0 ** np.arange(T)
    listed = np.ones((T, N), bool)
    members = np.ones((T, N), bool)
    members[:, 2] = False
    m = market_return(idx, listed, members, 5)
    assert np.all(np.isnan(m[:5]))
    assert m[6] == pytest.approx(((1.1 ** 5 - 1) + 0.0) / 2)
    listed[1, 0] = False      # not listed 5 rows before row 6
    m2 = market_return(idx, listed, members, 5)
    assert m2[6] == pytest.approx(0.0)


def test_credit_shock_z_by_hand():
    rng = np.random.default_rng(1)
    T, W = 40, 20
    m = rng.normal(size=T)
    ds = 0.3 + 2.0 * m + rng.normal(size=T) * 0.1
    z = credit_shock_z(ds, m, W)
    assert np.all(np.isnan(z[:W]))
    t = 30
    x, y = m[t - W:t], ds[t - W:t]
    b, a = np.polyfit(x, y, 1)
    res = y - a - b * x
    assert z[t] == pytest.approx((ds[t] - a - b * m[t]) / res.std(), rel=1e-9)


def test_credit_shock_needs_full_window_and_is_point_in_time():
    rng = np.random.default_rng(2)
    T, W = 60, 20
    m = rng.normal(size=T)
    ds = m + rng.normal(size=T)
    ds[25] = np.nan
    z = credit_shock_z(ds, m, W)
    assert np.all(np.isnan(z[25:46]))   # window contains the NaN through row 45
    assert np.isfinite(z[46])
    full = credit_shock_z(ds, m, W)
    for cut in (30, 50):
        ds2 = ds.copy()
        ds2[cut + 1:] = rng.normal(size=T - cut - 1) * 100
        z2 = credit_shock_z(ds2, m, W)
        np.testing.assert_array_equal(z2[:cut + 1], full[:cut + 1])


def test_positive_z_means_widening_beyond_market():
    T, W = 60, 20
    rng = np.random.default_rng(3)
    m = rng.normal(size=T)
    ds = -m + rng.normal(size=T) * 0.1
    ds[-1] = -m[-1] + 5.0
    z = credit_shock_z(ds, m, W)
    assert z[-1] > 5


def test_size_factor():
    z = np.array([np.nan, 0.0, 0.49, -0.5, 0.8, -1.7, 3.0])
    g = size_factor(z)
    np.testing.assert_allclose(g, [0, 0, 0, 0.5, 0.8, 1.0, 1.0])


def test_bank_mask():
    d = np.array([[np.nan, 0.0, 5.0, -1.0]])
    assert list(bank_mask(d)[0]) == [False, False, True, False]


def test_balance_sheet_features():
    assets = np.array([[100.0, 100.0, 100.0, np.nan, 0.0, 100.0]])
    debt = np.array([[50.0, 0.0, 40.0, 10.0, 10.0, np.nan]])
    debtc = np.array([[20.0, 0.0, 10.0, 1.0, 1.0, 1.0]])
    cash = np.array([[5.0, 30.0, 20.0, 1.0, 1.0, 1.0]])
    net, share, lev = balance_sheet_features(assets, debt, debtc, cash)
    np.testing.assert_allclose(net[0, :3], [0.15, -0.30, -0.10])
    np.testing.assert_allclose(share[0, :3], [0.4, 0.0, 0.25])
    np.testing.assert_allclose(lev[0, :3], [0.5, 0.0, 0.4])
    for f in (net, share, lev):
        assert np.all(np.isnan(f[0, 3:]))      # missing assets / assets 0 / missing debt


def test_realized_vol_min_days_and_value():
    rng = np.random.default_rng(4)
    T = 120
    r = rng.normal(scale=0.01, size=(T, 2))
    r[:40, 1] = np.nan                      # late listing
    v = realized_vol(r, 63, 50)
    assert np.all(np.isnan(v[:49, 0]))
    assert v[49, 0] == pytest.approx(np.std(r[:50, 0], ddof=1))
    assert v[100, 0] == pytest.approx(np.std(r[38:101, 0], ddof=1))
    assert np.all(np.isnan(v[:89, 1]))       # 50 valid days reached at row 89
    assert v[89, 1] == pytest.approx(np.std(r[40:90, 1], ddof=1))
    short = realized_vol(r[:10], 63, 50)     # view shorter than the window
    assert short.shape == (10, 2) and np.all(np.isnan(short))


def test_leg_book_legs_min_names_and_scaling():
    rng = np.random.default_rng(5)
    T, N = 3, 250
    score = rng.normal(size=(T, N))
    members = np.ones((T, N), bool)
    members[1, 160:] = False                 # 160 members: still >= 100
    members[2, 90:] = False                  # 90 members: flat
    score[0, :5] = np.nan                    # unscored
    for top in (0.1, 0.2, 0.3):
        w = leg_book(score, members, top)
        assert w[0].sum() == pytest.approx(0.0, abs=1e-9)
        assert w[w > 0].size and np.all(w[0] >= -0.5)
        assert w[0][w[0] > 0].sum() == pytest.approx(0.5)
        assert w[0][w[0] < 0].sum() == pytest.approx(-0.5)
        assert np.all(w[0, :5] == 0)
        assert np.all(w[1, 160:] == 0)
        assert np.all(w[2] == 0)
        # top leg holds the highest scores
        ok = np.isfinite(score[0])
        thr = np.quantile(score[0][ok], 1 - top)
        assert np.all(score[0][w[0] > 0] >= thr - 1e-9)
    n1 = (leg_book(score, members, 0.1)[0] > 0).sum()
    n3 = (leg_book(score, members, 0.3)[0] > 0).sum()
    assert n3 > n1
    g = size_factor(np.array([0.3, 0.7, 2.0]))
    scaled = leg_book(score, members, 0.2) * g[:, None]
    assert np.all(np.abs(scaled).sum(axis=1) <= 1.0 + 1e-9)
    assert np.all(scaled[0] == 0)
    assert np.abs(scaled[1]).sum() == pytest.approx(0.7)


def test_universe_exit_closes_at_first_row_outside():
    T, N = 6, 3
    w = np.full((T, N), np.nan)
    w[0] = [0.25, -0.25, 0.0]
    w[4] = [0.0, -0.25, 0.25]
    uni = np.ones((T, N), bool)
    uni[2, 0] = False
    uni[3, 0] = False
    uni[5, 1] = False
    out = apply_universe_exits(w, uni)
    assert out[2, 0] == 0.0 and np.isnan(out[2, 1]) and np.isnan(out[2, 2])
    assert np.all(np.isnan(out[3]))          # already closed, nothing to do
    np.testing.assert_array_equal(out[4], w[4])
    assert out[5, 1] == 0.0 and np.isnan(out[5, 0])
    assert np.all(np.isnan(out[1]))
