import dataclasses

import numpy as np
import pytest

from strategy import PARAMS, target_weights, decision_days, legs_from, shock_z, _easter, _ffill

D2004 = 12418  # day number of 2004-01-01 (a Thursday)


def _days(a, b):
    return np.arange(a, b).astype("datetime64[D]")


# ------------------------------------------------------------------ calendar
def test_easter_known_years():
    m, d = _easter(np.array([2000, 2004, 2008, 2019, 2011]))
    assert list(zip(m.tolist(), d.tolist())) == [(4, 23), (4, 11), (3, 23), (4, 21), (4, 24)]


def test_decision_days_friday_and_good_friday_thursday():
    days = _days(D2004 + 84, D2004 + 105)              # 2004-03-25 .. 2004-04-14
    wd = (days.astype(np.int64) + 3) % 7
    good_friday = D2004 + 99                           # 2004-04-09
    keep = (wd < 5) & (days.astype(np.int64) != good_friday)
    dates = days[keep]
    dec = decision_days(dates)
    dn = dates.astype(np.int64)
    got = set(dn[dec].tolist())
    # Fridays 2004-03-26, 2004-04-02, and Thursday 2004-04-08 (Good Friday week)
    assert got == {D2004 + 85, D2004 + 92, D2004 + 98}


def test_decision_days_july_third_and_christmas_eve():
    # 2015-07-03 was a Friday holiday (Jul 4 on Saturday): Thursday 2015-07-02 is the decision.
    d2015 = D2004 + 11 * 365 + 3                       # 2015-01-01 (leap years 2004, 2008, 2012)
    thu = d2015 + 181 + 1                              # 2015-07-02 (Jan..Jun = 181 days)
    assert (thu + 3) % 7 == 3
    assert decision_days(_days(thu, thu + 1))[0]
    # 2015-07-09 (an ordinary Thursday) is not
    assert not decision_days(_days(thu + 7, thu + 8))[0]
    # 2010-12-23 Thursday before Friday Dec 24 (Christmas on Saturday)
    d2010 = D2004 + 6 * 365 + 2
    thu2 = d2010 + 334 + 22                            # 2010-12-23
    assert (thu2 + 3) % 7 == 3
    assert decision_days(_days(thu2, thu2 + 1))[0]


def test_ffill_keeps_leading_nan():
    x = np.array([np.nan, np.nan, 1.0, np.nan, 3.0, np.nan])
    out = _ffill(x)
    assert np.isnan(out[:2]).all()
    assert out[2:].tolist() == [1.0, 1.0, 3.0, 3.0]


# ------------------------------------------------------------------ shock
def test_shock_z_residual_and_min_obs():
    rng = np.random.default_rng(0)
    K = 120
    Rm = rng.normal(0, 0.02, K)
    dS = 0.1 - 2.0 * Rm + rng.normal(0, 0.01, K)
    dS[110] = 0.1 - 2.0 * Rm[110] + 0.08             # credit-specific widening
    dS[111] = 0.1 - 2.0 * Rm[111] - 0.08             # tightening
    z = shock_z(dS, Rm)
    assert np.isnan(z[:78]).all()                     # fewer than 78 prior weeks
    assert np.isfinite(z[78])
    assert z[110] > 4 and z[111] < -4
    # by hand at k = 100: OLS over the 100 prior observations (all valid, window 104)
    x, y = Rm[:100], dS[:100]
    b = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    a = y.mean() - b * x.mean()
    sd = np.std(y - a - b * x, ddof=1)
    assert z[100] == pytest.approx((dS[100] - a - b * Rm[100]) / sd)
    # window is 104: at k = 115 observation 0..10 are dropped (k-104 = 11)
    x, y = Rm[11:115], dS[11:115]
    b = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    a = y.mean() - b * x.mean()
    sd = np.std(y - a - b * x, ddof=1)
    assert z[115] == pytest.approx((dS[115] - a - b * Rm[115]) / sd)


def test_shock_z_needs_78_valid_in_window():
    rng = np.random.default_rng(1)
    K = 110
    Rm = rng.normal(0, 0.02, K)
    dS = rng.normal(0, 0.05, K)
    dS[:30] = np.nan                                  # window of k (k <= 134) holds k - 30 valid obs
    z = shock_z(dS, Rm)
    assert np.isnan(z[107])                           # obs 3..106 -> 77 valid
    assert np.isfinite(z[108])                        # obs 4..107 -> 78 valid


# ------------------------------------------------------------------ legs
def test_legs_terciles_and_quintiles():
    rng = np.random.default_rng(2)
    beta = np.arange(90, dtype=float)
    vol = rng.permutation(90).astype(float)
    elig = np.ones(90, dtype=bool)
    low, high = legs_from(beta, vol, elig)
    assert low.size == 18 and high.size == 18         # 3 terciles x floor(30/5) = 6
    for g in range(3):
        cols = np.arange(30 * g, 30 * g + 30)
        o = cols[np.argsort(vol[cols])]
        assert set(low[(low >= 30 * g) & (low < 30 * g + 30)]) == set(o[:6])
        assert set(high[(high >= 30 * g) & (high < 30 * g + 30)]) == set(o[-6:])


def test_legs_skip_small_tercile_and_ineligible():
    beta = np.arange(74, dtype=float)
    vol = np.arange(74, dtype=float)[::-1].copy()
    low, high = legs_from(beta, vol, np.ones(74, dtype=bool))
    # tercile sizes 25 / 25 / 24 (beta order) -> top tercile has floor(24/5) = 4 per leg -> skipped
    assert low.size == 10 and high.size == 10
    assert not np.any(low >= 50) and not np.any(high >= 50)
    # vol descends with beta: LOW vol = highest columns of each tercile
    assert set(low.tolist()) == {20, 21, 22, 23, 24, 45, 46, 47, 48, 49}
    assert set(high.tolist()) == {0, 1, 2, 3, 4, 25, 26, 27, 28, 29}
    l2, h2 = legs_from(beta[:72], vol[:72], np.ones(72, dtype=bool))
    assert l2.size == 0 and h2.size == 0
    vol2 = vol.copy()
    vol2[0] = np.nan
    elig = np.ones(74, dtype=bool)
    elig[1] = False
    l3, h3 = legs_from(beta, vol2, elig)
    assert 0 not in l3 and 0 not in h3 and 1 not in l3 and 1 not in h3


# ------------------------------------------------------------------ full pipeline on hand-made data
def _view(T=900, N=120, seed=0, shock_every=None):
    from lab.framework.data import DataView
    rng = np.random.default_rng(seed)
    start = D2004
    days = np.arange(start, start + int(T * 1.5))
    wd = (days + 3) % 7
    days = days[wd < 5][:T]
    dates = days.astype("datetime64[D]")
    m = rng.normal(0, 0.01, T)
    vols = np.linspace(0.005, 0.04, N)
    betas = rng.uniform(0.3, 1.7, N)
    daily = m[:, None] * betas[None, :] + rng.normal(0, 1, (T, N)) * vols[None, :]
    ret_co = (daily * 0.3).astype(np.float64)
    ret_oc = (1 + daily) / (1 + ret_co) - 1
    close = 50 * np.cumprod(1 + daily, axis=0)
    listed = np.ones((T, N), dtype=bool)
    listed[:400, -5:] = False                         # late listings
    close[~listed] = np.nan
    ret_co[~listed] = 0.0
    ret_oc[~listed] = 0.0
    tradable = listed.copy()
    uni = listed.copy()
    uni[600:, 3] = False                              # leaves the universe
    spread = 1.0 + np.cumsum(rng.normal(0, 0.03, T))
    aaa = 5.0 + np.cumsum(rng.normal(0, 0.02, T))
    baa = aaa + spread
    aaa[:3] = np.nan
    baa[:3] = np.nan
    names = tuple("E%d" % (i + 1) for i in range(N))
    return DataView(
        dates=dates, instruments=names, asset_class=tuple("us_equity" for _ in names),
        ret_co=ret_co, ret_oc=ret_oc, tradable=tradable, listed=listed,
        delisting=np.zeros((T, N), dtype=bool), close=close,
        dollar_volume=np.where(listed, 1e8, np.nan), universe=uni,
        extras={}, series={"fred_moody_yields_1d.DAAA": aaa, "fred_moody_yields_1d.DBAA": baa},
        cash_ret=np.zeros(T),
    )


@pytest.fixture(scope="module")
def view():
    return _view()


@pytest.fixture(scope="module")
def weights(view):
    return target_weights(view, PARAMS)


def test_schedule_and_book(view, weights):
    W = weights
    dec = decision_days(view.dates)
    assert W.shape == (len(view.dates), len(view.instruments))
    # every decision row is decided (no NaN); non-decision rows are NaN except exits
    assert np.isfinite(W[dec]).all()
    active = dec & (np.abs(np.nan_to_num(W)).sum(axis=1) > 0)
    assert active.sum() > 5                           # some shock weeks
    assert active.sum() < dec.sum()                   # and some flat weeks
    # no signal before 78 valid prior weekly observations (the first decision row has t < 5: no dS)
    k_of_row = np.cumsum(dec) - 1
    assert np.all(k_of_row[active] >= 79)
    assert k_of_row[active].min() <= 90               # and the signal does start soon after
    for t in np.flatnonzero(active):
        row = W[t]
        assert row[row > 0].sum() == pytest.approx(0.5)
        assert row[row < 0].sum() == pytest.approx(-0.5)
        assert np.abs(row).sum() <= 1 + 1e-9
        assert np.all(view.universe[t][row != 0])
        # equal weights within each side
        assert np.allclose(row[row > 0], row[row > 0][0])
        assert np.allclose(row[row < 0], row[row < 0][0])


def test_exit_on_universe_leave(view, weights):
    W = weights
    dec = decision_days(view.dates)
    dprev = np.flatnonzero(dec & (np.arange(len(dec)) < 600))[-1]
    if np.nan_to_num(W[dprev, 3]) != 0 and not dec[600]:
        assert W[600, 3] == 0.0
    # never non-zero weight on a non-member decision row
    nz = np.nan_to_num(W) != 0
    assert not np.any(nz & ~view.universe)


def test_point_in_time_truncation(view, weights):
    for cut in (500, 640, 777):
        v = dataclasses.replace(
            view, dates=view.dates[:cut], ret_co=view.ret_co[:cut], ret_oc=view.ret_oc[:cut],
            tradable=view.tradable[:cut], listed=view.listed[:cut], delisting=view.delisting[:cut],
            close=view.close[:cut], dollar_volume=view.dollar_volume[:cut], universe=view.universe[:cut],
            series={k: x[:cut] for k, x in view.series.items()}, cash_ret=view.cash_ret[:cut])
        Wc = target_weights(v, PARAMS)
        assert np.array_equal(np.isnan(Wc), np.isnan(weights[:cut]))
        assert np.allclose(np.nan_to_num(Wc), np.nan_to_num(weights[:cut]))


def test_params_change_output(view, weights):
    dec = decision_days(view.dates)
    on = lambda W: (np.abs(np.nan_to_num(W[dec])).sum(axis=1) > 0).sum()
    W_hi = target_weights(view, {**PARAMS, "z_news": 1.0})
    W_lo = target_weights(view, {**PARAMS, "z_news": 0.5})
    assert on(W_hi) <= on(weights) <= on(W_lo)
    assert on(W_hi) < on(W_lo)
    W_v = target_weights(view, {**PARAMS, "vol_days": 126})
    assert not np.allclose(np.nan_to_num(W_v), np.nan_to_num(weights))


def test_missing_series_raises(view):
    v = dataclasses.replace(view, series={"fred_moody_yields_1d.DAAA": view.series["fred_moody_yields_1d.DAAA"]})
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)

