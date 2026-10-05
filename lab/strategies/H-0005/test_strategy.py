import numpy as np
import pytest
from lab.framework.data import DataView
from strategy import target_weights, PARAMS, cap_weights, weekday, regime_signal

KEY = "stablecoin_supply_1d.total_circulating_usd"
MONDAY0 = 17532  # epoch day count of a Monday


def make_data(T=200, N=12, supply=None, vols=None, dv=None, first_listed=None, universe=None, seed=0):
    dates = (np.arange(T) + MONDAY0).astype("datetime64[D]")
    names = tuple("S%02d" % i for i in range(N))
    rng = np.random.default_rng(seed)
    if vols is None:
        vols = np.linspace(0.02, 0.06, N)
    z = rng.standard_normal((T, N)) * vols
    close = 100.0 * np.exp(np.cumsum(z, axis=0))
    ret_oc = np.zeros((T, N))
    ret_oc[1:] = close[1:] / close[:-1] - 1.0
    ret_co = np.zeros((T, N))
    listed = np.ones((T, N), dtype=bool)
    if first_listed is not None:
        for j, r in first_listed.items():
            listed[:r, j] = False
    close = np.where(listed, close, np.nan)
    ret_oc = np.where(listed, ret_oc, 0.0)
    if first_listed is not None:
        for j, r in first_listed.items():
            ret_oc[r, j] = 0.0
    if dv is None:
        dv = (np.arange(N) + 1.0) * 1e6
    dollar_volume = np.where(listed, np.broadcast_to(dv, (T, N)), np.nan)
    tradable = listed.copy()
    delisting = np.zeros((T, N), dtype=bool)
    if supply is None:
        supply = np.linspace(1e9, 2e9, T)
    return DataView(dates, names, tuple("crypto_spot" for _ in names), ret_co, ret_oc, tradable, listed,
                    delisting, close, dollar_volume, universe=universe, series={KEY: supply})


def sundays(T=200):
    return np.flatnonzero(weekday((np.arange(T) + MONDAY0).astype("datetime64[D]")) == 6)


def test_weekday_and_decision_rows():
    d = make_data()
    w = target_weights(d, PARAMS)
    assert w.shape == (200, 12)
    sun = sundays()
    assert sun[0] == 6  # Monday start -> first Sunday is row 6
    assert np.all(np.diff(sun) == 7)
    nan_rows = np.all(np.isnan(w), axis=1)
    assert np.all(~nan_rows[sun])
    other = np.setdiff1d(np.arange(200), sun)
    assert np.all(nan_rows[other])


def test_cash_before_history_and_before_60_bars():
    d = make_data()
    w = target_weights(d, PARAMS)
    sun = sundays()
    # supply history: needs d_t - 1 - 28 >= first day -> t >= 29; bars >= 60 -> t >= 59
    for t in sun:
        if t < 59:
            assert np.all(w[t] == 0.0), t
        else:
            assert w[t].sum() == pytest.approx(1.0), t


def test_top10_by_volume_and_gross():
    d = make_data()
    w = target_weights(d, PARAMS)
    t = sundays()[-1]
    # dollar volume increasing in column index -> columns 0, 1 excluded
    assert np.all(w[t, :2] == 0.0)
    assert np.all(w[t, 2:] > 0.0)
    for t in sundays():
        assert np.nansum(np.abs(w[t])) <= 1.0 + 1e-9
        assert np.all(w[t] >= 0.0)
        assert np.all(w[t] <= 0.25 + 1e-12)


def test_inverse_volatility_weights():
    d = make_data()
    w = target_weights(d, PARAMS)
    t = sundays()[-1]
    close = d.close
    lr = np.diff(np.log(close[t - 60:t + 1]), axis=0)
    sig = lr.std(axis=0, ddof=1)
    inv = 1.0 / sig[2:]
    expected = cap_weights(inv, 0.25)
    assert np.allclose(w[t, 2:], expected)
    # lower vol (lower column index) gets a higher weight in this setup, broadly
    assert w[t, 2] > w[t, 11]


def test_cap_weights_redistribution():
    w = cap_weights(np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 0.25)
    # 10/15 capped at .25, remaining .75 split equally across 5
    assert w[0] == pytest.approx(0.25)
    assert np.allclose(w[1:], 0.15)
    assert w.sum() == pytest.approx(1.0)
    # iterative capping: second name exceeds cap after redistribution
    w = cap_weights(np.array([10.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 0.25)
    assert w[0] == pytest.approx(0.25)
    assert w[1] == pytest.approx(0.25)
    assert np.allclose(w[2:], 0.1)
    # fewer than 4 names: all at the cap, rest in cash
    w = cap_weights(np.array([1.0, 2.0]), 0.25)
    assert np.allclose(w, 0.25)


def test_risk_off_when_supply_shrinks():
    T = 200
    d = make_data(supply=np.linspace(2e9, 1e9, T))
    w = target_weights(d, PARAMS)
    for t in sundays():
        assert np.all(w[t] == 0.0)


def test_supply_uses_t_minus_1_value():
    T = 200
    t = sundays()[-1]
    supply = np.full(T, 1e9)
    supply[t:] = 2e9  # increase observed only on day t itself -> not used at t
    d = make_data(supply=supply)
    w = target_weights(d, PARAMS)
    assert np.all(w[t] == 0.0)
    supply = np.full(T, 1e9)
    supply[t - 1:] = 2e9  # increase on day t-1 -> risk-on at t
    d = make_data(supply=supply)
    w = target_weights(d, PARAMS)
    assert w[t].sum() == pytest.approx(1.0)


def test_lookback_param_changes_regime():
    T = 200
    t = sundays()[-1]
    supply = np.full(T, 1e9)
    # rising until t-1-20, then falling: 14-day growth < 0, 56-day growth > 0
    peak = t - 1 - 20
    supply[:peak + 1] = np.linspace(0.5e9, 1.5e9, peak + 1)
    supply[peak + 1:] = 1.5e9 - np.arange(1, T - peak) * 1e6
    d = make_data(supply=supply)
    w14 = target_weights(d, {"lookback_days": 14})
    w56 = target_weights(d, {"lookback_days": 56})
    assert np.all(w14[t] == 0.0)
    assert w56[t].sum() == pytest.approx(1.0)
    G, valid = regime_signal(d.dates, supply, 14)
    assert valid[t]
    assert G[t] == pytest.approx(np.log(supply[t - 1] / supply[t - 15]))


def test_stale_or_missing_supply():
    T = 200
    t = sundays()[-1]
    supply = np.linspace(1e9, 2e9, T)
    s1 = supply.copy()
    s1[t - 4:] = np.nan  # last value 3 days before t-1: still usable (forward-filled)
    w = target_weights(make_data(supply=s1), PARAMS)
    assert w[t].sum() == pytest.approx(1.0)
    G, valid = regime_signal(make_data(supply=s1).dates, s1, 28)
    assert G[t] == pytest.approx(np.log(supply[t - 5] / supply[t - 29]))
    s2 = supply.copy()
    s2[t - 9:] = np.nan  # last value 8 days before t-1: stale -> cash
    w = target_weights(make_data(supply=s2), PARAMS)
    assert np.all(w[t] == 0.0)
    s3 = supply.copy()
    s3[:150] = np.nan  # short history
    w = target_weights(make_data(supply=s3), PARAMS)
    t_short = [s for s in sundays() if 150 <= s < 150 + 29]
    for s in t_short:
        assert np.all(w[s] == 0.0)


def test_late_listing_needs_60_bars():
    T = 200
    t = sundays()[-1]
    first = t - 58  # 59 bars up to t
    d = make_data(first_listed={11: first})
    w = target_weights(d, PARAMS)
    assert w[t, 11] == 0.0
    assert w[t, 1] > 0.0  # next in line by volume enters
    d = make_data(first_listed={11: t - 59})  # exactly 60 bars
    w = target_weights(d, PARAMS)
    assert w[t, 11] > 0.0
    assert w[t, 1] == 0.0


def test_universe_membership_restricts_holdings():
    T, N = 200, 12
    u = np.ones((T, N), dtype=bool)
    u[:, 5] = False
    d = make_data(universe=u)
    w = target_weights(d, PARAMS)
    t = sundays()[-1]
    assert w[t, 5] == 0.0
    assert w[t, 1] > 0.0


def test_deterministic():
    d = make_data()
    assert np.array_equal(target_weights(d, PARAMS), target_weights(d, PARAMS), equal_nan=True)
