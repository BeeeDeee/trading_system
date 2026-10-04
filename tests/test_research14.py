import numpy as np
import pytest

from qlab.research14 import strategy as S
from qlab.research14.setup import epoch_hour, hour_grid, DEV


def _market(T=6000, seed=0):
    rng = np.random.default_rng(seed)
    hours = np.arange(epoch_hour(DEV[0]), epoch_hour(DEV[0]) + T, dtype=np.int64)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, T)))
    real = rng.random(T) > 0.01
    for t in range(1, T):                       # forward-filled hours
        if not real[t]:
            close[t] = close[t - 1]
    return hours, close, real


CFG = S.Config(6, 1.5, 12, "LS")


@pytest.mark.parametrize("name", ["regime", "z", "signals", "sigma"])
def test_point_in_time(name):
    hours, close, real = _market()
    fns = {"regime": lambda h, c, r: S.regime(h, c),
           "z": lambda h, c, r: S.zscore(c, r, 24),
           "signals": lambda h, c, r: S.signals(h, c, r, CFG),
           "sigma": lambda h, c, r: S.sigma(c, r)}
    fn = fns[name]
    full = fn(hours, close, real).astype(float)
    rng = np.random.default_rng(1)
    for t in rng.integers(4000, len(hours) - 2, 15):
        trunc = fn(hours[:t + 1], close[:t + 1], real[:t + 1]).astype(float)
        np.testing.assert_allclose(trunc, full[:t + 1], equal_nan=True)
        c2 = close.copy()
        c2[t + 1:] *= rng.uniform(0.5, 1.5, len(c2) - t - 1)
        r2 = real.copy()
        r2[t + 1:] = rng.random(len(r2) - t - 1) > 0.5
        np.testing.assert_allclose(fn(hours, c2, r2).astype(float)[:t + 1], full[:t + 1], equal_nan=True)


def test_daily_sma_uses_only_closed_days():
    hours = np.arange(epoch_hour(DEV[0]), epoch_hour(DEV[0]) + 24 * 4, dtype=np.int64)
    close = np.arange(len(hours), dtype=float)
    ma = S._sma_of_closed_bars(close, hours % 24 == 23, 2)
    assert np.isnan(ma[46])                      # only one day closed before hour 47
    assert ma[47] == (23 + 47) / 2               # day 2 closes at hour 47 itself
    assert ma[70] == (23 + 47) / 2               # unchanged inside day 3
    assert ma[71] == (47 + 71) / 2


def test_week_close_is_sunday_23_utc():
    h = hour_grid()
    wk = h[h % 168 == S.WEEK_CLOSE_MOD][:3]
    ts = (wk * 3600).astype("datetime64[s]")
    for t in ts:
        assert t.astype(object).weekday() == 6 and t.astype(object).hour == 23


def test_z_needs_real_candles():
    hours, close, real = _market()
    z = S.zscore(close, real, 6)
    t = np.flatnonzero(~real[1000:])[0] + 1000
    assert np.isnan(z[t]) and np.isnan(z[t + 6])
    assert (S.signals(hours, close, real, CFG)[~real] == 0).all()


def test_trade_by_hand_with_costs_and_funding():
    hours = np.arange(epoch_hour(DEV[0]), epoch_hour(DEV[0]) + 30, dtype=np.int64)   # starts 00:00
    close = 100.0 + np.arange(30)
    fund = S.funding_rate(hours, np.where(hours % 8 == 0, 2e-4, np.nan))
    net, tr = S.simulate(close, fund, np.array([5]), np.array([-1], np.int8), 12, len(hours))
    # short entered at close[5]=105, bars 6..17, settlements at hours 8 and 16 (short receives 2 bps each)
    hr = -(close[6:18] / close[5:17] - 1)
    hr[[2, 10]] += 2e-4
    hr[0] -= S.COST
    hr[-1] -= S.COST
    np.testing.assert_allclose(net[6:18], hr)
    assert net[:6].sum() == 0 and net[18:].sum() == 0
    np.testing.assert_allclose(tr[0], np.prod(1 + hr) - 1)


def test_default_funding_before_data():
    hours = np.arange(0, 48, dtype=np.int64)
    f = S.funding_rate(hours, np.where(hours >= 24, np.where(hours % 8 == 0, 5e-4, np.nan), np.nan))
    assert f[0] == f[8] == f[16] == S.FUNDING_DEFAULT and f[24] == 5e-4 and f[1] == 0


def test_no_overlap_and_forced_exit():
    sig = np.zeros(100, np.int8)
    sig[[3, 10, 15, 16, 40, 95]] = 1
    e, s = S.pick_entries(sig, 12, 0, 100)
    assert e.tolist() == [3, 16, 40, 95]
    close = np.linspace(100, 110, 100)
    net, tr = S.simulate(close, np.zeros(100), e, s, 12, 100)
    assert net[96:].sum() > 0 and np.isfinite(tr).all()          # last trade closed at hour 99


def test_random_entries_match_counts_regime_and_overlap():
    hours, close, real = _market(T=8000)
    reg = S.regime(hours, close)
    sig = S.signals(hours, close, real, CFG)
    e, s = S.pick_entries(sig, CFG.H, 3000, 8000)
    ok = np.isfinite(S.zscore(close, real, CFG.L))
    re, rs = S.random_entries(np.random.default_rng(0), reg, real, ok, hours, e, s, CFG.H, 3000, 8000)
    assert (reg[re] == rs).all() and real[re].all()
    assert (np.diff(re) > CFG.H).all()
    month = lambda x: (hours[x] * 3600).astype("datetime64[s]").astype("datetime64[M]")
    for side in (1, -1):
        want = dict(zip(*np.unique(month(e[s == side]), return_counts=True)))
        for m, n in zip(*np.unique(month(re[rs == side]), return_counts=True)):
            assert n <= want[m]
    assert len(re) >= 0.9 * len(e)


def test_positions_cost_and_holding():
    close = np.array([100.0, 101, 102, 101, 100])
    pos = np.array([0.0, 1, 1, 0, 0])
    net = S.simulate_positions(close, np.zeros(5), pos, cost=1e-3)
    np.testing.assert_allclose(net, [0, -1e-3, 102 / 101 - 1, 101 / 102 - 1 - 1e-3, 0])
