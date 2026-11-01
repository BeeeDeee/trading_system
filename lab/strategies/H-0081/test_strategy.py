import numpy as np
import pytest
from dataclasses import dataclass, field
from typing import Any

from strategy import target_weights, PARAMS
import strategy as strat


@dataclass
class Fake:
    """Minimal stand-in for DataView with the fields the strategy reads."""
    instruments: tuple
    close: Any
    ret_co: Any
    ret_oc: Any
    universe: Any
    tradable: Any
    series: dict = field(default_factory=dict)
    dollar_volume: Any = None

    def col(self, name):
        return self.instruments.index(name)


def make(T=120, N=20, seed=0, vix=30.0):
    rng = np.random.default_rng(seed)
    names = ("BTCUSDT.P",) + tuple("C%02dUSDT.P" % i for i in range(N))
    btc = rng.normal(0, 0.03, T)
    betas = np.linspace(0.5, 1.5, N)
    idio = rng.normal(0, 0.02, (T, N))
    r = np.column_stack([btc, betas[None, :] * btc[:, None] + idio])
    ret_co = np.zeros_like(r)
    ret_oc = r.copy()
    close = 100.0 * np.cumprod(1 + r, axis=0)
    uni = np.ones_like(r, dtype=bool)
    v = np.full(T, vix)
    return Fake(names, close, ret_co, ret_oc, uni, uni.copy(), {"fred_macro.VIXCLS": v})


def test_no_cohort_before_beta_warmup_and_gross():
    d = make()
    w = target_weights(d, PARAMS)
    assert w.shape == d.close.shape
    # 30 joint observations need rows 1..30 -> first eligible row is 30
    assert np.all(w[:30] == 0)
    assert np.any(w[30] != 0)
    assert np.all(np.abs(w).sum(axis=1) <= 1 + 1e-9)
    assert np.allclose(w.sum(axis=1), 0)          # dollar-neutral
    assert np.all(w[:, 0] == 0)                   # BTC never traded


def test_legs_and_weights_on_known_row():
    d = make()
    t = 70
    e = strat._residuals(d)
    order = np.argsort(e[t, 1:]) + 1
    long_leg, short_leg, n_leg = strat.cohort_legs(d, PARAMS)
    assert n_leg[t] == 4                          # floor(0.2 * 20)
    assert set(np.nonzero(long_leg[t])[0]) == set(order[:4])
    assert set(np.nonzero(short_leg[t])[0]) == set(order[-4:])
    # beta matches a direct OLS over the 60 days ending t (rows t-59..t inclusive)
    x = d.ret_oc[t - 59:t + 1, 0]
    y = d.ret_oc[t - 59:t + 1, 5]
    b = np.polyfit(x, y, 1)[0]
    assert np.isclose(e[t, 5], d.ret_oc[t, 5] - b * d.ret_oc[t, 0])


def test_hold_days_overlap():
    d = make(T=80)
    # gate only on row 40
    v = np.full(80, 10.0)
    v[40] = 30.0
    d.series["fred_macro.VIXCLS"] = v
    for hold in (2, 3, 5):
        w = target_weights(d, {"hold_days": hold, "vix_min": 25})
        nz = np.nonzero(np.abs(w).sum(axis=1) > 0)[0]
        assert list(nz) == list(range(40, 40 + hold))
        assert np.isclose(np.abs(w[40]).sum(), 1.0 / hold)
        assert np.isclose(w[40][w[40] > 0][0], 1.0 / (2 * hold * 4))


def test_vix_threshold_and_carry_forward():
    d = make(T=80)
    v = np.full(80, np.nan)
    v[35] = 22.0                                  # carried forward afterwards
    d.series["fred_macro.VIXCLS"] = v
    w20 = target_weights(d, {"hold_days": 3, "vix_min": 20})
    w25 = target_weights(d, {"hold_days": 3, "vix_min": 25})
    assert np.all(w20[:35] == 0) and np.any(w20[35] != 0) and np.any(w20[79] != 0)
    assert np.all(w25 == 0)


def test_min_eligible_and_missing_returns():
    d = make(N=16)
    d.close = d.close.copy()
    d.close[50, 1:3] = np.nan                     # two coins miss row 50 -> also row 51 return missing
    long_leg, short_leg, n_leg = strat.cohort_legs(d, PARAMS)
    assert n_leg[49] == 3
    assert n_leg[50] == 0 and n_leg[51] == 0      # 14 eligible < 15
    assert n_leg[52] == 3


def test_universe_exit_closes_position_for_good():
    d = make()
    t = 50
    w = target_weights(d, PARAMS)
    j = int(np.nonzero(w[t] > 0)[0][0])
    uni = d.universe.copy()
    uni[t + 1, j] = False                         # leaves for one row then returns
    d.universe = uni
    w2 = target_weights(d, PARAMS)
    long_leg, short_leg, n_leg = strat.cohort_legs(d, PARAMS)
    assert not long_leg[t + 1, j] and not short_leg[t + 1, j]
    assert w2[t + 1, j] == 0                      # every live cohort drops j at t+1
    # at t+2 only the cohort formed at t+2 can hold j again (older cohorts closed j for good)
    unit = 1.0 / (2 * 3 * n_leg[t + 2])
    assert np.isclose(w2[t + 2, j], unit * (float(long_leg[t + 2, j]) - float(short_leg[t + 2, j])))
    assert w[t + 1, j] != 0                       # without the exit the position was held


def test_point_in_time_truncation():
    d = make(T=120)
    w = target_weights(d, PARAMS)
    cut = 70
    dt = Fake(d.instruments, d.close[:cut], d.ret_co[:cut], d.ret_oc[:cut], d.universe[:cut],
              d.tradable[:cut], {"fred_macro.VIXCLS": d.series["fred_macro.VIXCLS"][:cut]})
    assert np.allclose(target_weights(dt, PARAMS), w[:cut])


def make_dv(T=120, N=40, seed=1):
    """Market without a supplied universe: coin j has dollar volume ~ (j + 1), BTC the largest."""
    d = make(T=T, N=N, seed=seed)
    rng = np.random.default_rng(seed + 7)
    base = np.arange(1, N + 2, dtype=float)
    base[0] = 1e6                                 # BTC
    dv = base[None, :] * (1.0 + 0.01 * rng.random((T, N + 1)))
    d.dollar_volume = dv
    d.universe = None
    return d


def test_computed_universe_top30_by_median_dollar_volume():
    d = make_dv()
    uni = strat._universe(d)
    assert not uni[:29].any()                     # 30-day window not full before row 29
    assert np.all(uni[29:].sum(axis=1) == 30)
    expected = {0} | set(range(41 - 29, 41))      # BTC + the 29 largest coins (cols 12..40)
    assert set(np.nonzero(uni[60])[0]) == expected
    # a gap in one day removes the coin for the 30 rows whose window contains it
    dv = d.dollar_volume.copy()
    dv[50, 40] = np.nan
    d.dollar_volume = dv
    uni2 = strat._universe(d)
    assert uni2[49, 40] and not uni2[50:80, 40].any() and uni2[80, 40]
    assert uni2[50:80, 11].all()                  # next coin moves in
    # a supplied universe is used as is
    d.universe = np.ones_like(dv, dtype=bool)
    assert strat._universe(d).all()


def test_computed_universe_weights_only_on_members_and_point_in_time():
    d = make_dv()
    w = target_weights(d, PARAMS)
    uni = strat._universe(d)
    assert np.any(w != 0)
    assert np.all(w[~uni] == 0)
    assert np.all(w[:, :12] == 0)                 # never-members and BTC
    cut = 75
    dt = Fake(d.instruments, d.close[:cut], d.ret_co[:cut], d.ret_oc[:cut], None, d.tradable[:cut],
              {"fred_macro.VIXCLS": d.series["fred_macro.VIXCLS"][:cut]}, d.dollar_volume[:cut])
    assert np.allclose(target_weights(dt, PARAMS), w[:cut])


def test_missing_inputs_raise():
    d = make()
    d.series = {}
    with pytest.raises(KeyError):
        target_weights(d, PARAMS)
    d = make()
    d.instruments = tuple("X%02d" % i for i in range(len(d.instruments)))
    with pytest.raises(ValueError):
        target_weights(d, PARAMS)
