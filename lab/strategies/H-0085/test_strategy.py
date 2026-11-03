import numpy as np
import pytest
from lab.framework.data import DataView
from strategy import target_weights, PARAMS

PFX = "coinmetrics_supply_1d."
BASE = 18231  # day number of a Sunday (weekday arithmetic: (d + 3) % 7 == 6)

BETAS = np.array([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                  1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70])
GROWTH = np.array([0.05, 0.01, 0.30, 0.20, 0.02, 0.10, 0.40,
                   0.03, 0.50, 0.00, 0.15, 0.60, 0.04, 0.25])
EXP_LONG = [1, 4, 7, 9]
EXP_SHORT = [2, 6, 8, 11]


def make_view(T=400, betas=BETAS, growth=GROWTH, seed=1, n_names=None):
    N = len(betas)
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0, 0.03, T)
    r = betas[None, :] * m[:, None] + rng.normal(0.0, 0.002, (T, N))
    close = 100.0 * np.exp(np.cumsum(r, axis=0))
    names = tuple("C%02dUSDT.P" % j for j in range(N))
    days = np.arange(T)
    supply = 1e6 * np.exp(growth[None, :] * days[:, None] / 365.0)
    series = {PFX + names[j]: supply[:, j].copy() for j in range(N)}
    series[PFX + "OTHERUSDT.P"] = np.full(T, 5.0)
    return DataView(
        dates=(days + BASE).astype("datetime64[D]"),
        instruments=names,
        asset_class=tuple("crypto_perp" for _ in names),
        ret_co=np.zeros((T, N)),
        ret_oc=np.exp(r) - 1.0,
        tradable=np.ones((T, N), dtype=bool),
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.ones((T, N)),
        universe=np.ones((T, N), dtype=bool),
        extras={},
        series=series,
        cash_ret=np.zeros(T),
    )


def P(**kw):
    p = dict(PARAMS)
    p.update(kw)
    return p


FIRST = 98  # first Sunday row with row - 1 - 91 >= 0 and 91 returns (lookback 91)


def test_schedule_and_first_formation():
    v = make_view()
    W = target_weights(v, P(lookback_days=91, hold_weeks=4))
    wd = (np.arange(len(v.dates)) + BASE + 3) % 7
    assert np.all(np.isnan(W[wd != 6]))
    assert np.all(np.isfinite(W[wd == 6]))
    assert np.all(W[[0, 7, 91]] == 0.0)
    row = W[FIRST]
    assert np.allclose(row[EXP_LONG], 0.125 / 4)
    assert np.allclose(row[EXP_SHORT], -0.125 / 4)
    others = [j for j in range(14) if j not in EXP_LONG + EXP_SHORT]
    assert np.all(row[others] == 0.0)


def test_overlap_ramp_and_expiry():
    v = make_view()
    W = target_weights(v, P(lookback_days=91, hold_weeks=3))
    g = [np.abs(W[FIRST + 7 * k]).sum() for k in range(6)]
    assert np.allclose(g, [1 / 3, 2 / 3, 1, 1, 1, 1])
    for t in range(len(v.dates)):
        if np.isfinite(W[t, 0]):
            assert np.abs(W[t]).sum() <= 1 + 1e-12
            assert abs(W[t].sum()) < 1e-12  # dollar neutral


def test_odd_count_and_min_eligible():
    # remove coin 0 from supply -> 13 eligible: low half 6 (coins 1..6), high half 7
    v = make_view()
    del v.series[PFX + v.instruments[0]]
    W = target_weights(v, P(lookback_days=91, hold_weeks=1))
    row = W[FIRST]
    # low half coins 1..6, g = .01 .30 .20 .02 .10 .40, k = 2 -> long 1, 4; short 6, 2
    assert set(np.flatnonzero(row > 0)) == {1, 4, 7, 9}
    assert set(np.flatnonzero(row < 0)) == {2, 6, 8, 11}
    assert row[0] == 0.0
    # only 11 eligible -> no cohort
    del v.series[PFX + v.instruments[1]]
    del v.series[PFX + v.instruments[2]]
    W = target_weights(v, P(lookback_days=91, hold_weeks=1))
    assert np.all(W[FIRST] == 0.0)


def test_supply_jump_filter_and_lag():
    v = make_view()
    s = v.series[PFX + v.instruments[1]]
    s[FIRST - 10:] *= np.exp(0.25)          # jump inside the window -> coin 1 ineligible
    v2 = make_view()
    v2.series[PFX + v2.instruments[1]][FIRST - 1] = np.nan   # supply at t-1 missing
    v3 = make_view()
    v3.series[PFX + v3.instruments[1]][FIRST] *= np.exp(0.5)  # value at t itself is not read
    v4 = make_view(T=500)
    v4.series[PFX + v4.instruments[1]][:5] *= np.exp(-0.25)  # jump older than the window
    for view, elig in ((v, False), (v2, False), (v3, True)):
        W = target_weights(view, P(lookback_days=91, hold_weeks=1))
        assert (W[FIRST, 1] > 0) == elig
    W = target_weights(v4, P(lookback_days=91, hold_weeks=1))
    assert W[FIRST, 1] > 0


def test_missing_return_makes_ineligible():
    v = make_view()
    v.close[FIRST - 30, 1] = np.nan
    W = target_weights(v, P(lookback_days=91, hold_weeks=1))
    assert W[FIRST, 1] == 0.0
    # 91 days later the gap is out of the window
    assert W[FIRST + 91, 1] > 0


def test_universe_exit_not_reentered():
    v = make_view()
    H = 4
    t_out = FIRST + 7
    v.universe[t_out, 1] = False
    W = target_weights(v, P(lookback_days=91, hold_weeks=H))
    assert np.isclose(W[FIRST, 1], 0.125 / H)
    # at t_out coin 1 is out of the universe: first cohort drops it, no new cohort includes it
    assert W[t_out, 1] == 0.0
    # back in the universe: only the new cohorts hold it, the first cohort does not re-enter
    assert np.isclose(W[t_out + 7, 1], 0.125 / H)
    assert np.isclose(W[t_out + 14, 1], 2 * 0.125 / H)


def test_tie_break_by_id():
    g = GROWTH.copy()
    g[1] = g[4] = g[0] = 0.01   # coins 0, 1, 4 tie at the lowest growth -> lowest ids 0, 1 long
    v = make_view(growth=g)
    W = target_weights(v, P(lookback_days=91, hold_weeks=1))
    assert set(np.flatnonzero(W[FIRST, :7] > 0)) == {0, 1}


def test_point_in_time_truncation():
    v = make_view(T=420)
    full = target_weights(v, P(lookback_days=91, hold_weeks=3))
    for cut in (FIRST, FIRST + 1, FIRST + 7, 300, 419):
        T = cut + 1
        tv = DataView(
            dates=v.dates[:T], instruments=v.instruments, asset_class=v.asset_class,
            ret_co=v.ret_co[:T], ret_oc=v.ret_oc[:T], tradable=v.tradable[:T],
            listed=v.listed[:T], delisting=v.delisting[:T], close=v.close[:T],
            dollar_volume=v.dollar_volume[:T], universe=v.universe[:T], extras={},
            series={k: x[:T] for k, x in v.series.items()}, cash_ret=v.cash_ret[:T])
        part = target_weights(tv, P(lookback_days=91, hold_weeks=3))
        assert np.array_equal(np.isnan(part), np.isnan(full[:T]))
        assert np.allclose(np.nan_to_num(part), np.nan_to_num(full[:T]))


def test_params_change_output():
    v = make_view(T=500)
    a = target_weights(v, P(lookback_days=91, hold_weeks=26))
    b = target_weights(v, P(lookback_days=182, hold_weeks=26))
    c = target_weights(v, P(lookback_days=91, hold_weeks=13))
    assert not np.allclose(np.nan_to_num(a), np.nan_to_num(b))
    assert not np.allclose(np.nan_to_num(a), np.nan_to_num(c))
    # first formation with lookback 182: first Sunday row >= 183
    assert np.all(np.nan_to_num(b[:189]) == 0.0) and np.abs(b[189]).sum() > 0


def test_missing_or_empty_dataset_raises():
    v = make_view()
    v.series.clear()
    with pytest.raises(KeyError):
        target_weights(v, P(lookback_days=91))
    v = make_view()
    for k in v.series:
        v.series[k][:] = np.nan
    with pytest.raises(ValueError):
        target_weights(v, P(lookback_days=91))

