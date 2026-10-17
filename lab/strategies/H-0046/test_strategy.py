import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS, simulate_breaches, events

NAMES = ("SPY", "IEF", "VTI", "DIA", "IWM", "TLT", "AGG", "SHY")


def make_view(r_spy, r_ief, names=NAMES, start=18000, no_bar=None):
    """Hand-made view: SPY/IEF close-to-close returns put entirely in ret_oc; other instruments flat."""
    T = len(r_spy)
    N = len(names)
    dates = (np.arange(T) + start).astype("datetime64[D]")
    ret_co = np.zeros((T, N))
    ret_oc = np.zeros((T, N))
    ret_oc[:, names.index("SPY")] = r_spy
    if "IEF" in names:
        ret_oc[:, names.index("IEF")] = r_ief
    close = np.full((T, N), 100.0)
    listed = np.ones((T, N), dtype=bool)
    tradable = np.ones((T, N), dtype=bool)
    if no_bar is not None:
        t, name = no_bar
        close[t, names.index(name)] = np.nan
        ret_oc[t, names.index(name)] = 0.0
        tradable[t, names.index(name)] = False
    return DataView(
        dates=dates,
        instruments=tuple(names),
        asset_class=tuple("us_etf" for _ in names),
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=tradable,
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.full((T, N), 1e9),
        universe=None,
        extras={},
        series={},
        cash_ret=np.zeros(T),
    )


# --- simulated rebalancer -------------------------------------------------------------------------------

def test_equity_rally_overweight_and_reset():
    T = 6
    r_s = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    r_b = np.zeros(T)
    d = simulate_breaches(r_s, r_b, np.ones(T, bool), 1.0)
    # w_1 = 0.63 / 1.03 = 0.6117 > 0.61 -> +1; reset, so no later trigger
    assert d.tolist() == [0, 1, 0, 0, 0, 0]
    # with a 2 pp band 0.6117 < 0.62: no trigger
    assert simulate_breaches(r_s, r_b, np.ones(T, bool), 2.0).tolist() == [0] * T


def test_bond_selloff_triggers_equity_overweight():
    r_s = np.zeros(4)
    r_b = np.array([0.0, -0.05, 0.0, 0.0])
    # w_1 = 0.6 / (0.6 + 0.38) = 0.6122
    assert simulate_breaches(r_s, r_b, np.ones(4, bool), 1.0).tolist() == [0, 1, 0, 0]


def test_underweight():
    r_s = np.array([0.0, -0.05, 0.0])
    # w_1 = 0.57 / 0.97 = 0.5876 < 0.59
    assert simulate_breaches(r_s, np.zeros(3), np.ones(3, bool), 1.0).tolist() == [0, -1, 0]


def test_drift_accumulates_until_breach_and_reset_restarts():
    # +2.5 % each day: w_1 = 0.615/1.015 = 0.6059, w_2 = 0.6304/1.0304 = 0.6118 -> trigger on 2, reset,
    # then again on 4
    r_s = np.full(5, 0.025)
    r_s[0] = 0.0
    assert simulate_breaches(r_s, np.zeros(5), np.ones(5, bool), 1.0).tolist() == [0, 0, 1, 0, 1]


def test_no_bar_day_cannot_trigger_but_drifts():
    r_s = np.array([0.0, 0.05, 0.0, 0.0])
    has = np.array([True, False, True, True])
    # day 1 has no bar -> no trigger; drift kept, so day 2 triggers
    assert simulate_breaches(r_s, np.zeros(4), has, 1.0).tolist() == [0, 0, 1, 0]


# --- book -----------------------------------------------------------------------------------------------

def test_weights_tilt_one_session_then_back():
    T = 10
    r_s = np.zeros(T)
    r_s[3] = 0.05
    dv = make_view(r_s, np.zeros(T))
    w = target_weights(dv, PARAMS)
    s, b = dv.col("SPY"), dv.col("IEF")
    assert w.shape == (T, len(NAMES))
    # row 0 establishes the book
    assert w[0, s] == pytest.approx(0.6) and w[0, b] == pytest.approx(0.4)
    # trigger on row 3: SPY = 0.6 - 0.4 = 0.2
    assert w[3, s] == pytest.approx(0.2) and w[3, b] == pytest.approx(0.8)
    # back to 60/40 after one session
    assert w[4, s] == pytest.approx(0.6) and w[4, b] == pytest.approx(0.4)
    # other instruments never held on decision rows
    others = [dv.col(n) for n in NAMES if n not in ("SPY", "IEF")]
    assert np.all(w[[0, 3, 4]][:, others] == 0.0)
    # rows without decision are NaN (keep positions), unless a month start
    dates = dv.dates
    for t in (1, 2, 5, 6, 7, 8, 9):
        month_start = t > 0 and dates[t].astype("datetime64[M]") != dates[t - 1].astype("datetime64[M]")
        if not month_start:
            assert np.all(np.isnan(w[t]))


def test_underweight_tilt_and_param_changes():
    T = 8
    r_s = np.zeros(T)
    r_s[2] = -0.05
    dv = make_view(r_s, np.zeros(T))
    s, b = dv.col("SPY"), dv.col("IEF")
    w = target_weights(dv, PARAMS)
    assert w[2, s] == pytest.approx(1.0) and w[2, b] == pytest.approx(0.0)   # clip(0.6 + 0.4) = 1
    w2 = target_weights(dv, {"threshold_pp": 1.0, "tilt": 0.2})
    assert w2[2, s] == pytest.approx(0.8) and w2[2, b] == pytest.approx(0.2)
    w3 = target_weights(dv, {"threshold_pp": 2.0, "tilt": 0.4})              # 0.5876 > 0.58: no trigger
    assert np.all(np.isnan(w3[2]))
    w4 = target_weights(dv, {"threshold_pp": 0.5, "tilt": 0.4})
    assert w4[2, s] == pytest.approx(1.0)


def test_overweight_big_tilt_clipped_at_zero():
    T = 5
    r_s = np.zeros(T)
    r_s[1] = 0.10
    dv = make_view(r_s, np.zeros(T))
    w = target_weights(dv, {"threshold_pp": 1.0, "tilt": 0.6})
    assert w[1, dv.col("SPY")] == pytest.approx(0.0)
    assert w[1, dv.col("IEF")] == pytest.approx(1.0)


def test_consecutive_triggers_keep_tilting():
    T = 6
    r_s = np.array([0.0, 0.05, 0.05, 0.0, 0.0, 0.0])
    dv = make_view(r_s, np.zeros(T))
    w = target_weights(dv, PARAMS)
    s = dv.col("SPY")
    assert w[1, s] == pytest.approx(0.2)
    assert w[2, s] == pytest.approx(0.2)
    assert w[3, s] == pytest.approx(0.6)


def test_month_start_reset():
    # start 18000 = 2019-04-14; 2019-05-01 is row 17
    T = 25
    dv = make_view(np.zeros(T), np.zeros(T))
    w = target_weights(dv, PARAMS)
    s = dv.col("SPY")
    assert w[17, s] == pytest.approx(0.6)
    assert np.isnan(w[16, s]) and np.isnan(w[18, s])


def test_no_bar_on_ief_blocks_trigger():
    T = 5
    r_s = np.zeros(T)
    r_s[1] = 0.05
    dv = make_view(r_s, np.zeros(T), no_bar=(1, "IEF"))
    w = target_weights(dv, PARAMS)
    s = dv.col("SPY")
    assert np.all(np.isnan(w[1]))
    assert w[2, s] == pytest.approx(0.2)


def test_gross_and_long_only_on_synthetic_like_input():
    rng = np.random.default_rng(0)
    T = 600
    dv = make_view(rng.normal(0, 0.012, T), rng.normal(0, 0.005, T))
    for th in (0.5, 1.0, 2.0):
        for tilt in (0.2, 0.4, 0.6):
            w = target_weights(dv, {"threshold_pp": th, "tilt": tilt})
            dec = ~np.all(np.isnan(w), axis=1)
            assert np.all(w[dec] >= 0.0)
            assert np.all(np.abs(w[dec]).sum(axis=1) <= 1.0 + 1e-12)
            assert np.allclose(w[dec].sum(axis=1), 1.0)


def test_point_in_time_truncation():
    rng = np.random.default_rng(1)
    T = 300
    r_s = rng.normal(0, 0.012, T)
    r_b = rng.normal(0, 0.005, T)
    full = target_weights(make_view(r_s, r_b), PARAMS)
    cut = 170
    part = target_weights(make_view(r_s[:cut], r_b[:cut]), PARAMS)
    assert np.array_equal(np.isnan(full[:cut]), np.isnan(part))
    assert np.allclose(np.nan_to_num(full[:cut]), np.nan_to_num(part))


def test_missing_instrument_raises():
    names = ("SPY", "VTI", "DIA", "IWM", "TLT", "AGG", "SHY")
    dv = make_view(np.zeros(5), None, names=names)
    with pytest.raises(Exception):
        target_weights(dv, PARAMS)


# --- diagnostic -----------------------------------------------------------------------------------------

def test_events_mark_equities_with_side():
    T = 8
    r_s = np.zeros(T)
    r_s[2] = 0.05
    r_s[5] = -0.05
    dv = make_view(r_s, np.zeros(T))
    mask, side = events(dv, PARAMS)
    eq = [dv.col(n) for n in ("SPY", "VTI", "DIA", "IWM")]
    bonds = [dv.col(n) for n in ("IEF", "TLT", "AGG", "SHY")]
    assert mask[:, eq].sum() == 8 and mask[:, bonds].sum() == 0
    assert np.all(mask[2, eq]) and np.all(mask[5, eq])
    assert np.all(side[2, eq] == -1.0) and np.all(side[5, eq] == 1.0)
