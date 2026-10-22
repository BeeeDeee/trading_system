import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

INSTR = ("VIXY", "SHY", "IEF", "TLT", "AGG", "GLD")
FRESH_VXV = 20.0


def make_view(vix, vxv=None, vixy_listed_from=0, tradable_vixy=None):
    vix = np.asarray(vix, dtype=float)
    T = len(vix)
    if vxv is None:
        # a VIX3M series that prints a new value every session (never stale), level ~20
        vxv = FRESH_VXV + 0.001 * np.arange(T)
    N = len(INSTR)
    dates = (np.arange(T) + 16440).astype("datetime64[D]")
    listed = np.ones((T, N), dtype=bool)
    listed[:vixy_listed_from, 0] = False
    tradable = listed.copy()
    if tradable_vixy is not None:
        tradable[:, 0] = tradable[:, 0] & tradable_vixy
    close = np.where(listed, 20.0, np.nan)
    zeros = np.zeros((T, N))
    return DataView(
        dates=dates,
        instruments=INSTR,
        asset_class=tuple("us_etf" for _ in INSTR),
        ret_co=zeros.copy(),
        ret_oc=zeros.copy(),
        tradable=tradable,
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.where(listed, 1e8, np.nan),
        universe=None,
        extras={},
        series={"fred_macro.VIXCLS": vix,
                "fred_vxv_1d.VXVCLS": np.asarray(vxv, dtype=float)},
        cash_ret=np.zeros(T),
    )


def vixy(w):
    return w[:, INSTR.index("VIXY")]


def check(col, expected):
    assert len(col) == len(expected)
    for a, b in zip(col, expected):
        if b is None:
            assert np.isnan(a)
        else:
            assert a == pytest.approx(b, abs=1e-9)


def test_sizing_linear_never_long_and_band():
    # vxv = 20 every row (fresh, tiny drift), r = vix / vxv
    vxv = np.full(9, 20.0) + 0.0001 * np.arange(9)
    r = np.array([0.70, 1.20, 0.98, 0.95, 0.94, 0.92, 0.90, 0.50, 1.00])
    w = target_weights(make_view(r * vxv, vxv), PARAMS)
    # targets: -0.8, 0, -0.16, -0.4, -0.48, -0.64, -0.8, -0.8, 0
    # band 0.05 against the held weight; rows without a trade are NaN
    check(vixy(w), [-0.8, 0.0, -0.16, -0.4, -0.48, -0.64, -0.8, None, 0.0])
    assert np.all(w[:, 1:] == 0.0)
    assert np.all(np.nan_to_num(vixy(w)) <= 0.0)


def test_small_moves_stay_inside_band_and_target_zero_closes_fully():
    vxv = np.full(6, 20.0) + 0.0001 * np.arange(6)
    r = np.array([0.997, 0.993, 0.99, 0.985, 0.9999, 1.01])
    # targets: -0.024, -0.056, -0.08, -0.12, -0.0008, 0
    # row 4 moves by 0.1192 >= 0.05 -> trades to -0.0008; row 5: target 0 with a position -> close fully
    w = target_weights(make_view(r * vxv, vxv), PARAMS)
    check(vixy(w), [None, -0.056, None, -0.12, -0.0008, 0.0])


def test_staleness_more_than_five_sessions_goes_flat():
    vix = np.array([14.0, 14.2, 14.2, 14.2, 14.2, 14.2, 14.2, 14.1])
    vxv = np.array([20.0, 20.1, 20.1, 20.1, 20.1, 20.1, 20.1, 20.2])
    # both frozen from row 1: age 1..5 on rows 1..5, 6 on row 6 -> flat, fresh again on row 7
    w = target_weights(make_view(vix, vxv), PARAMS)
    check(vixy(w), [-0.8, None, None, None, None, None, 0.0, -0.8])


def test_one_frozen_series_makes_the_ratio_stale():
    vix = 14.0 + 0.01 * np.arange(8)
    vxv = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
    w = target_weights(make_view(vix, vxv), PARAMS)
    # VIX3M printed once (row 0): age 6 on row 5
    check(vixy(w), [-0.8, None, None, None, None, 0.0, None, None])


def test_missing_and_nonpositive_values():
    vix = np.array([np.nan, 14.0, 0.0, 14.0, -1.0, 14.0])
    vxv = 20.0 + 0.01 * np.arange(6)
    w = target_weights(make_view(vix, vxv), PARAMS)
    check(vixy(w), [None, -0.8, 0.0, -0.8, 0.0, -0.8])
    # a NaN print keeps the last value (series arrive forward-filled), age counted from that value
    vix2 = np.array([14.0, np.nan, np.nan, np.nan, np.nan, np.nan, 14.0])
    w2 = target_weights(make_view(vix2, 20.0 + 0.01 * np.arange(7)), PARAMS)
    check(vixy(w2), [-0.8, None, None, None, None, 0.0, -0.8])


def test_late_listing_and_not_tradable_keep_position():
    vix = np.array([14.0, 14.0, 14.01, 22.0, 21.0, 21.01])
    vxv = 20.0 + 0.01 * np.arange(6)
    trad = np.array([True, True, True, False, True, True])
    w = target_weights(make_view(vix, vxv, vixy_listed_from=2, tradable_vixy=trad), PARAMS)
    check(vixy(w), [None, None, -0.8, None, 0.0, None])


def test_each_param_changes_output():
    vxv = np.full(2, 20.0) + 0.001 * np.arange(2)
    r = np.array([0.97, 0.97])
    dv = make_view(r * vxv, vxv)
    base = vixy(target_weights(dv, PARAMS))[0]
    assert base == pytest.approx(-0.24)
    assert vixy(target_weights(dv, {**PARAMS, "k": 1.05}))[0] == pytest.approx(-0.64)
    assert np.isnan(vixy(target_weights(dv, {**PARAMS, "k": 0.95}))[0])     # target 0, nothing held
    assert vixy(target_weights(dv, {**PARAMS, "d": 0.05}))[0] == pytest.approx(-0.48)
    assert vixy(target_weights(dv, {**PARAMS, "d": 0.15}))[0] == pytest.approx(-0.16)


def test_gross_and_point_in_time_truncation():
    rng = np.random.default_rng(0)
    T = 400
    vxv = 20 + rng.normal(0, 1, T).cumsum() * 0.1
    vix = vxv * (0.92 + rng.normal(0, 0.08, T))
    full = target_weights(make_view(vix, vxv), PARAMS)
    assert full.shape == (T, len(INSTR))
    assert np.all(np.nansum(np.abs(full), axis=1) <= 0.8 + 1e-12)
    for cut in (50, 137, 260, 399):
        part = target_weights(make_view(vix[:cut], vxv[:cut]), PARAMS)
        np.testing.assert_array_equal(np.isnan(part), np.isnan(full[:cut]))
        np.testing.assert_allclose(np.nan_to_num(part), np.nan_to_num(full[:cut]))
    # future perturbation
    vix_p = vix.copy()
    vix_p[200:] = rng.uniform(5, 80, T - 200)
    pert = target_weights(make_view(vix_p, vxv), PARAMS)
    np.testing.assert_array_equal(np.isnan(pert[:200]), np.isnan(full[:200]))
    np.testing.assert_allclose(np.nan_to_num(pert[:200]), np.nan_to_num(full[:200]))


def test_missing_series_raises():
    dv = make_view(np.array([15.0, 15.1]))
    del dv.series["fred_vxv_1d.VXVCLS"]
    with pytest.raises(KeyError):
        target_weights(dv, PARAMS)
