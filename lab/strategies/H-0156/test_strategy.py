import numpy as np
import pytest
from lab.framework.data import DataView

from strategy import target_weights, PARAMS
from strategy import (EM_ETFS, FX_FOREIGN_PER_USD, FX_USD_PER_FOREIGN, basket_beta,
                      cohort_position, dollar_change, monthly_signal)
from lab.framework.api import period_starts, total_return_index

NAMES = EM_ETFS + ("SPY", "DIA", "QQQ", "IWM", "MDY")


def make_view(T=900, fx_trend=-0.001, em_mult=1.5, late=None, fx_nan=()):
    # Calendar-day dates (day 10957 of the epoch). SPY has a seeded return pattern; EM ETFs return
    # em_mult x SPY. FX: log(foreign per USD) moves by fx_trend per row (negative = dollar weakens).
    dates = (np.arange(T) + 10957).astype("datetime64[D]")
    N = len(NAMES)
    rng = np.random.default_rng(0)
    spy_r = rng.normal(0.0, 0.01, T)
    ret_oc = np.zeros((T, N))
    ret_co = np.zeros((T, N))
    for j, name in enumerate(NAMES):
        ret_oc[:, j] = spy_r * (em_mult if name in EM_ETFS else 1.0)
    close = np.full((T, N), 100.0)
    tradable = np.ones((T, N), dtype=bool)
    listed = np.ones((T, N), dtype=bool)
    if late is not None:
        name, start = late
        j = NAMES.index(name)
        close[:start, j] = np.nan
        ret_oc[:start, j] = 0.0
        tradable[:start + 1, j] = False
        listed[:start, j] = False
    series = {}
    t = np.arange(T, dtype=float)
    for key in FX_FOREIGN_PER_USD:
        series[key] = np.exp(fx_trend * t).astype(float)
    for key in FX_USD_PER_FOREIGN:
        series[key] = np.exp(-fx_trend * t).astype(float)
    for key in fx_nan:
        series[key] = np.full(T, np.nan)
    return DataView(
        dates=dates, instruments=NAMES, asset_class=tuple("us_etf" for _ in NAMES),
        ret_co=ret_co, ret_oc=ret_oc, tradable=tradable, listed=listed,
        delisting=np.zeros((T, N), dtype=bool), close=close, dollar_volume=close * 1e6,
        universe=None, extras={}, series=series, cash_ret=np.zeros(T),
    )


def test_dollar_change_value_and_min_currencies():
    v = make_view(T=200, fx_trend=-0.001)
    D = dollar_change(v, 1)
    assert np.all(np.isnan(D[:21]))
    assert D[21] == pytest.approx(-0.021)
    assert D[150] == pytest.approx(-0.021)
    # 2 currencies missing: 5 of 7 left -> still defined
    v2 = make_view(T=200, fx_nan=FX_USD_PER_FOREIGN[:2])
    assert np.isfinite(dollar_change(v2, 1)[50])
    # 3 missing -> undefined
    v3 = make_view(T=200, fx_nan=FX_USD_PER_FOREIGN)
    assert np.all(np.isnan(dollar_change(v3, 1)))


def test_dollar_change_ffill_and_sign_convention():
    v = make_view(T=100, fx_trend=0.0)
    # dollar strengthens vs JPY only: log(JPY per USD) up by 0.07 at row 50 (with a gap after it)
    jp = np.ones(100)
    jp[50:] = np.exp(0.07)
    jp[51:55] = np.nan
    v.series["fred_macro.DEXJPUS"] = jp
    D = dollar_change(v, 1)
    assert D[53] == pytest.approx(0.01)          # 0.07 / 7, filled through the gap
    # EUR quoted USD per EUR: EUR falling = dollar stronger = positive D
    v2 = make_view(T=100, fx_trend=0.0)
    eu = np.ones(100)
    eu[50:] = np.exp(-0.07)
    v2.series["fred_macro.DEXUSEU"] = eu
    assert dollar_change(v2, 1)[60] == pytest.approx(0.01)


def test_signal_and_cohorts():
    s = monthly_signal(np.array([np.nan, -0.1, 0.2, 0.0]))
    assert list(s) == [0.0, 1.0, -1.0, 0.0]
    p = cohort_position(np.array([1.0, 1.0, 1.0, -1.0, 0.0]), 3)
    assert np.allclose(p, [1 / 3, 2 / 3, 1.0, 1 / 3, 0.0])


def test_beta_exact_clip_and_min_obs():
    v = make_view(T=400, em_mult=1.5)
    tri = np.asarray(total_return_index(v), dtype=float)
    priced = np.isfinite(v.close)
    em = np.array([v.col(n) for n in EM_ETFS])
    spy = v.col("SPY")
    # 5-session returns of a product of (1+1.5 r) are not exactly 1.5x, so allow a small tolerance
    assert basket_beta(tri, priced, 350, em, spy) == pytest.approx(1.5, abs=0.05)
    v3 = make_view(T=400, em_mult=3.0)
    tri3 = np.asarray(total_return_index(v3), dtype=float)
    assert basket_beta(tri3, priced, 350, em, spy) == 2.0
    # fewer than 40 non-overlapping blocks of history -> 1
    assert basket_beta(tri, priced, 150, em, spy) == 1.0


def test_weights_structure_and_gross():
    params = {"lookback_months": 1, "hold_months": 3}
    v = make_view(T=900, fx_trend=-0.001)
    W = target_weights(v, params)
    dec = np.asarray(period_starts(v.dates, "M"), dtype=bool)
    assert np.all(np.isnan(W[~dec]))
    assert np.all(np.isfinite(W[dec]))
    gross = np.abs(W[dec]).sum(axis=1)
    assert np.all(gross <= 1.0 + 1e-9)
    for n in ("DIA", "QQQ", "IWM", "MDY"):
        assert np.all(W[dec, v.col(n)] == 0.0)
    dec_idx = np.flatnonzero(dec)
    # first decision (row 0): D undefined -> flat; then s=+1 each month, p = 1/3, 2/3, 1
    assert np.all(W[dec_idx[0]] == 0.0)
    em = np.array([v.col(n) for n in EM_ETFS])
    spy = v.col("SPY")
    tri = np.asarray(total_return_index(v), dtype=float)
    priced = np.isfinite(v.close)
    for k, p in ((1, 1 / 3), (2, 2 / 3), (3, 1.0), (20, 1.0)):
        t = dec_idx[k]
        b = basket_beta(tri, priced, t, em, spy)
        row = W[t]
        assert np.allclose(row[em], p / (1 + b) / len(em))
        assert row[spy] == pytest.approx(-p * b / (1 + b))
        assert np.abs(row).sum() == pytest.approx(p)
    # dollar strengthening -> short EM, long SPY
    W2 = target_weights(make_view(T=900, fx_trend=0.001), params)
    t = dec_idx[20]
    assert np.all(W2[t, em] < 0) and W2[t, spy] > 0


def test_late_listing_joins_after_21_priced_sessions():
    params = {"lookback_months": 1, "hold_months": 1}
    start = 400
    v = make_view(T=900, late=("INDA", start))
    W = target_weights(v, params)
    j = v.col("INDA")
    dec_idx = np.flatnonzero(np.asarray(period_starts(v.dates, "M"), dtype=bool))
    for t in dec_idx[1:]:
        if t < start + 21:
            assert W[t, j] == 0.0
            assert np.count_nonzero(W[t, :15]) == 14
        else:
            assert W[t, j] > 0
            assert np.count_nonzero(W[t, :15]) == 15


def test_params_change_output():
    v = make_view(T=900, fx_trend=-0.001)
    base = target_weights(v, {"lookback_months": 1, "hold_months": 3})
    other_h = target_weights(v, {"lookback_months": 1, "hold_months": 6})
    other_l = target_weights(v, {"lookback_months": 6, "hold_months": 3})
    assert not np.allclose(np.nan_to_num(base), np.nan_to_num(other_h))
    assert not np.allclose(np.nan_to_num(base), np.nan_to_num(other_l))
    # lookback 6 months: no signal before 126 rows
    dec_idx = np.flatnonzero(np.asarray(period_starts(v.dates, "M"), dtype=bool))
    for t in dec_idx[dec_idx < 126]:
        assert np.all(other_l[t] == 0.0)


def test_primary_params_and_determinism():
    assert PARAMS == {"lookback_months": 12, "hold_months": 12}
    v = make_view(T=900)
    a = target_weights(v, PARAMS)
    b = target_weights(v, PARAMS)
    assert np.array_equal(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9))
