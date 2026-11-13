import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS, daily_diff, rally_state, book


def _alt(n, a=0.01, start=1.0):
    sign = start * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return a * sign


def test_daily_diff_compounds_co_and_oc():
    co_s = np.array([0.01, 0.0, 0.0])
    oc_s = np.array([0.02, 0.0, 0.0])
    co_i = np.array([0.0, 0.01, 0.0])
    oc_i = np.array([-0.01, 0.0, 0.0])
    valid = np.array([True, True, False])
    d = daily_diff(co_s, oc_s, co_i, oc_i, valid)
    assert d[0] == pytest.approx(1.01 * 1.02 - 1 - (0.99 - 1))
    assert d[1] == pytest.approx(-0.01)
    assert np.isnan(d[2])


def _spike_series():
    d = np.concatenate([_alt(260), np.full(5, 0.01), _alt(10)])
    return d


def test_rally_fires_on_large_5day_outperformance():
    d = _spike_series()
    valid = np.ones(d.shape[0], dtype=bool)
    r = rally_state(d, valid, 1.5)
    # s ~ 0.01 * sqrt(5) = 0.0224, 1.5 s = 0.0335 <= rel5 = 0.05, rel21 > 0
    assert r[264]
    # alternating +-1% never reaches the threshold
    assert not r[:260].any()
    # k = 2.5: 2.5 * 0.0224 = 0.056 > 0.05
    r2 = rally_state(d, valid, 2.5)
    assert not r2[264]


def test_rally_needs_200_valid_values():
    d = np.concatenate([_alt(150), np.full(5, 0.01)])
    valid = np.ones(d.shape[0], dtype=bool)
    assert not rally_state(d, valid, 1.5).any()
    # 260 rows but 100 of them missing -> fewer than 200 valid in the window
    d = _spike_series()
    d[100:200] = np.nan
    assert not rally_state(d, np.isfinite(d), 1.5)[264]


def test_rally_needs_positive_rel21():
    d = _spike_series()
    d[244:260] = -0.01   # the 21-day sum becomes 5*0.01 - 16*0.01 < 0
    valid = np.ones(d.shape[0], dtype=bool)
    assert not rally_state(d, valid, 1.5)[264]


def test_no_rally_without_bar_today():
    d = _spike_series()
    valid = np.ones(d.shape[0], dtype=bool)
    valid[264] = False
    d[264] = np.nan
    assert not rally_state(d, valid, 1.5)[264]


def test_book_tilts_for_five_sessions_and_rebalances_monthly():
    T = 40
    rally = np.zeros(T, dtype=bool)
    rally[10] = True
    valid = np.ones(T, dtype=bool)
    ms = np.zeros(T, dtype=bool)
    ms[0] = True
    ms[30] = True
    w_s, w_i = book(rally, valid, ms, 0.45)
    dec = np.where(np.isfinite(w_s))[0]
    assert list(dec) == [0, 10, 15, 30]
    assert w_s[0] == pytest.approx(0.60) and w_i[0] == pytest.approx(0.40)
    assert w_s[10] == pytest.approx(0.15) and w_i[10] == pytest.approx(0.85)
    assert w_s[15] == pytest.approx(0.60) and w_i[15] == pytest.approx(0.40)
    assert w_s[30] == pytest.approx(0.60)
    # tilt parameter changes the tilted row only
    w_s2, w_i2 = book(rally, valid, ms, 0.3)
    assert w_s2[10] == pytest.approx(0.30) and w_i2[10] == pytest.approx(0.70)
    assert w_s2[0] == pytest.approx(0.60)


def test_book_keeps_weights_without_bar():
    T = 40
    rally = np.zeros(T, dtype=bool)
    rally[10] = True
    valid = np.ones(T, dtype=bool)
    valid[15] = False
    valid[30] = False
    ms = np.zeros(T, dtype=bool)
    ms[0] = True
    ms[30] = True
    w_s, _ = book(rally, valid, ms, 0.45)
    dec = list(np.where(np.isfinite(w_s))[0])
    assert dec == [0, 10, 16, 31]


def test_overlapping_rally_extends_window():
    T = 30
    rally = np.zeros(T, dtype=bool)
    rally[[5, 8]] = True
    valid = np.ones(T, dtype=bool)
    ms = np.zeros(T, dtype=bool)
    w_s, _ = book(rally, valid, ms, 0.45)
    assert list(np.where(np.isfinite(w_s))[0]) == [0, 5, 13]


def _view(T=400, seed=0):
    rng = np.random.default_rng(seed)
    inst = ("SPY", "IEF", "VTI", "DIA", "IWM", "TLT", "SHY", "AGG")
    N = len(inst)
    dates = np.arange(12000, 12000 + T).astype("datetime64[D]")
    ret_co = rng.normal(0, 0.003, (T, N))
    ret_oc = rng.normal(0, 0.008, (T, N))
    close = 100 * np.cumprod(1 + ret_co + ret_oc, axis=0)
    ones = np.ones((T, N), dtype=bool)
    return DataView(dates=dates, instruments=inst, asset_class=("us_etf",) * N,
                    ret_co=ret_co, ret_oc=ret_oc, tradable=ones, listed=ones.copy(),
                    delisting=np.zeros((T, N), dtype=bool), close=close,
                    dollar_volume=np.full((T, N), 1e9), universe=None, extras={}, series={},
                    cash_ret=np.zeros(T))


def test_target_weights_schema():
    data = _view()
    w = target_weights(data, PARAMS)
    assert w.shape == (400, 8)
    dec = np.isfinite(w).all(axis=1)
    assert dec[0]
    assert np.allclose(np.abs(w[dec]).sum(axis=1), 1.0)
    assert (w[dec] >= 0).all()
    others = [i for i in range(8) if i not in (0, 1)]
    assert (w[dec][:, others] == 0).all()
    # rows are either full decisions or all-NaN
    assert (np.isnan(w).all(axis=1) | dec).all()
