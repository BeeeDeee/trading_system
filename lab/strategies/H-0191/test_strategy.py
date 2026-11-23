import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

T = 60
N = 6
NAMES = tuple("E%d" % (i + 1) for i in range(N))


def make_view(close, universe=None, tradable=None, ret_co=None, ret_oc=None):
    close = np.asarray(close, dtype=np.float64)
    T_, N_ = close.shape
    dates = (np.arange(T_) + 10957).astype("datetime64[D]")
    if ret_co is None:
        ret_co = np.zeros((T_, N_))
    if ret_oc is None:
        ret_oc = np.zeros((T_, N_))
    if universe is None:
        universe = np.ones((T_, N_), dtype=bool)
    if tradable is None:
        tradable = np.ones((T_, N_), dtype=bool)
    return DataView(
        dates=dates,
        instruments=NAMES[:N_],
        asset_class=tuple("us_equity" for _ in range(N_)),
        ret_co=np.asarray(ret_co, dtype=np.float64),
        ret_oc=np.asarray(ret_oc, dtype=np.float64),
        tradable=np.asarray(tradable, dtype=bool),
        listed=np.ones((T_, N_), dtype=bool),
        delisting=np.zeros((T_, N_), dtype=bool),
        close=close,
        dollar_volume=np.full((T_, N_), 1e9),
        universe=np.asarray(universe, dtype=bool),
        extras={},
        series={},
        cash_ret=np.zeros(T_),
    )


def flat_close():
    return np.full((T, N), 100.0)


def split(close, t, j, ratio):
    close[t:, j] = close[t:, j] / ratio
    return close


def test_two_for_one_split_slot():
    c = split(flat_close(), 10, 0, 2.0)
    w = target_weights(make_view(c), PARAMS)
    assert w.shape == (T, N)
    assert np.all(w[:10] == 0)
    for t in range(10, 15):
        assert w[t, 0] == pytest.approx(0.25)
        assert np.allclose(w[t, 1:], -0.05)
    assert np.all(w[15:] == 0)


def test_hold_days_param():
    c = split(flat_close(), 10, 0, 2.0)
    w = target_weights(make_view(c), dict(PARAMS, hold_days=3))
    assert np.all(w[10:13, 0] == 0.25)
    assert np.all(w[13:] == 0)


def test_min_ratio_param():
    c = split(flat_close(), 10, 0, 1.5)
    c = split(c, 20, 1, 4.0 / 3.0)
    w = target_weights(make_view(c), PARAMS)          # min_ratio 1.5
    assert w[10, 0] == pytest.approx(0.25) and w[20, 1] == 0
    w = target_weights(make_view(c), dict(PARAMS, min_ratio=2.0))
    assert np.all(w == 0)
    w = target_weights(make_view(c), dict(PARAMS, min_ratio=1.25))
    assert w[10, 0] == pytest.approx(0.25) and w[20, 1] == pytest.approx(0.25)


def test_non_split_factors_ignored():
    c = flat_close()
    c = split(c, 10, 0, 1.8)          # non-standard factor
    c = split(c, 20, 1, 0.5)          # reverse split
    c = split(c, 30, 2, 1.02)         # dividend-like drop
    c = split(c, 40, 3, 2.0 * 1.035)  # outside the 3 % band
    assert np.all(target_weights(make_view(c), PARAMS) == 0)
    c2 = split(flat_close(), 40, 3, 2.0 * 1.025)  # inside the band
    assert target_weights(make_view(c2), PARAMS)[40, 3] == pytest.approx(0.25)


def test_total_return_enters_factor():
    # close_u halves but the stock also lost 4 % in total return: f = 0.5*... -> 2 * 0.96 = 1.92 (in band)
    c = split(flat_close(), 10, 0, 2.0)
    c[10:, 0] *= 0.96
    roc = np.zeros((T, N)); roc[10, 0] = -0.04
    w = target_weights(make_view(c, ret_oc=roc), PARAMS)
    assert w[10, 0] == pytest.approx(0.25)
    # same close_u path but total return says the price really fell 50 %: f = 1, no event
    roc2 = np.zeros((T, N)); roc2[10, 0] = -0.5
    c2 = split(flat_close(), 10, 0, 2.0)
    assert np.all(target_weights(make_view(c2, ret_oc=roc2), PARAMS) == 0)


def test_echo_rule_21_sessions():
    c = split(flat_close(), 10, 0, 2.0)
    c = split(c, 30, 0, 2.0)          # 20 rows later: ignored
    w = target_weights(make_view(c), PARAMS)
    assert np.all(w[30:] == 0)
    c = split(flat_close(), 10, 0, 2.0)
    c = split(c, 31, 0, 2.0)          # 21 rows later: accepted
    w = target_weights(make_view(c), PARAMS)
    assert w[31, 0] == pytest.approx(0.25)


def test_missing_inputs_no_event():
    c = split(flat_close(), 10, 0, 2.0)
    c[9, 0] = np.nan
    assert np.all(target_weights(make_view(c), PARAMS) == 0)
    c = split(flat_close(), 10, 0, 2.0)
    rco = np.zeros((T, N)); rco[10, 0] = np.nan
    assert np.all(target_weights(make_view(c, ret_co=rco), PARAMS) == 0)
    c = split(flat_close(), 10, 0, 2.0)
    uni = np.ones((T, N), bool); uni[10, 0] = False
    assert np.all(target_weights(make_view(c, universe=uni), PARAMS) == 0)
    trd = np.ones((T, N), bool); trd[10, 0] = False
    assert np.all(target_weights(make_view(c, tradable=trd), PARAMS) == 0)


def test_first_row_no_event():
    c = flat_close(); c[0, 0] = 50.0
    assert np.all(target_weights(make_view(c), PARAMS) == 0)


def test_early_close_when_leaving_universe():
    c = split(flat_close(), 10, 0, 2.0)
    uni = np.ones((T, N), bool); uni[12:, 0] = False
    w = target_weights(make_view(c, universe=uni), PARAMS)
    assert np.all(w[10:12, 0] == 0.25)
    assert np.all(w[12:] == 0)
    trd = np.ones((T, N), bool); trd[13, 0] = False
    w = target_weights(make_view(c, tradable=trd), PARAMS)
    assert np.all(w[10:13, 0] == 0.25)
    assert np.all(w[13:] == 0)


def test_hedge_excludes_untradable_and_drops_leavers():
    c = split(flat_close(), 10, 0, 2.0)
    trd = np.ones((T, N), bool); trd[10, 5] = False
    uni = np.ones((T, N), bool); uni[12:, 4] = False
    w = target_weights(make_view(c, universe=uni, tradable=trd), PARAMS)
    assert np.all(w[10:15, 5] == 0)
    assert np.allclose(w[10:12, 1:5], -0.0625)
    assert np.all(w[12:15, 4] == 0)
    assert np.allclose(w[12:15, 1:4], -0.0625)


def test_gross_cap_scales_slots():
    c = flat_close()
    for j in range(3):
        c = split(c, 10, j, 2.0)
    w = target_weights(make_view(c), PARAMS)
    # longs 3 x 0.25, hedges over E4..E6 3 x 0.25 -> gross 1.5 -> scaled by 1/1.5
    assert np.allclose(w[10, :3], 0.25 / 1.5)
    assert np.allclose(w[10, 3:], -0.25 / 1.5)
    assert np.abs(w).sum(axis=1).max() <= 1.0 + 1e-12


def test_overlapping_slots_exclude_open_longs():
    c = split(flat_close(), 10, 0, 2.0)
    c = split(c, 12, 1, 2.0)
    w = target_weights(make_view(c), PARAMS)
    # row 12: slot A hedges E2..E6 at -0.05, slot B long E2, hedges E3..E6 (not E1) at -0.0625
    assert w[12, 0] == pytest.approx(0.25)
    assert w[12, 1] == pytest.approx(0.25 - 0.05)
    assert np.allclose(w[12, 2:], -0.05 - 0.0625)
    # slot A ends at row 15, slot B at 17
    assert w[15, 0] == 0 and w[15, 1] == pytest.approx(0.25)
    assert np.all(w[17:] == 0)


def test_point_in_time_truncation():
    c = split(flat_close(), 10, 0, 2.0)
    c = split(c, 40, 2, 3.0)
    full = target_weights(make_view(c), PARAMS)
    for cut in (11, 12, 25, 41, 43):
        part = target_weights(make_view(c[:cut]), PARAMS)
        assert np.array_equal(part, full[:cut])


def test_entries_on_events_only():
    c = split(flat_close(), 10, 0, 2.0)
    c = split(c, 25, 0, 2.0)          # echo, ignored
    c = split(c, 40, 2, 1.5)
    w = target_weights(make_view(c), PARAMS)
    longs = w > 0
    starts = longs & ~np.vstack([np.zeros((1, N), bool), longs[:-1]])
    assert set(zip(*np.nonzero(starts))) == {(10, 0), (40, 2)}
    w = target_weights(make_view(c), dict(PARAMS, min_ratio=2.0))
    longs = w > 0
    starts = longs & ~np.vstack([np.zeros((1, N), bool), longs[:-1]])
    assert set(zip(*np.nonzero(starts))) == {(10, 0)}
