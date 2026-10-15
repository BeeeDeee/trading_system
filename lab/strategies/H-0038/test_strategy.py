import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

NEXT_KEY = "treasury_auctions_1d.next_coupon5_sessions"
LAST_KEY = "treasury_auctions_1d.last_coupon5_sessions"
NAN = np.nan

# Auction cluster on rows 7 and 8 (e.g. 10y then 30y), announced on row 2; previous auction long ago.
NEXT = np.array([NAN, NAN, 5, 4, 3, 2, 1, 1, NAN, NAN, NAN, NAN, NAN, NAN, NAN], dtype=float)
LAST = np.array([20, 21, 22, 23, 24, 25, 26, 0, 0, 1, 2, 3, 4, 5, 6], dtype=float)


def make_view(nxt=NEXT, lst=LAST, close_nan_rows=(), with_series=True):
    T = len(nxt)
    dates = np.arange(T).astype("datetime64[D]") + np.timedelta64(12000, "D")
    instruments = ("TLT", "SHY")
    N = len(instruments)
    close = np.full((T, N), 100.0)
    for r in close_nan_rows:
        close[r, 0] = np.nan
    series = {}
    if with_series:
        series = {NEXT_KEY: np.asarray(nxt, dtype=float), LAST_KEY: np.asarray(lst, dtype=float)}
    return DataView(
        dates=dates,
        instruments=instruments,
        asset_class=("us_etf", "us_etf"),
        ret_co=np.zeros((T, N)),
        ret_oc=np.zeros((T, N)),
        tradable=np.ones((T, N), dtype=bool),
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.full((T, N), 1e9),
        universe=None,
        extras={},
        series=series,
        cash_ret=np.zeros(T),
    )


def tlt(w):
    return w[:, 0]


def test_primary_window_3():
    w = target_weights(make_view(), {"window_days": 3})
    exp = np.array([0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    np.testing.assert_array_equal(tlt(w), exp)
    np.testing.assert_array_equal(w[:, 1], 0.0)          # SHY always flat
    assert PARAMS == {"window_days": 3}


def test_window_2_and_4():
    w2 = target_weights(make_view(), {"window_days": 2})
    np.testing.assert_array_equal(
        tlt(w2), np.array([0, 0, 0, 0, 0, -1, -1, -1, 1, 1, 0, 0, 0, 0, 0], dtype=float))
    w4 = target_weights(make_view(), {"window_days": 4})
    np.testing.assert_array_equal(
        tlt(w4), np.array([0, 0, 0, -1, -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0], dtype=float))


def test_pre_takes_precedence_over_post():
    # auction on row 3 and another two sessions later (row 5): short through the cluster
    nxt = np.array([NAN, 3, 2, 2, 1, NAN, NAN, NAN, NAN], dtype=float)
    lst = np.array([30, 31, 32, 0, 1, 0, 1, 2, 3], dtype=float)
    w = target_weights(make_view(nxt, lst), {"window_days": 3})
    np.testing.assert_array_equal(tlt(w), np.array([0, -1, -1, -1, -1, 1, 1, 1, 0], dtype=float))


def test_nan_series_does_not_fire():
    nxt = np.full(6, NAN)
    lst = np.full(6, NAN)
    w = target_weights(make_view(nxt, lst), PARAMS)
    np.testing.assert_array_equal(w, 0.0)


def test_no_tlt_bar_keeps_position():
    w = target_weights(make_view(close_nan_rows=(9,)), PARAMS)
    assert np.isnan(w[9, 0])
    assert w[8, 0] == 1.0 and w[10, 0] == 1.0
    assert w[9, 1] == 0.0


def test_gross_le_one_and_signs():
    for p in (2, 3, 4):
        w = target_weights(make_view(), {"window_days": p})
        g = np.nansum(np.abs(w), axis=1)
        assert np.all(g <= 1.0 + 1e-12)


def test_missing_series_raises():
    with pytest.raises(KeyError):
        target_weights(make_view(with_series=False), PARAMS)


def test_point_in_time_truncation():
    full = target_weights(make_view(), PARAMS)
    for t in range(1, len(NEXT)):
        part = target_weights(make_view(NEXT[:t], LAST[:t]), PARAMS)
        np.testing.assert_array_equal(part, full[:t])

