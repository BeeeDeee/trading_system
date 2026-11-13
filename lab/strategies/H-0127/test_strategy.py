import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

NAMES = ("TLT", "IEF", "SHY", "SHV", "BIL", "VGSH", "SCHO")
NEXT_KEY = "treasury_auctions_1d.next_coupon5_sessions"
LAST_KEY = "treasury_auctions_1d.last_coupon5_sessions"
NAN = np.nan


def make_view(nxt, lst, close=None, tradable=None):
    nxt = np.asarray(nxt, dtype=float)
    lst = np.asarray(lst, dtype=float)
    T = len(nxt)
    N = len(NAMES)
    dates = np.arange(14613, 14613 + T).astype("datetime64[D]")
    if close is None:
        close = np.full((T, N), 100.0)
    if tradable is None:
        tradable = np.ones((T, N), dtype=bool)
    return DataView(
        dates=dates,
        instruments=NAMES,
        asset_class=tuple("us_etf" for _ in NAMES),
        ret_co=np.zeros((T, N)),
        ret_oc=np.zeros((T, N)),
        tradable=np.asarray(tradable, dtype=bool),
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=np.asarray(close, dtype=float),
        dollar_volume=np.full((T, N), 1e9),
        universe=None,
        extras={},
        series={NEXT_KEY: nxt, LAST_KEY: lst},
        cash_ret=np.zeros(T),
    )


def col(name):
    return NAMES.index(name)


def test_pre_window_rows_and_lead():
    # row 2: next auction 4 sessions ahead (h=3), no recent auction -> PRE rows 2,3,4
    nxt = [NAN, NAN, 4, 3, 2, 1, 5, 4, NAN, NAN, NAN, NAN]
    lst = [NAN, NAN, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5]
    w = target_weights(make_view(nxt, lst), {"hold_sessions": 3})
    exp = np.zeros(12)
    exp[2:5] = -0.5
    # row 6: auction day, next is 5 sessions ahead (>=3) -> POST rows 6,7,8
    exp[6:9] = 0.5
    # row 7: next == 4 but last == 1 (< 3) -> no PRE
    assert np.allclose(w[:, col("TLT")], exp)
    assert np.allclose(w[:, col("IEF")], exp)
    for p in ("SHY", "SHV", "BIL", "VGSH", "SCHO"):
        assert np.all(w[:, col(p)] == 0)


def test_cluster_rules():
    # auction on row 3 with next auction 1 session later -> not last of cluster
    # auction on row 4 with next absent -> POST
    # auction on row 8 with next == 2 -> no POST; row 10 auction, next == 3 -> POST
    nxt = [NAN, NAN, NAN, 1, NAN, NAN, NAN, NAN, 2, 1, 3, NAN, NAN, NAN, NAN]
    lst = [5, 6, 7, 0, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 4]
    w = target_weights(make_view(nxt, lst), {"hold_sessions": 2})[:, col("TLT")]
    exp = np.zeros(15)
    exp[4:6] = 0.5
    exp[10:12] = 0.5
    assert np.allclose(w, exp)


def test_pre_absent_last_allowed_and_absent_keys_no_event():
    nxt = [NAN, 4, NAN, NAN, NAN, NAN, NAN, NAN]
    lst = [NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN]
    w = target_weights(make_view(nxt, lst), PARAMS)[:, col("TLT")]
    exp = np.zeros(8)
    exp[1:4] = -0.5
    assert np.allclose(w, exp)
    w0 = target_weights(make_view([NAN] * 8, [NAN] * 8), PARAMS)
    assert np.all(w0 == 0)


def test_overlap_pre_and_post_cancel_and_clip():
    # POST at row 0 (rows 0..2), PRE at row 2 (rows 2..4) with h=3 -> row 2 cancels
    nxt = [5, NAN, 4, NAN, NAN, NAN, NAN, NAN]
    lst = [0, 1, 3, 4, 5, 6, 7, 8]
    w = target_weights(make_view(nxt, lst), {"hold_sessions": 3})[:, col("IEF")]
    assert np.allclose(w, [0.5, 0.5, 0.0, -0.5, -0.5, 0, 0, 0])
    # two POST windows overlapping clip to 0.5
    nxt2 = [NAN, NAN, NAN, NAN, NAN, NAN]
    lst2 = [0, 0, 1, 2, 3, 4]
    w2 = target_weights(make_view(nxt2, lst2), {"hold_sessions": 3})[:, col("IEF")]
    assert np.allclose(w2, [0.5, 0.5, 0.5, 0.5, 0, 0])


def test_missing_price_or_untradable_blocks_window_per_instrument():
    nxt = [NAN, 4, NAN, NAN, NAN, NAN]
    lst = [NAN, 9, 10, 11, 12, 13]
    close = np.full((6, len(NAMES)), 100.0)
    close[1, col("TLT")] = np.nan
    tradable = np.ones((6, len(NAMES)), dtype=bool)
    tradable[1, col("IEF")] = False
    w = target_weights(make_view(nxt, lst, close=close, tradable=tradable), PARAMS)
    assert np.all(w[:, col("TLT")] == 0)
    assert np.all(w[:, col("IEF")] == 0)
    w_ok = target_weights(make_view(nxt, lst), PARAMS)
    assert np.allclose(w_ok[1:4, col("TLT")], -0.5)


def test_hold_param_changes_output_and_lead():
    nxt = [NAN, 3, 4, 5, NAN, NAN, NAN, NAN, NAN, NAN]
    lst = [NAN, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    v = make_view(nxt, lst)
    w2 = target_weights(v, {"hold_sessions": 2})[:, col("TLT")]
    w3 = target_weights(v, {"hold_sessions": 3})[:, col("TLT")]
    w4 = target_weights(v, {"hold_sessions": 4})[:, col("TLT")]
    assert np.allclose(w2, [0, -0.5, -0.5, 0, 0, 0, 0, 0, 0, 0])
    assert np.allclose(w3, [0, 0, -0.5, -0.5, -0.5, 0, 0, 0, 0, 0])
    assert np.allclose(w4, [0, 0, 0, -0.5, -0.5, -0.5, -0.5, 0, 0, 0])


def test_gross_and_point_in_time_truncation():
    rng = np.random.default_rng(0)
    T = 300
    nxt = rng.integers(1, 7, T).astype(float)
    nxt[rng.random(T) < 0.5] = np.nan
    lst = rng.integers(0, 6, T).astype(float)
    full = target_weights(make_view(nxt, lst), PARAMS)
    assert np.all(np.abs(full).sum(axis=1) <= 1.0 + 1e-12)
    for cut in (5, 50, 151, 299):
        part = target_weights(make_view(nxt[:cut], lst[:cut]), PARAMS)
        assert np.allclose(part, full[:cut])


def test_missing_series_raises():
    v = make_view([NAN] * 5, [NAN] * 5)
    del v.series[NEXT_KEY]
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)

