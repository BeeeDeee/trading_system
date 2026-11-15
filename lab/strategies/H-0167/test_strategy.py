import numpy as np
import pytest
from lab.framework.data import DataView

from strategy import target_weights, PARAMS

INSTR = ("SPY", "IEF", "EWJ", "EWH", "EWY", "EWT", "EWS", "EWM", "EWA")
COUNTRY = ("EWJ", "EWH", "EWY", "EWT", "EWS", "EWM", "EWA")
MIC = {"EWJ": "XTKS", "EWH": "XHKG", "EWY": "XKRX", "EWT": "XTAI", "EWS": "XSES", "EWM": "XKLS", "EWA": "XASX"}
T = 40


def make_view(k=None, close=None, tradable=None):
    N = len(INSTR)
    dates = (np.arange(T) + 12000).astype("datetime64[D]")
    if close is None:
        close = np.full((T, N), 50.0)
    if tradable is None:
        tradable = np.isfinite(close)
    listed = np.isfinite(close)
    if k is None:
        k = {name: np.full(T, 10.0) for name in COUNTRY}
    series = {"exchange_holidays_ahead_1d." + MIC[n] + ".sessions_to_closure": np.asarray(k[n], dtype=np.float32)
              for n in COUNTRY}
    return DataView(
        dates=dates,
        instruments=INSTR,
        asset_class=tuple("us_etf" for _ in INSTR),
        ret_co=np.zeros((T, N)),
        ret_oc=np.zeros((T, N)),
        tradable=tradable,
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.where(listed, 1e8, np.nan),
        universe=None,
        extras={},
        series=series,
        cash_ret=np.zeros(T),
    )


def col(name):
    return INSTR.index(name)


def base_k():
    return {name: np.full(T, 10.0) for name in COUNTRY}


def runup(k, name, close_row):
    """closure night after row close_row: k = close_row - t for the 10 rows before (capped 10)."""
    t = np.arange(T)
    v = close_row - t
    v = np.where(v < 0, 10.0, np.minimum(v, 10.0))
    k[name] = v.astype(float)


def test_single_runup_weights():
    k = base_k()
    runup(k, "EWJ", 30)       # k=3 at row 27, k=2 at row 28, k=1 at row 29, k=0 at row 30
    w = target_weights(make_view(k), PARAMS)
    j = col("EWJ")
    assert np.allclose(w[27, j], 0.25) and np.allclose(w[28, j], 0.25)
    assert w[26, j] == 0 and w[29, j] == 0 and w[30, j] == 0
    assert np.allclose(w[27, col("SPY")], 0.35)
    assert np.allclose(w[26, col("SPY")], 0.6)
    assert np.allclose(w[:, col("IEF")], 0.4)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert (w >= 0).all()


def test_pre_sessions_and_tilt_params():
    k = base_k()
    runup(k, "EWH", 30)
    j = col("EWH")
    w1 = target_weights(make_view(k), {"pre_sessions": 1, "tilt_weight": 0.25})
    w3 = target_weights(make_view(k), {"pre_sessions": 3, "tilt_weight": 0.15})
    assert w1[27, j] == 0 and np.allclose(w1[28, j], 0.25)
    assert np.allclose(w3[26, j], 0.15) and np.allclose(w3[28, j], 0.15) and w3[25, j] == 0
    assert np.allclose(w3[26, col("SPY")], 0.45)


def test_common_closure_excluded():
    k = base_k()
    for name in ("EWJ", "EWH", "EWY", "EWT"):     # 4 of 7 share the closure night -> excluded
        runup(k, name, 30)
    runup(k, "EWA", 31)                           # different night -> still marked
    w = target_weights(make_view(k), PARAMS)
    for name in ("EWJ", "EWH", "EWY", "EWT"):
        assert w[28, col(name)] == 0
    assert np.allclose(w[28, col("EWA")], 0.25)
    # exactly 3 of 7 share it: not more than half -> marked, w = min(0.25, 0.6/3) = 0.2
    k = base_k()
    for name in ("EWJ", "EWH", "EWY"):
        runup(k, name, 30)
    w = target_weights(make_view(k), PARAMS)
    for name in ("EWJ", "EWH", "EWY"):
        assert np.allclose(w[28, col(name)], 0.2)
    assert np.allclose(w[28, col("SPY")], 0.0)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_missing_k_and_late_listing_not_eligible():
    k = base_k()
    runup(k, "EWJ", 30)
    runup(k, "EWS", 30)
    k["EWJ"][28] = np.nan
    close = np.full((T, len(INSTR)), 50.0)
    close[:10, col("EWS")] = np.nan              # first bar row 10 -> 20 earlier bars only from row 30
    w = target_weights(make_view(k, close=close), PARAMS)
    assert w[28, col("EWJ")] == 0 and np.allclose(w[27, col("EWJ")], 0.25)
    assert w[28, col("EWS")] == 0
    runup(k, "EWS", 35)
    w = target_weights(make_view(k, close=close), PARAMS)
    assert np.allclose(w[33, col("EWS")], 0.25)
    # missing k shrinks the eligible count used by the common-closure rule
    k = base_k()
    for name in ("EWJ", "EWH", "EWY"):
        runup(k, name, 30)
    for name in ("EWT", "EWS"):
        k[name][:] = np.nan
    w = target_weights(make_view(k), PARAMS)     # 3 of 5 eligible share it -> excluded
    assert w[28, col("EWJ")] == 0


def test_not_tradable_not_eligible():
    k = base_k()
    runup(k, "EWM", 30)
    tr = np.ones((T, len(INSTR)), dtype=bool)
    tr[28, col("EWM")] = False
    w = target_weights(make_view(k, tradable=tr), PARAMS)
    assert np.allclose(w[27, col("EWM")], 0.25) and w[28, col("EWM")] == 0


def test_missing_series_raises():
    v = make_view()
    del v.series["exchange_holidays_ahead_1d.XTKS.sessions_to_closure"]
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)


def test_point_in_time_truncation():
    k = base_k()
    runup(k, "EWJ", 30)
    runup(k, "EWT", 25)
    full = target_weights(make_view(k), PARAMS)
    v = make_view(k)
    cut = 26
    short = DataView(
        dates=v.dates[:cut], instruments=v.instruments, asset_class=v.asset_class,
        ret_co=v.ret_co[:cut], ret_oc=v.ret_oc[:cut], tradable=v.tradable[:cut], listed=v.listed[:cut],
        delisting=v.delisting[:cut], close=v.close[:cut], dollar_volume=v.dollar_volume[:cut], universe=None,
        extras={}, series={key: val[:cut] for key, val in v.series.items()}, cash_ret=v.cash_ret[:cut],
    )
    assert np.allclose(target_weights(short, PARAMS), full[:cut])
