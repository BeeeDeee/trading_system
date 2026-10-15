import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

NAMES = ("SPY", "QQQ", "IWM", "IEF", "TLT")
SPY, QQQ, IWM, IEF, TLT = range(5)
T = 120


def _dates(n):
    d = np.arange(12000, 12000 + 2 * n, dtype="datetime64[D]")
    d = d[np.is_busday(d)]
    return d[:n]


def make_view(spy=None, tlt=None, spy_close_nan=(), n=T, names=NAMES):
    """SPY alternates +1 % / -1 % (sigma ~ 1 %), TLT +0.1 %/day; overrides by row via dicts."""
    N = len(names)
    dates = _dates(n)
    ret_oc = np.zeros((n, N))
    base = np.where(np.arange(n) % 2 == 0, 0.01, -0.01)
    cols = {nm: i for i, nm in enumerate(names)}
    if "SPY" in cols:
        ret_oc[:, cols["SPY"]] = base
        for t, v in (spy or {}).items():
            if t < n:
                ret_oc[t, cols["SPY"]] = v
    if "TLT" in cols:
        ret_oc[:, cols["TLT"]] = 0.001
        for t, v in (tlt or {}).items():
            if t < n:
                ret_oc[t, cols["TLT"]] = v
    ret_oc[0] = 0.0
    close = 100.0 * np.cumprod(1.0 + ret_oc, axis=0)
    if "SPY" in cols:
        for t in spy_close_nan:
            close[t, cols["SPY"]] = np.nan
    return DataView(
        dates=dates,
        instruments=tuple(names),
        asset_class=tuple("us_etf" for _ in names),
        ret_co=np.zeros((n, N)),
        ret_oc=ret_oc,
        tradable=np.ones((n, N), dtype=bool),
        listed=np.ones((n, N), dtype=bool),
        delisting=np.zeros((n, N), dtype=bool),
        close=close,
        dollar_volume=np.full((n, N), 1e9),
        universe=None,
        extras={},
        series={},
        cash_ret=np.zeros(n),
    )


def month_starts(dates):
    m = dates.astype("datetime64[M]")
    return np.concatenate([[True], m[1:] != m[:-1]])


def is_base(row):
    return row[SPY] == pytest.approx(0.6) and row[IEF] == pytest.approx(0.4) and row[[QQQ, IWM, TLT]].sum() == 0


def is_tilt(row):
    return row[SPY] == pytest.approx(1.0) and row[[QQQ, IWM, IEF, TLT]].sum() == 0


def test_no_event_only_base_on_first_row_and_month_starts():
    d = make_view()
    w = target_weights(d, PARAMS)
    ms = month_starts(d.dates)
    ms[0] = True
    for t in range(T):
        if ms[t]:
            assert is_base(w[t])
        else:
            assert np.all(np.isnan(w[t]))


def test_event_tilts_for_hold_days_then_resets():
    d = make_view(spy={80: -0.03})
    w = target_weights(d, PARAMS)
    for t in range(80, 85):
        assert is_tilt(w[t])
    assert is_base(w[85])
    assert not is_tilt(w[79]) if not np.all(np.isnan(w[79])) else True
    gross = np.nansum(np.abs(w), axis=1)
    assert np.all(gross <= 1.0 + 1e-12)
    assert np.all(np.nan_to_num(w) >= 0)


def test_tlt_down_is_not_an_event():
    d = make_view(spy={80: -0.03}, tlt={80: -0.002})
    w = target_weights(d, PARAMS)
    assert not any(is_tilt(w[t]) for t in range(T) if not np.all(np.isnan(w[t])))
    d0 = make_view(spy={80: -0.03}, tlt={80: 0.0})  # TLT must be strictly positive
    w0 = target_weights(d0, PARAMS)
    assert not any(is_tilt(w0[t]) for t in range(T) if not np.all(np.isnan(w0[t])))


def test_k_changes_output():
    d = make_view(spy={80: -0.018})  # sigma ~ 1 %: event at k = 1.5, not at k = 2.0
    w15 = target_weights(d, dict(PARAMS, k=1.5))
    w20 = target_weights(d, dict(PARAMS, k=2.0))
    assert is_tilt(w15[80])
    assert not any(is_tilt(w20[t]) for t in range(T) if not np.all(np.isnan(w20[t])))


def test_hold_days_changes_output():
    d = make_view(spy={80: -0.03})
    w3 = target_weights(d, dict(PARAMS, hold_days=3))
    for t in range(80, 83):
        assert is_tilt(w3[t])
    assert is_base(w3[83])
    w7 = target_weights(d, dict(PARAMS, hold_days=7))
    for t in range(80, 87):
        assert is_tilt(w7[t])
    assert is_base(w7[87])


def test_windows_merge():
    d = make_view(spy={80: -0.03, 83: -0.03})
    w = target_weights(d, PARAMS)
    for t in range(80, 88):
        assert is_tilt(w[t])
    assert is_base(w[88])


def test_no_event_before_60_returns():
    d = make_view(spy={40: -0.05})
    w = target_weights(d, PARAMS)
    assert not any(is_tilt(w[t]) for t in range(T) if not np.all(np.isnan(w[t])))


def test_day_t_excluded_from_sigma():
    # A huge drop on day t must still count: if day t were in the window it would inflate sigma.
    d = make_view(spy={80: -0.08})
    w = target_weights(d, dict(PARAMS, k=2.0))
    assert is_tilt(w[80])


def test_missing_spy_price_keeps_previous_weights():
    d = make_view(spy={80: -0.03}, spy_close_nan=(85,))
    w = target_weights(d, PARAMS)
    assert np.all(np.isnan(w[85]))
    assert is_base(w[86])  # the tilt-end reset is done at the next row with prices


def test_point_in_time_truncation():
    d = make_view(spy={70: -0.03, 100: -0.04})
    w = target_weights(d, PARAMS)
    for cut in (65, 72, 82, 101):
        dc = make_view(spy={70: -0.03, 100: -0.04}, n=cut)
        wc = target_weights(dc, PARAMS)
        np.testing.assert_array_equal(wc, w[:cut])


def test_missing_instrument_raises():
    d = make_view(names=("SPY", "QQQ", "IWM", "IEF"))
    with pytest.raises(Exception):
        target_weights(d, PARAMS)
