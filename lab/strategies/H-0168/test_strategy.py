import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

NAMES = ("SPY", "QQQ", "IWM", "DIA", "MDY", "IEF", "TLT", "SHY")
EV_ROW = 80


def _dates(T):
    d = np.arange(12000, 12000 + 3 * T).astype("datetime64[D]")
    d = d[np.is_busday(d)]
    return d[:T]


def make_view(T=120, spy=None, ief=None, tradable=None):
    N = len(NAMES)
    ret_oc = np.zeros((T, N))
    # SPY alternates +1 % / -1 %: trailing vol ~ 1 %
    ret_oc[:, 0] = np.where(np.arange(T) % 2 == 0, 0.01, -0.01)
    ret_oc[:, 5] = 0.001
    if spy is not None:
        for t, r in spy.items():
            ret_oc[t, 0] = r
    if ief is not None:
        for t, r in ief.items():
            ret_oc[t, 5] = r
    ret_co = np.zeros((T, N))
    trad = np.ones((T, N), dtype=bool) if tradable is None else tradable
    listed = np.ones((T, N), dtype=bool)
    close = np.full((T, N), 100.0)
    return DataView(
        dates=_dates(T),
        instruments=NAMES,
        asset_class=tuple("us_etf" for _ in NAMES),
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=trad,
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.full((T, N), 1e9),
        universe=None,
        extras={},
        series={},
        cash_ret=np.zeros(T),
    )


def _row(w, t):
    return dict(zip(NAMES, w[t]))


def _month_starts(dates):
    m = dates.astype("datetime64[M]")
    out = np.zeros(len(dates), dtype=bool)
    out[0] = True
    out[1:] = m[1:] != m[:-1]
    return out


def test_no_event_is_benchmark_only():
    v = make_view()
    w = target_weights(v, PARAMS)
    dec = ~np.isnan(w).all(axis=1)
    assert dec[0]
    ms = _month_starts(v.dates)
    assert np.array_equal(dec, ms)
    for t in np.flatnonzero(dec):
        r = _row(w, t)
        assert r["SPY"] == pytest.approx(0.6) and r["IEF"] == pytest.approx(0.4)
        assert abs(w[t]).sum() == pytest.approx(1.0)


def test_event_window_and_exit():
    v = make_view(spy={EV_ROW: -0.03}, ief={EV_ROW: -0.002})
    w = target_weights(v, PARAMS)
    r = _row(w, EV_ROW)
    assert r["SPY"] == pytest.approx(0.68)
    for n in ("QQQ", "IWM", "DIA", "MDY"):
        assert r[n] == pytest.approx(0.08)
    assert r["IEF"] == 0 and r["TLT"] == 0 and r["SHY"] == 0
    assert w[EV_ROW].sum() == pytest.approx(1.0)
    hold = PARAMS["hold_days"]
    end = _row(w, EV_ROW + hold)
    assert end["SPY"] == pytest.approx(0.6) and end["IEF"] == pytest.approx(0.4) and end["QQQ"] == 0
    ms = _month_starts(v.dates)
    for t in range(EV_ROW + 1, EV_ROW + hold):
        assert np.isnan(w[t]).all()  # inside the window: hold (no monthly reset)
    assert not np.isnan(w[EV_ROW + hold]).all()
    _ = ms


def test_bonds_rally_no_event():
    v = make_view(spy={EV_ROW: -0.03}, ief={EV_ROW: 0.002})
    w = target_weights(v, PARAMS)
    assert np.isnan(w[EV_ROW]).all() or _row(w, EV_ROW)["IEF"] == pytest.approx(0.4)


def test_ief_flat_no_event():
    v = make_view(spy={EV_ROW: -0.03}, ief={EV_ROW: 0.0})
    w = target_weights(v, PARAMS)
    assert np.nansum(w[:, 1]) == 0


def test_k_sigma_changes_output():
    v = make_view(spy={EV_ROW: -0.018}, ief={EV_ROW: -0.002})
    w15 = target_weights(v, dict(PARAMS, k_sigma=1.5))
    w20 = target_weights(v, dict(PARAMS, k_sigma=2.0))
    assert _row(w15, EV_ROW)["QQQ"] == pytest.approx(0.08)
    assert np.nansum(w20[:, 1]) == 0


def test_hold_days_changes_exit():
    v = make_view(spy={EV_ROW: -0.03}, ief={EV_ROW: -0.002})
    for h in (3, 5, 8):
        w = target_weights(v, dict(PARAMS, hold_days=h))
        assert _row(w, EV_ROW + h)["IEF"] == pytest.approx(0.4)
        for t in range(EV_ROW + 1, EV_ROW + h):
            assert np.isnan(w[t]).all()


def test_restart_does_not_stack():
    v = make_view(spy={EV_ROW: -0.03, EV_ROW + 3: -0.03}, ief={EV_ROW: -0.002, EV_ROW + 3: -0.002})
    w = target_weights(v, PARAMS)
    hold = PARAMS["hold_days"]
    r = _row(w, EV_ROW + 3)
    assert r["SPY"] == pytest.approx(0.68) and r["QQQ"] == pytest.approx(0.08)
    assert np.isnan(w[EV_ROW + hold]).all()          # old exit cancelled
    assert _row(w, EV_ROW + 3 + hold)["IEF"] == pytest.approx(0.4)


def test_untradable_side_etf_goes_to_spy():
    T = 120
    trad = np.ones((T, len(NAMES)), dtype=bool)
    trad[EV_ROW, 2] = False   # IWM
    v = make_view(T, spy={EV_ROW: -0.03}, ief={EV_ROW: -0.002}, tradable=trad)
    w = target_weights(v, PARAMS)
    r = _row(w, EV_ROW)
    assert r["IWM"] == 0
    assert r["SPY"] == pytest.approx(0.76)
    assert w[EV_ROW].sum() == pytest.approx(1.0)


def test_vol_needs_history():
    # an event-sized drop before 50 valid trailing returns is not an event
    v = make_view(spy={40: -0.05}, ief={40: -0.002})
    w = target_weights(v, PARAMS)
    assert np.nansum(w[:, 1]) == 0


def test_point_in_time_truncation():
    v = make_view(spy={EV_ROW: -0.03}, ief={EV_ROW: -0.002})
    w = target_weights(v, PARAMS)
    cut = EV_ROW + 2
    vt = make_view(cut, spy={EV_ROW: -0.03}, ief={EV_ROW: -0.002})
    wt = target_weights(vt, PARAMS)
    assert np.array_equal(np.isnan(w[:cut]), np.isnan(wt))
    assert np.allclose(np.nan_to_num(w[:cut]), np.nan_to_num(wt))


def test_gross_le_one_long_only():
    v = make_view(spy={EV_ROW: -0.03, 100: -0.04}, ief={EV_ROW: -0.002, 100: -0.01})
    w = target_weights(v, PARAMS)
    g = np.nansum(np.abs(w), axis=1)
    assert (g <= 1.0 + 1e-12).all()
    assert (np.nan_to_num(w) >= 0).all()
    assert np.nansum(w[:, 6]) == 0 and np.nansum(w[:, 7]) == 0
