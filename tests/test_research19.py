"""Research 19: signals (F4)."""

import numpy as np
import pytest

from qlab.data.panel import Panel
from qlab.data.synthetic import generate_market
from qlab.data.normalize import normalize_prices
from qlab.data.panel import panel_from_bars
from qlab.features.basic import tr_index
from qlab.research19.signals import (donchian_state, ensemble, realized_vol, rsi_wilder, sma, tom_window)
from qlab.validation.leakage import assert_point_in_time


def col(x):
    return np.asarray(x, dtype=float)[:, None]


def ones(x):
    return np.ones((len(x), 1), dtype=bool)


def test_donchian_hand_example():
    P = col([1, 2, 3, 4, 3.4, 3.6, 2, 5])
    s, st = donchian_state(P, ones(P), 3)
    # t=3: 4 > max(1,2,3)=3 -> enter, stop = mid(2..4) = 3; t=4: stop -> 3.5 and 3.4 < 3.5 -> exit
    # t=7: 5 > max(3.4,3.6,2)... window P[4..6] = 3.4,3.6,2 -> 3.6 -> enter, stop = mid(3.6,2,5) = 3.5
    assert s[:, 0].tolist() == [0, 0, 0, 1, 0, 0, 0, 1]
    np.testing.assert_allclose(st[[3, 7], 0], [3.0, 3.5])
    assert np.isnan(st[[0, 1, 2, 4, 5, 6], 0]).all()


def test_donchian_needs_strict_break_and_stop_never_falls():
    P = col([1, 2, 3, 3])                                  # equals the maximum -> no entry
    assert donchian_state(P, ones(P), 3)[0].sum() == 0
    rng = np.random.default_rng(1)
    P = np.exp(np.cumsum(rng.normal(0.002, 0.02, (600, 5)), axis=0))
    s, st = donchian_state(P, np.ones(P.shape, bool), 20)
    assert s.sum() > 0
    for j in range(5):
        run = s[:, j] == 1
        d = np.diff(st[:, j])
        both = run[1:] & run[:-1]
        assert (d[both] >= -1e-12).all()


def test_donchian_invalid_window_resets():
    P = col(np.arange(1.0, 13.0))
    v = ones(P)
    v[6, 0] = False
    s, st = donchian_state(P, v, 3)
    assert s[5, 0] == 1 and s[6:10, 0].sum() == 0          # flat until 4 valid days after the gap
    assert np.isnan(st[6:10, 0]).all() and s[10, 0] == 1


def test_ensemble_levels():
    P = col(np.arange(1.0, 40.0))
    e = ensemble(P, ones(P), (3, 10))
    assert e[3, 0] == 0.5 and e[-1, 0] == 1.0 and e[0, 0] == 0.0


def test_rsi_hand_table():
    P = col([10, 11, 10, 9, 10, 12])
    r = rsi_wilder(P, ones(P), 2)[:, 0]
    assert np.isnan(r[:2]).all()
    np.testing.assert_allclose(r[2:], [50.0, 25.0, 62.5, 87.5])
    Q = col([1, 2, 3])
    assert rsi_wilder(Q, ones(Q), 2)[2, 0] == 100.0        # AL = 0


def test_rsi_resets_over_gap():
    P = col([10, 11, 10, 9, 10, 12, 11, 12])
    v = ones(P)
    v[4, 0] = False
    r = rsi_wilder(P, v, 2)[:, 0]
    assert np.isnan(r[4:7]).all() and not np.isnan(r[7])


def test_sma_and_vol():
    P = col([1, 2, 3, 4, 5])
    np.testing.assert_allclose(sma(P, ones(P), 3)[:, 0], [np.nan, np.nan, 2, 3, 4])
    v = ones(P)
    v[2, 0] = False
    assert np.isnan(sma(P, v, 3)[:5, 0]).all()
    Q = col([100, 110, 99, 108.9])
    rv = realized_vol(Q, ones(Q), 3, 252)[:, 0]
    r = np.array([0.1, -0.1, 0.1])
    np.testing.assert_allclose(rv[3], r.std(ddof=1) * np.sqrt(252))
    assert np.isnan(rv[:3]).all()


def test_tom_window_weekend_and_holiday():
    d = np.array(["2024-02-29", "2024-03-01", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-26",
                  "2024-03-27", "2024-03-28", "2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04"],
                 dtype="datetime64[D]")
    # Feb 29 = last day of Feb (day -1); Mar 1, 4, 5 = +1..+3; Mar 29 is a holiday so Thu Mar 28 is day -1
    # of March; Apr 1, 2, 3 = +1..+3; Apr 4 is outside the window
    assert tom_window(d).tolist() == [True, True, True, True, False, False, False, True, True, True, True, False]
    assert not tom_window(d[:8])[7]                              # last row of the calendar: month end unknown


@pytest.fixture(scope="module")
def panel():
    m = generate_market(n_assets=20, seed=19, start="2010-01-01", end="2012-12-31")
    return panel_from_bars(normalize_prices(m.prices, m.delistings, m.calendar), m.calendar)


def _vp(p: Panel):
    return tr_index(p), np.asarray(p.listed & p.tradable)


def test_signals_are_point_in_time(panel):
    fns = {"ensemble": lambda p: ensemble(*_vp(p), (5, 20, 60))[:, :5],
           "rsi2": lambda p: rsi_wilder(*_vp(p), 2),
           "vol": lambda p: realized_vol(*_vp(p), 30, 252),
           "sma": lambda p: sma(*_vp(p), 50)}
    for name, fn in fns.items():
        assert_point_in_time(fn, panel, name, min_day=100)
