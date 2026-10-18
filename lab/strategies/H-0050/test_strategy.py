from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from strategy import target_weights, PARAMS, vol_events

NAMES = ("SPY", "IVV", "VTI", "DIA", "IEF", "TLT", "SHY", "AGG")
EPOCH_2020 = 18262   # 2020-01-01 in days since 1970-01-01


@dataclass
class FakeView:
    dates: Any
    instruments: Any
    ret_co: Any
    ret_oc: Any
    tradable: Any
    listed: Any
    universe: Any = None

    def col(self, name):
        return self.instruments.index(name)


def make(T=150, big=(50,), small=0.001, size=0.05, missing=()):
    dates = np.arange(EPOCH_2020, EPOCH_2020 + T).astype("datetime64[D]")
    N = len(NAMES)
    roc = np.full((T, N), small)
    for k in big:
        roc[k, 0] = size
    rco = np.zeros((T, N))
    tradable = np.ones((T, N), dtype=bool)
    for m in missing:
        tradable[m, 0] = False
        roc[m, 0] = 0.0
    listed = np.ones((T, N), dtype=bool)
    return FakeView(dates, NAMES, rco, roc, tradable, listed)


def test_event_when_big_day_leaves_20_window():
    # V20 = (19e-6 + 0.0025)/20 -> E = 0.562; V20n ~ 7.3e-6, V60n ~ 4.33e-5 -> En = 0.957; rel = 0.70
    ev = vol_events(make(), 0.1)
    assert list(np.flatnonzero(ev)) == [69]


def test_60_window_dropout_relative_increase_is_small():
    # at t = 109 the big day leaves the 60 window: E = 0.9645 -> En = 1.0, rel = 0.037
    assert list(np.flatnonzero(vol_events(make(), 0.03))) == [69, 109]
    assert list(np.flatnonzero(vol_events(make(), 0.05))) == [69]


def test_no_event_before_60_returns():
    ev = vol_events(make(big=(10,)), 0.1)   # leaves the 20 window at t = 29 < 60 returns
    assert not ev.any()


def test_missing_returns_rule():
    # one missing day in the 20 window (19 of 20 present): event still evaluated
    assert vol_events(make(missing=(60,)), 0.1)[69]
    # three missing days (17 of 20): no event
    assert not vol_events(make(missing=(60, 61, 62)), 0.1).any()


def test_weights_book():
    w = target_weights(make(), PARAMS)
    s, i = NAMES.index("SPY"), NAMES.index("IEF")
    # tilt: decision rows 69, 70 (held open 70 .. close 71), back to 60/40 at row 71
    assert w[69, s] == 1.0 and w[69, i] == 0.0
    assert w[70, s] == 1.0 and w[70, i] == 0.0
    assert w[71, s] == pytest.approx(0.6) and w[71, i] == pytest.approx(0.4)
    # first row and month starts (Feb 1 = row 31, Mar 1 = row 60, Apr 1 = row 91, May 1 = row 121)
    for r in (0, 31, 60, 91, 121):
        assert w[r, s] == pytest.approx(0.6) and w[r, i] == pytest.approx(0.4)
    decided = sorted(set(np.flatnonzero(~np.isnan(w).all(axis=1))))
    assert decided == [0, 31, 60, 69, 70, 71, 91, 121]
    # only SPY and IEF are traded, long only, gross <= 1
    others = [j for j in range(len(NAMES)) if j not in (s, i)]
    dec = ~np.isnan(w[:, 0])
    assert np.all(w[dec][:, others] == 0.0)
    assert np.all(w[dec] >= 0.0)
    assert np.all(np.abs(w[dec]).sum(axis=1) <= 1.0 + 1e-12)


def test_hold_days_param():
    w = target_weights(make(), dict(PARAMS, hold_days=3))
    s = NAMES.index("SPY")
    assert w[69, s] == 1.0 and w[71, s] == 1.0 and w[72, s] == pytest.approx(0.6)
    w1 = target_weights(make(), dict(PARAMS, hold_days=1))
    assert w1[69, s] == 1.0 and w1[70, s] == pytest.approx(0.6)


def test_threshold_param():
    w = target_weights(make(), dict(PARAMS, threshold=0.03))
    s = NAMES.index("SPY")
    assert w[109, s] == 1.0 and w[110, s] == 1.0 and w[111, s] == pytest.approx(0.6)
    w2 = target_weights(make(), PARAMS)
    assert np.isnan(w2[109, s])


def test_new_event_extends_window():
    data = make(big=(50, 51))
    ev = vol_events(data, 0.1)
    # 69, 70: each big day leaves the 20 window; 109: day 50 leaves the 60 window while day 51 keeps
    # V60 high (E 0.686 -> En 0.950, rel 0.385); 110: rel 0.037 (as in the single-day case)
    assert list(np.flatnonzero(ev)) == [69, 70, 109]
    w = target_weights(data, PARAMS)
    s = NAMES.index("SPY")
    assert w[69, s] == 1.0 and w[70, s] == 1.0 and w[71, s] == 1.0
    assert w[72, s] == pytest.approx(0.6)


def test_point_in_time_truncation():
    data = make(big=(50, 90))
    full = target_weights(data, PARAMS)
    for cut in (60, 69, 70, 100, 120):
        d = FakeView(data.dates[:cut], NAMES, data.ret_co[:cut], data.ret_oc[:cut],
                     data.tradable[:cut], data.listed[:cut])
        np.testing.assert_array_equal(target_weights(d, PARAMS), full[:cut])

