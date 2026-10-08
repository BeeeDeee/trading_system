import dataclasses
from typing import Any

import numpy as np
import pytest

from strategy import target_weights, PARAMS


@dataclasses.dataclass
class FakeData:
    universe: Any
    tradable: Any
    extras: dict
    instruments: tuple


def make(T=12, N=4, b91=None, b182=None, b365=None, universe=None, tradable=None):
    z = np.zeros((T, N), dtype=np.float32)
    b91 = z.copy() if b91 is None else b91
    b182 = z.copy() if b182 is None else b182
    b365 = z.copy() if b365 is None else b365
    universe = np.ones((T, N), dtype=bool) if universe is None else universe
    tradable = np.ones((T, N), dtype=bool) if tradable is None else tradable
    extras = {"ins_buy_n_91d": b91, "ins_buy_n_182d": b182, "ins_buy_n_365d": b365}
    return FakeData(universe, tradable, extras, tuple("E%d" % i for i in range(N)))


def single_event(row=3, col=0, T=12, N=4):
    b91 = np.zeros((T, N), dtype=np.float32)
    b182 = np.zeros((T, N), dtype=np.float32)
    b91[row:, col] = 1
    b182[row:, col] = 1
    return b91, b182


def test_single_event_hold_window_and_weights():
    b91, b182 = single_event()
    w = target_weights(make(b91=b91, b182=b182), PARAMS)
    H = PARAMS["hold_days"]
    assert np.all(w[:3] == 0)
    for t in range(3, 3 + H):
        assert w[t, 0] == pytest.approx(0.05)
        assert np.allclose(w[t, 1:], -0.05 / 3)
        assert w[t].sum() == pytest.approx(0.0)
    assert np.all(w[3 + H:] == 0)  # leaves after H-th holding day, book flat


def test_no_event_when_quiet_window_not_quiet_or_nan():
    b91, b182 = single_event()
    b182_busy = b182.copy()
    b182[:, 0] = 1  # a purchase in the prior 182 days -> routine, no event
    w = target_weights(make(b91=b91, b182=b182), PARAMS)
    assert np.all(w == 0)
    b182_nan = b182_busy.copy()
    b182_nan[:3, 0] = np.nan
    w = target_weights(make(b91=b91, b182=b182_nan), PARAMS)
    assert np.all(w == 0)
    b91_nan = b91.copy()
    b91_nan[:, 0] = np.nan
    w = target_weights(make(b91=b91_nan, b182=b182_busy), PARAMS)
    assert np.all(w == 0)


def test_quiet_window_param_selects_field():
    b91, b182 = single_event()
    b182[:, 0] = 1  # busy in 182d window, quiet in 91d window
    d = make(b91=b91, b182=b182)
    assert np.all(target_weights(d, PARAMS) == 0)
    w91 = target_weights(d, dict(PARAMS, quiet_window_days=91))
    assert w91[3, 0] == pytest.approx(0.05)
    d365 = make(b91=b91, b182=single_event()[1], b365=np.ones((12, 4), dtype=np.float32))
    assert np.all(target_weights(d365, dict(PARAMS, quiet_window_days=365)) == 0)
    assert target_weights(d365, PARAMS)[3, 0] == pytest.approx(0.05)


def test_hold_days_param():
    b91, b182 = single_event()
    d = make(b91=b91, b182=b182)
    for H in (3, 5, 7):
        w = target_weights(d, dict(PARAMS, hold_days=H))
        assert np.all(w[3:3 + H, 0] > 0)
        assert np.all(w[3 + H:, 0] == 0)


def test_retrigger_no_extension_beyond_latest_event():
    T, N = 14, 4
    b91, b182 = single_event(T=T, N=N)
    # second filing after quiet: quiet field back to 0 on row 5, new filing usable on row 6
    b182[5, 0] = 0
    w = target_weights(make(T=T, N=N, b91=b91, b182=b182), PARAMS)
    H = PARAMS["hold_days"]
    assert np.all(w[3:6 + H, 0] > 0)
    assert np.all(w[6 + H:, 0] == 0)


def test_not_tradable_or_not_member_on_event_day_no_event():
    b91, b182 = single_event()
    tr = np.ones((12, 4), dtype=bool)
    tr[3, 0] = False
    assert np.all(target_weights(make(b91=b91, b182=b182, tradable=tr), PARAMS) == 0)
    un = np.ones((12, 4), dtype=bool)
    un[3, 0] = False
    assert np.all(target_weights(make(b91=b91, b182=b182, universe=un), PARAMS) == 0)


def test_leaves_long_set_when_not_tradable_and_short_leg_excludes_non_tradable():
    b91, b182 = single_event()
    tr = np.ones((12, 4), dtype=bool)
    tr[5, 0] = False
    tr[4, 3] = False
    w = target_weights(make(b91=b91, b182=b182, tradable=tr), PARAMS)
    assert np.all(w[5] == 0)  # long set empty -> flat
    assert w[6, 0] == pytest.approx(0.05)
    assert w[4, 3] == 0
    assert np.allclose(w[4, 1:3], -0.025)


def test_many_events_cap_long_leg_and_gross():
    T, N = 8, 30
    b91 = np.zeros((T, N), dtype=np.float32)
    b182 = np.zeros((T, N), dtype=np.float32)
    b91[2:, :20] = 1
    b182[2:, :20] = 1
    w = target_weights(make(T=T, N=N, b91=b91, b182=b182), PARAMS)
    assert np.allclose(w[2, :20], 0.025)
    assert np.allclose(w[2, 20:], -0.05)
    assert np.abs(w).sum(axis=1).max() <= 1.0 + 1e-12
    assert np.allclose(w.sum(axis=1), 0.0)
    # few events: 5 % per name cap -> long leg 0.1 with two names
    b91 = np.zeros((T, N), dtype=np.float32)
    b91[2:, :2] = 1
    w = target_weights(make(T=T, N=N, b91=b91), PARAMS)
    assert np.allclose(w[2, :2], 0.05)
    assert w[2].sum() == pytest.approx(0.0)


def test_point_in_time_truncation():
    rng = np.random.default_rng(0)
    T, N = 60, 10
    b91 = (rng.random((T, N)) < 0.1).astype(np.float32)
    b182 = (rng.random((T, N)) < 0.5).astype(np.float32)
    d = make(T=T, N=N, b91=b91, b182=b182)
    full = target_weights(d, PARAMS)
    for cut in (10, 30, 45):
        dt = make(T=cut, N=N, b91=b91[:cut], b182=b182[:cut])
        assert np.array_equal(target_weights(dt, PARAMS), full[:cut])


def test_missing_field_raises():
    d = make()
    del d.extras["ins_buy_n_182d"]
    with pytest.raises(KeyError):
        target_weights(d, PARAMS)
