import numpy as np
import pytest

from strategy import PARAMS, target_weights, book, breakout_events, fallback_universe, events


class FakeView:
    def __init__(self, close, vol, names, universe=None):
        self.close = np.asarray(close, dtype=np.float64)
        self.dollar_volume = np.asarray(vol, dtype=np.float64)
        self.instruments = tuple(names)
        self.tradable = np.isfinite(self.close)
        self.universe = universe

    def col(self, name):
        return self.instruments.index(name)

    def truncate(self, t):
        u = None if self.universe is None else self.universe[: t + 1]
        return FakeView(self.close[: t + 1], self.dollar_volume[: t + 1], self.instruments, u)


NAMES = ("BTCUSDT.P", "AAAUSDT.P", "BBBUSDT.P", "CCCUSDT.P", "DDDUSDT.P")
P5 = {"hold_days": 3, "range_days": 5}


def make_view(T=130):
    """Wide ranges on rows 0..99, tight 100/100.1 on rows 100..119, breakout on row 120."""
    N = len(NAMES)
    close = np.zeros((T, N))
    vol = np.full((T, N), 1e6)
    wide = np.where(np.arange(100) % 2 == 0, 90.0, 110.0)
    tight = np.where(np.arange(20) % 2 == 0, 100.0, 100.1)
    base = np.concatenate([wide, tight, np.full(T - 120, 100.05)])
    for j in range(N):
        close[:, j] = base
    close[:, 0] = 100.0                       # BTC flat: return 0
    close[120, 1] = 105.0                     # A: up breakout, volume ok
    vol[120, 1] = 2e6
    close[120, 2] = 95.0                      # B: down breakout, volume ok
    vol[120, 2] = 2e6
    close[120, 3] = 105.0                     # C: up breakout, volume below median
    vol[120, 3] = 5e5
    close[:30, 4] = np.nan                    # D: listed late (needs 95 valid closes before 120)
    vol[:30, 4] = np.nan
    close[120, 4] = 105.0
    vol[120, 4] = 2e6
    return FakeView(close, vol, NAMES, np.ones((T, N), dtype=bool))


def test_events_by_hand():
    v = make_view()
    s = breakout_events(v, P5, v.universe)
    nz = np.argwhere(s != 0)
    assert sorted(map(tuple, nz)) == [(120, 1), (120, 2)]
    assert s[120, 1] == 1.0 and s[120, 2] == -1.0
    mask, side = events(v, P5)
    assert np.array_equal(mask, s != 0)
    assert side[120, 1] == 1.0 and side[120, 2] == -1.0


def test_coin_specific_filter():
    v = make_view()
    v.close[120, 0] = 110.0                   # BTC +10 %: A (+5 %) lagged BTC -> no up-event
    s = breakout_events(v, P5, v.universe)
    assert s[120, 1] == 0.0
    assert s[120, 2] == -1.0                  # B fell and lagged BTC: still a down-event
    assert (s[:, 0] == 0).all()               # BTC is never an event


def test_not_quiet_no_event():
    v = make_view()
    v.close[100:120, 1] = np.where(np.arange(20) % 2 == 0, 80.0, 104.0)   # wide range before t
    v.close[120, 1] = 120.0
    s = breakout_events(v, P5, v.universe)
    assert s[120, 1] == 0.0


def test_missing_close_or_universe():
    v = make_view()
    v.dollar_volume[120, 1] = np.nan
    v.universe[120, 2] = False
    s = breakout_events(v, P5, v.universe)
    assert s[120, 1] == 0.0 and s[120, 2] == 0.0


def test_weights_netted_hedge():
    v = make_view()
    w = target_weights(v, P5)
    assert w.shape == v.close.shape
    assert np.allclose(w[:120], 0.0)
    for t in (120, 121, 122):
        assert w[t, 1] == pytest.approx(0.15)
        assert w[t, 2] == pytest.approx(-0.15)
        assert w[t, 0] == pytest.approx(0.0)  # hedges net out
    assert np.allclose(w[123:], 0.0)


def test_single_slot_hedge_and_hold_param():
    v = make_view()
    v.dollar_volume[120, 2] = 1e5             # drop B's event
    w3 = target_weights(v, P5)
    assert w3[120, 1] == pytest.approx(0.15) and w3[120, 0] == pytest.approx(-0.15)
    w5 = target_weights(v, {"hold_days": 5, "range_days": 5})
    assert np.allclose(w5[120:125, 1], 0.15) and np.allclose(w5[125:, 1], 0.0)
    assert np.allclose(w3[123:, 1], 0.0)


def test_range_days_changes_events():
    rng = np.random.default_rng(1)
    T, N = 400, len(NAMES)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, (T, N)), axis=0))
    vol = rng.uniform(1e6, 3e6, (T, N))
    v = FakeView(close, vol, NAMES, np.ones((T, N), dtype=bool))
    a = breakout_events(v, {"hold_days": 3, "range_days": 10}, v.universe)
    b = breakout_events(v, {"hold_days": 3, "range_days": 20}, v.universe)
    assert (a != 0).sum() > 0 and not np.array_equal(a, b)
    assert (a[:100] == 0).all()               # N + 90 closes needed before t


def test_book_mechanics():
    T, N = 10, 4
    uni = np.ones((T, N), dtype=bool)
    side = np.zeros((T, N))
    side[2, 1] = 1.0
    side[3, 1] = -1.0                         # ignored: slot open
    side[5, 1] = -1.0                         # slot closed at row 5 -> new slot
    side[2, 2] = -1.0
    uni[3, 2] = False                         # early exit at row 3
    w = book(side, uni, 0, 3)
    assert np.allclose(w[2:5, 1], 0.15)
    assert np.allclose(w[5:8, 1], -0.15) and w[8, 1] == 0.0
    assert w[2, 2] == pytest.approx(-0.15) and np.allclose(w[3:, 2], 0.0)
    assert w[2, 0] == pytest.approx(0.0) and w[3, 0] == pytest.approx(-0.15)


def test_gross_scaled_to_one():
    T, N = 3, 12
    side = np.zeros((T, N))
    side[0, 1:11] = 1.0
    w = book(side, np.ones((T, N), dtype=bool), 0, 3)
    assert np.abs(w[0]).sum() == pytest.approx(1.0)
    assert w[0, 1] == pytest.approx(0.05) and w[0, 0] == pytest.approx(-0.5)


def test_point_in_time_truncation():
    rng = np.random.default_rng(2)
    T, N = 300, len(NAMES)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, (T, N)), axis=0))
    vol = rng.uniform(1e6, 3e6, (T, N))
    v = FakeView(close, vol, NAMES, None)
    full = target_weights(v, {"hold_days": 3, "range_days": 10})
    for t in (5, 50, 120, 250):
        part = target_weights(v.truncate(t), {"hold_days": 3, "range_days": 10})
        assert np.array_equal(part, full[: t + 1])
    assert (np.abs(full).sum(axis=1) <= 1.0 + 1e-12).all()


def test_fallback_universe():
    close = np.array([[1.0, 1.0, np.nan], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    vol = np.array([[5.0, 1.0, np.nan], [5.0, 1.0, 100.0], [5.0, 1.0, 100.0]])
    u = fallback_universe(close, vol, n=2, days=2)
    assert u[0].tolist() == [True, True, False]
    assert u[1].tolist() == [True, False, True]


def test_params_default():
    assert PARAMS == {"hold_days": 3, "range_days": 20}
