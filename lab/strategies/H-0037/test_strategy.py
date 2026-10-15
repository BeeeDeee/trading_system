import numpy as np
import pytest
from dataclasses import dataclass, field
from typing import Any

from strategy import target_weights, PARAMS
from strategy import trailing_median, hold_state

NAMES = ("HYG", "JNK", "AGG", "BND", "TIP", "SCHP", "EEM", "IEMG", "EFA", "IEFA")


@dataclass
class FakeView:
    dates: Any
    instruments: tuple
    ret_co: Any
    ret_oc: Any
    listed: Any
    tradable: Any
    close: Any
    dollar_volume: Any
    universe: Any = None

    def col(self, name):
        return self.instruments.index(name)


def make(T=70):
    N = len(NAMES)
    t = np.arange(T)
    base = 1e-4 * np.where(t % 2 == 0, 1.0, -1.0)
    ret_oc = np.zeros((T, N))
    # A legs wiggle by +-1 bp so that the spread has a non-zero std; B legs flat
    for j in (0, 2, 4, 6, 8):
        ret_oc[:, j] = base
    ret_co = np.zeros((T, N))
    listed = np.ones((T, N), dtype=bool)
    tradable = np.ones((T, N), dtype=bool)
    close = np.full((T, N), 50.0)
    dv = np.full((T, N), 2.0e7)
    return FakeView(np.arange(T).astype("datetime64[D]"), NAMES, ret_co, ret_oc, listed, tradable, close, dv)


def jump(v, name, row, size):
    """A one-day gap of `size` on `name` at `row`, undone at row + 1 (log-exact)."""
    j = v.col(name)
    v.ret_oc[row, j] = (1 + v.ret_oc[row, j]) * (1 + size) - 1
    v.ret_oc[row + 1, j] = (1 + v.ret_oc[row + 1, j]) / (1 + size) - 1


def drop_bar(v, name, row):
    j = v.col(name)
    v.tradable[row, j] = False
    v.close[row, j] = np.nan
    v.dollar_volume[row, j] = np.nan
    v.ret_co[row, j] = 0.0
    v.ret_oc[row, j] = 0.0


def W(v, **kw):
    p = dict(PARAMS)
    p.update(kw)
    return target_weights(v, p)


def test_no_gap_no_position():
    w = W(make())
    assert np.all(w == 0)


def test_rich_A_is_shorted_and_held_five_sessions():
    v = make()
    jump(v, "HYG", 40, 0.01)
    w = W(v)
    h, k = v.col("HYG"), v.col("JNK")
    assert np.all(w[:40] == 0)
    for t in range(40, 45):
        assert w[t, h] == pytest.approx(-0.5) and w[t, k] == pytest.approx(0.5)
    assert np.all(w[45:] == 0)
    assert np.all(np.abs(w).sum(1) <= 1 + 1e-12)
    assert np.allclose(w.sum(1), 0)


def test_cheap_A_is_bought():
    v = make()
    jump(v, "HYG", 40, -0.01)
    w = W(v)
    assert w[40, v.col("HYG")] == pytest.approx(0.5)
    assert w[40, v.col("JNK")] == pytest.approx(-0.5)


def test_two_pairs_share_gross():
    v = make()
    jump(v, "HYG", 40, 0.01)
    jump(v, "EEM", 42, -0.01)
    w = W(v)
    h, k, e, i = v.col("HYG"), v.col("JNK"), v.col("EEM"), v.col("IEMG")
    assert w[41, h] == pytest.approx(-0.5)
    for t in (42, 43, 44):
        assert w[t, h] == pytest.approx(-0.25) and w[t, k] == pytest.approx(0.25)
        assert w[t, e] == pytest.approx(0.25) and w[t, i] == pytest.approx(-0.25)
        assert np.abs(w[t]).sum() == pytest.approx(1.0)
    for t in (45, 46):
        assert w[t, h] == 0 and w[t, e] == pytest.approx(0.5)
    assert np.all(w[47:] == 0)


def test_same_direction_trigger_restarts_clock():
    v = make()
    jump(v, "HYG", 40, 0.01)
    jump(v, "HYG", 43, 0.01)
    w = W(v)
    h = v.col("HYG")
    assert np.allclose(w[40:48, h], -0.5)
    assert np.all(w[48:] == 0)


def test_opposite_trigger_flips():
    v = make()
    jump(v, "HYG", 40, 0.01)
    jump(v, "HYG", 42, -0.01)
    w = W(v)
    h = v.col("HYG")
    assert np.allclose(w[40:42, h], -0.5)
    assert np.allclose(w[42:47, h], 0.5)
    assert np.all(w[47:] == 0)


def test_missing_bar_delays_exit():
    v = make()
    jump(v, "HYG", 40, 0.01)
    drop_bar(v, "JNK", 45)
    w = W(v)
    h = v.col("HYG")
    assert np.allclose(w[40:46, h], -0.5)
    assert np.all(w[46:] == 0)


def test_missing_bar_blocks_trigger():
    v = make()
    jump(v, "HYG", 40, 0.01)
    drop_bar(v, "JNK", 40)
    w = W(v)
    assert np.all(w == 0)


def test_late_listing_needs_L_plus_1_bars():
    # JNK first bar on row 30: with L = 20, row 50 is the first row with 21 bars
    for row, expect in ((49, False), (50, True)):
        v = make()
        k = v.col("JNK")
        v.listed[:30, k] = False
        v.tradable[:30, k] = False
        v.close[:30, k] = np.nan
        v.dollar_volume[:30, k] = np.nan
        jump(v, "HYG", row, 0.01)
        w = W(v, lookback_days=20)
        assert (w[row, k] != 0) == expect
        assert np.all(w[:30, k] == 0)


def test_lookback_param_changes_eligibility():
    v = make()
    k = v.col("JNK")
    v.listed[:30, k] = False
    v.tradable[:30, k] = False
    v.close[:30, k] = np.nan
    v.dollar_volume[:30, k] = np.nan
    jump(v, "HYG", 45, 0.01)
    assert np.all(W(v, lookback_days=20) == 0)
    assert W(v, lookback_days=10)[45, k] == pytest.approx(0.5)


def test_z_entry_param():
    v = make()
    jump(v, "HYG", 40, 0.01)
    assert np.any(W(v, z_entry=2.0) != 0)
    assert np.all(W(v, z_entry=1e6) == 0)


def test_gap_floor_25bps():
    v = make()
    jump(v, "HYG", 40, 0.002)
    assert np.all(W(v) == 0)
    v = make()
    jump(v, "HYG", 40, 0.003)
    assert W(v)[40, v.col("HYG")] == pytest.approx(-0.5)


def test_liquidity_screen():
    v = make()
    v.dollar_volume[:, v.col("JNK")] = 5.0e6
    jump(v, "HYG", 40, 0.01)
    assert np.all(W(v) == 0)


def test_trailing_median():
    x = np.array([[1.0], [np.nan], [3.0], [10.0], [2.0]])
    m = trailing_median(x, 3)
    assert np.isclose(m[0, 0], 1.0)
    assert np.isclose(m[1, 0], 1.0)
    assert np.isclose(m[2, 0], 2.0)
    assert np.isclose(m[3, 0], 6.5)
    assert np.isclose(m[4, 0], 3.0)
    assert np.isnan(trailing_median(np.full((3, 1), np.nan), 2)).all()


def test_hold_state_logic():
    trig = np.zeros(12, dtype=bool)
    d = np.zeros(12)
    both = np.ones(12, dtype=bool)
    trig[1] = True
    d[1] = 0.01
    s = hold_state(trig, d, both)
    assert np.allclose(s[1:6], -1) and s[0] == 0 and np.all(s[6:] == 0)


def test_point_in_time_truncation():
    v = make()
    jump(v, "HYG", 40, 0.01)
    jump(v, "EEM", 50, -0.01)
    full = W(v)
    for cut in (35, 41, 44, 52, 60):
        tv = FakeView(v.dates[:cut], NAMES, v.ret_co[:cut], v.ret_oc[:cut], v.listed[:cut], v.tradable[:cut],
                      v.close[:cut], v.dollar_volume[:cut])
        assert np.allclose(W(tv), full[:cut])


def test_missing_instrument_raises():
    v = make()
    names = tuple(n for n in NAMES if n != "IEFA")
    j = NAMES.index("IEFA")
    keep = [i for i in range(len(NAMES)) if i != j]
    tv = FakeView(v.dates, names, v.ret_co[:, keep], v.ret_oc[:, keep], v.listed[:, keep], v.tradable[:, keep],
                  v.close[:, keep], v.dollar_volume[:, keep])
    with pytest.raises(ValueError):
        W(tv)
