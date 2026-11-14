from dataclasses import dataclass

import numpy as np
import pytest

from strategy import PARAMS, target_positions
from strategy import atr_simple, book_positions, calendar, session_bands


# ---------------------------------------------------------------- hand-made band scenario (V=100, sigma=1)

def base(T=30):
    d = {
        "h": np.full(T, 99.7), "l": np.full(T, 99.4), "c": np.full(T, 99.5),
        "V": np.full(T, 100.0), "sig": np.full(T, 1.0), "atr": np.full(T, 0.4),
        "ok": np.ones(T, bool), "real": np.ones(T, bool), "link": np.ones(T, bool),
        "day": np.zeros(T, np.int64), "sto": np.ones(T, bool), "last": np.zeros(T, bool),
    }
    d["link"][0] = False
    d["last"][T - 1] = True
    return d


def setrow(d, i, c, h, l):
    d["c"][i], d["h"][i], d["l"][i] = c, h, l


def scenario(T=30):
    d = base(T)
    for i in (10, 11, 12):                       # excursion: 3 closes < L=99
        setrow(d, i, 98.8, 99.0, 98.7)
    d["l"][11] = 98.4                            # extension: 98.4 <= 99 - 0.5
    setrow(d, 13, 99.2, 99.3, 98.9)              # return r = 13
    setrow(d, 14, 99.5, 99.7, 99.4)              # acceptance close (r+1)
    setrow(d, 15, 99.3, 99.45, 99.2)             # pullback b = 15 (l <= 99.25)
    setrow(d, 16, 99.4, 99.44, 99.3)             # neither trigger nor pullback
    setrow(d, 17, 99.6, 99.7, 99.35)             # trigger s = 17 (c > h_b = 99.45)
    setrow(d, 18, 99.7, 99.8, 99.5)
    setrow(d, 19, 99.9, 100.05, 99.7)            # target V = 100 touched
    return d


def run(d, x=0.5, aggressive=False, opposite=False):
    return book_positions(d["h"], d["l"], d["c"], d["V"], d["sig"], d["atr"], d["ok"], d["real"], d["link"],
                          d["day"], d["sto"], d["last"], x, aggressive, opposite, 288)


def expected(T, a, b, w=0.5):
    e = np.zeros(T)
    e[a:b] = w
    return e


def test_primary_long_trade():
    d = scenario()
    np.testing.assert_allclose(run(d), expected(30, 17, 19))


def test_aggressive_entry_on_return():
    d = scenario()
    np.testing.assert_allclose(run(d, aggressive=True), expected(30, 13, 19))


def test_opposite_band_target_exits_at_session_end():
    d = scenario()
    np.testing.assert_allclose(run(d, opposite=True), expected(30, 17, 29))
    np.testing.assert_allclose(run(d, aggressive=True, opposite=True), expected(30, 13, 29))


def test_excursion_distance_param():
    d = scenario()
    assert np.all(run(d, x=0.75) == 0)            # 98.4 > 99 - 0.75
    np.testing.assert_allclose(run(d, x=0.25), expected(30, 17, 19))


def test_stop_exit():
    d = scenario()
    d["l"][18] = 99.05                           # stop = 99.2 - 0.1 = 99.1
    np.testing.assert_allclose(run(d), expected(30, 17, 18))


def test_short_is_mirror():
    d = scenario()
    m = dict(d)
    m["h"], m["l"], m["c"], m["V"] = 200 - d["l"], 200 - d["h"], 200 - d["c"], 200 - d["V"]
    np.testing.assert_allclose(run(m), -expected(30, 17, 19))
    np.testing.assert_allclose(run(m, aggressive=True), -expected(30, 13, 19))


def test_close_below_L_cancels():
    d = scenario()
    setrow(d, 16, 98.9, 99.1, 98.85)
    assert np.all(run(d) == 0)


def test_gap_breaks_run():
    d = scenario()
    d["real"][11] = False
    d["ok"][11] = False
    assert np.all(run(d) == 0)
    assert np.all(run(d, aggressive=True) == 0)


def test_invalid_band_candles_do_not_count():
    d = scenario()
    d["ok"][:11] = False
    d["ok"][10] = False
    d["ok"][11] = False                          # run reduced to row 12 only
    assert np.all(run(d) == 0)


def test_no_signal_after_2200():
    d = scenario()
    d["sto"][17] = False
    assert np.all(run(d) == 0)


def test_skip_when_close_beyond_target():
    d = scenario()
    d["V"][17] = 99.55                           # target below c_s = 99.6
    assert np.all(run(d) == 0)


def test_pullback_needs_two_acceptance_closes():
    d = scenario()
    setrow(d, 14, 99.3, 99.45, 99.2)             # pullback-like at r+1: not allowed
    setrow(d, 15, 99.4, 99.44, 99.3)             # no pullback at r+2
    setrow(d, 16, 99.6, 99.7, 99.35)             # c > h_14 but 14 is no pullback
    setrow(d, 17, 99.5, 99.7, 99.4)
    assert np.all(run(d) == 0)


def test_pullback_moves_to_newer_candle():
    d = scenario()
    setrow(d, 16, 99.3, 99.40, 99.22)            # newer pullback b = 16
    setrow(d, 17, 99.42, 99.5, 99.35)            # 99.42 > h_16 = 99.40 (but < h_15)
    out = run(d)
    np.testing.assert_allclose(out, expected(30, 17, 19))


def test_trigger_window():
    d = scenario()
    for i in range(16, 22):
        setrow(d, i, 99.4, 99.44, 99.3)
    setrow(d, 22, 99.6, 99.7, 99.35)            # s = 22 > b + 6 = 21: no trade
    for i in range(23, 30):
        setrow(d, i, 99.5, 99.7, 99.4)
    assert np.all(run(d) == 0)
    setrow(d, 21, 99.6, 99.7, 99.35)            # s = 21 = b + 6: trade
    setrow(d, 22, 99.7, 100.1, 99.5)            # target at 22
    np.testing.assert_allclose(run(d), expected(30, 21, 22))


def test_expiry_r_plus_24():
    T = 50
    d = base(T)
    for i in (10, 11, 12):
        setrow(d, i, 98.8, 99.0, 98.7)
    d["l"][11] = 98.4
    setrow(d, 13, 99.2, 99.3, 98.9)
    for i in range(14, T):
        setrow(d, i, 99.4, 99.44, 99.3)          # accepted, no pullback, no trigger
    setrow(d, 36, 99.3, 99.45, 99.2)             # pullback
    e = d["c"].copy(), d["h"].copy(), d["l"].copy()
    setrow(d, 37, 99.6, 99.7, 99.35)             # s = 37 = r + 24: allowed
    out = run(d)
    assert out[37] == 0.5 and out[36] == 0
    d["c"], d["h"], d["l"] = e
    setrow(d, 38, 99.6, 99.7, 99.35)             # s = 38 > r + 24: expired
    assert np.all(run(d) == 0)


def test_session_end_cancels_armed_setup():
    d = scenario()
    d["day"][16:] = 1
    d["link"][16] = False
    d["last"][15] = True
    assert np.all(run(d) == 0)


def test_position_blocks_new_setups():
    d = scenario(T=60)
    d["atr"][:] = 4.0                            # first trade (aggressive, s=13): stop 98.9 - 1 = 97.9
    d["h"][19] = 99.9                            # no target touch at 19
    for i in (20, 21, 22):                       # second long excursion while the first trade is open
        setrow(d, i, 98.8, 99.0, 98.4)
    d["V"][23] = 100.5                           # the second setup would target 100.5
    setrow(d, 23, 99.6, 99.7, 99.0)              # return at 23 (c >= L = 99.5)
    setrow(d, 25, 99.9, 100.05, 99.7)            # first trade's target 100 touched at 25
    out = run(d, aggressive=True)
    np.testing.assert_allclose(out, expected(60, 13, 25))   # the signal at 23 was ignored
    d["h"][25] = 99.9                            # without the touch at 25 the first trade runs to session end
    np.testing.assert_allclose(run(d, aggressive=True), expected(60, 13, 59))


# ---------------------------------------------------------------- bands, ATR, calendar

def test_session_vwap_and_sigma():
    tp = np.array([10.0, 20.0, 30.0, 50.0, 70.0])
    w = np.array([1.0, 1.0, 2.0, 1.0, 3.0])
    qv = tp * w
    day = np.array([0, 0, 0, 1, 1])
    V, S = session_bands(tp, tp, tp, qv, np.ones(5, bool), day)
    np.testing.assert_allclose(V[:3], [10.0, 15.0, 22.5])
    np.testing.assert_allclose(S[2], np.sqrt(68.75))
    np.testing.assert_allclose(V[3:], [50.0, 65.0])       # reset at the new session
    np.testing.assert_allclose(S[3], 0.0)


def test_gap_candle_has_no_weight():
    tp = np.array([10.0, 20.0, 30.0])
    real = np.array([True, False, True])
    V, _ = session_bands(tp, tp, tp, tp, real, np.zeros(3, np.int64))
    np.testing.assert_allclose(V, [10.0, 10.0, 20.0])


def test_atr_simple_mean():
    T = 20
    c = np.full(T, 100.0)
    h = np.full(T, 101.0)
    l = np.full(T, 99.0)
    h[15] = 104.0                                 # TR 5 instead of 2
    a = atr_simple(h, l, c, 14)
    assert np.all(np.isnan(a[:14]))
    np.testing.assert_allclose(a[14], 2.0)
    np.testing.assert_allclose(a[15], 2.0 + 3.0 / 14)


def test_calendar():
    minutes = np.array([0, 110, 115, 1315, 1320, 1435, 1440])
    times = (minutes + 1440 * 30).astype("datetime64[m]")
    day, valid, sto, last, link = calendar(times, 288)
    assert list(valid) == [False, False, True, True, True, True, False]
    assert list(sto) == [True, True, True, True, False, False, True]
    assert list(last) == [False, False, False, False, False, True, False]
    assert day[6] == day[0] + 1
    assert not link[6] and not link[1]
    t2 = (np.array([120, 125, 135]) + 1440 * 30).astype("datetime64[m]")
    assert list(calendar(t2, 288)[4]) == [False, True, False]   # 10-minute hole breaks the link


# ---------------------------------------------------------------- end to end on a fake view

@dataclass
class FakeView:
    times: np.ndarray
    instruments: tuple
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    qv: np.ndarray
    real: np.ndarray
    tradable: np.ndarray
    universe: object
    bars_per_day: int

    def col(self, name):
        return self.instruments.index(name)


def make_view(T=288 * 4, seed=1):
    rng = np.random.default_rng(seed)
    names = ("BTCUSDT", "ETHUSDT")
    r = rng.standard_normal((T, 2)) * 0.002
    c = 100.0 * np.exp(np.cumsum(r, axis=0))
    o = np.vstack([c[:1], c[:-1]])
    h = np.maximum(o, c) * (1 + np.abs(rng.standard_normal((T, 2))) * 0.001)
    l = np.minimum(o, c) * (1 - np.abs(rng.standard_normal((T, 2))) * 0.001)
    qv = rng.uniform(1e6, 5e6, (T, 2))
    real = np.ones((T, 2), bool)
    real[min(500, T - 1), 0] = False
    times = (np.arange(T) * 5 + 1440 * 100).astype("datetime64[m]")
    return FakeView(times, names, o, h, l, c, qv, real, real.copy(), None, 288)


def sub(v, t):
    return FakeView(v.times[:t], v.instruments, v.o[:t], v.h[:t], v.l[:t], v.c[:t], v.qv[:t], v.real[:t],
                    v.tradable[:t], None, v.bars_per_day)


def test_end_to_end_shape_gross_and_values():
    v = make_view()
    for mode in ("conservative", "aggressive"):
        for tgt in ("vwap", "opposite_band"):
            p = dict(PARAMS, entry_mode=mode, target=tgt)
            w = target_positions(v, p)
            assert w.shape == (len(v.times), 2)
            assert np.all(np.isfinite(w))
            assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)
            assert set(np.unique(w).tolist()) <= {-0.5, 0.0, 0.5}


def test_end_to_end_deterministic_and_point_in_time():
    v = make_view()
    p = dict(PARAMS, entry_mode="aggressive")
    full = target_positions(v, p)
    np.testing.assert_array_equal(full, target_positions(v, p))
    for t in (100, 300, 333, 700, 1000):
        np.testing.assert_array_equal(target_positions(sub(v, t), p), full[:t])


def test_bad_params_raise():
    v = make_view(T=300)
    with pytest.raises(ValueError):
        target_positions(v, dict(PARAMS, entry_mode="other"))
    with pytest.raises(ValueError):
        target_positions(v, dict(PARAMS, target="other"))


def test_missing_instrument_raises():
    v = make_view(T=300)
    v2 = FakeView(v.times, ("BTCUSDT", "XXXUSDT"), v.o, v.h, v.l, v.c, v.qv, v.real, v.tradable, None, 288)
    with pytest.raises(ValueError):
        target_positions(v2, PARAMS)
