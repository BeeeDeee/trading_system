from dataclasses import dataclass

import numpy as np
import pytest

from strategy import target_positions, events, PARAMS
from lab.framework.intraday import synthetic_view

REF = 672


@dataclass
class FakeView:
    c: np.ndarray
    qv: np.ndarray
    tbq: np.ndarray
    real: np.ndarray
    listed: np.ndarray
    universe: np.ndarray
    instruments: tuple


def make_view(T=900, N=2, seed=0):
    """Quiet market: alternating +-0.1% 15m returns, qv = 100 per candle, taker share 0.5."""
    rng = np.random.default_rng(seed)
    lr = np.where(np.arange(T) % 2 == 0, 0.001, -0.001)[:, None] * np.ones((1, N))
    lr = lr * (1.0 + 0.1 * rng.standard_normal((T, N)))
    c = 100.0 * np.exp(np.cumsum(lr, axis=0))
    qv = np.full((T, N), 100.0)
    tbq = np.full((T, N), 50.0)
    ones = np.ones((T, N), dtype=bool)
    return FakeView(c=c, qv=qv, tbq=tbq, real=ones.copy(), listed=ones.copy(), universe=ones.copy(),
                    instruments=tuple("I%d" % j for j in range(N)))


def inject(v, t, j, move, qv_mult=4.0, buy_share=0.2):
    """1-hour move of `move` (log) over candles t-3..t, all of it on candle t (so the window ending at t-1 shows
    no move); 1-hour qv = 4 * 100 * qv_mult and taker-buy share = buy_share over t-3..t."""
    logc = np.log(v.c[:, j])
    later = np.diff(np.log(v.c[t:, j]))       # keep later returns as before
    base = logc[t - 4]
    logc[t - 3:t] = base
    logc[t] = base + move
    logc[t + 1:] = logc[t] + np.cumsum(later)
    v.c[:, j] = np.exp(logc)
    v.qv[t - 3:t, j] = 100.0
    v.tbq[t - 3:t, j] = 50.0
    v.qv[t, j] = 100.0 * (4.0 * qv_mult - 3.0)
    v.tbq[t, j] = buy_share * 400.0 * qv_mult - 150.0


def sigma_at(v, t, j):
    lr = np.diff(np.log(v.c[:, j]))          # lr[k-1] = ln(c[k]/c[k-1])
    window = lr[t - 4 - REF:t - 4]          # returns of candles t-675..t-4
    return np.std(window)


def test_long_event_and_hold():
    v = make_view()
    t = 800
    inject(v, t, 0, -0.05, buy_share=0.2)
    ev = events(v, PARAMS)
    assert ev[t, 0] == 1
    assert not ev[:t, 0].any()
    assert ev[:, 1].sum() == 0
    pos = target_positions(v, PARAMS)
    h = PARAMS["hold_candles"]
    assert np.all(pos[t:t + h, 0] == 1.0 / 20)
    assert pos[t - 1, 0] == 0 and pos[t + h, 0] == 0
    assert np.all(pos[:, 1] == 0)


def test_short_event():
    v = make_view()
    t = 800
    inject(v, t, 1, 0.05, buy_share=0.8)
    ev = events(v, PARAMS)
    assert ev[t, 1] == -1
    pos = target_positions(v, PARAMS)
    assert np.all(pos[t:t + 8, 1] == -1.0 / 20)


def test_z_definition_threshold():
    v = make_view()
    t = 800
    sig = sigma_at(v, t, 0)
    # move of exactly 3.9 and 4.1 one-hour sigmas
    v1 = make_view()
    inject(v1, t, 0, -3.9 * 2 * sig, buy_share=0.2)
    assert abs(sigma_at(v1, t, 0) - sig) < 1e-12
    assert events(v1, PARAMS)[t, 0] == 0
    assert events(v1, dict(PARAMS, z_threshold=3))[t, 0] == 1
    v2 = make_view()
    inject(v2, t, 0, -4.1 * 2 * sig, buy_share=0.2)
    assert events(v2, PARAMS)[t, 0] == 1
    assert events(v2, dict(PARAMS, z_threshold=5))[t, 0] == 0


def test_volume_and_taker_filters():
    t = 800
    v = make_view()
    inject(v, t, 0, -0.05, qv_mult=2.9, buy_share=0.2)
    assert events(v, PARAMS)[t, 0] == 0
    v = make_view()
    inject(v, t, 0, -0.05, qv_mult=3.1, buy_share=0.2)
    assert events(v, PARAMS)[t, 0] == 1
    # taker side must match: a drop with buy share >= 0.5 is not a long cascade
    v = make_view()
    inject(v, t, 0, -0.05, buy_share=0.5)
    assert events(v, PARAMS)[t, 0] == 0
    v = make_view()
    inject(v, t, 0, 0.05, buy_share=0.4)
    assert events(v, PARAMS)[t, 0] == 0


def test_eligibility():
    t = 800
    # not in universe
    v = make_view()
    inject(v, t, 0, -0.05)
    v.universe[t, 0] = False
    assert events(v, PARAMS)[t, 0] == 0
    # a gap candle inside t-3..t
    v = make_view()
    inject(v, t, 0, -0.05)
    v.real[t - 2, 0] = False
    assert events(v, PARAMS)[t, 0] == 0
    # not listed
    v = make_view()
    inject(v, t, 0, -0.05)
    v.listed[t, 0] = False
    assert events(v, PARAMS)[t, 0] == 0
    # fewer than 672 real candles of history
    v = make_view()
    inject(v, t, 0, -0.05)
    v.real[:200, 0] = False
    assert events(v, PARAMS)[t, 0] == 0


def test_late_listing_and_short_history():
    v = make_view(T=900)
    v.c[:100, 0] = np.nan
    v.real[:100, 0] = False
    v.listed[:100, 0] = False
    # event right after 672 real candles but sigma window reaches before listing -> no event
    t = 100 + REF + 1
    inject(v, t, 0, -0.08)
    assert events(v, PARAMS)[t, 0] == 0
    pos = target_positions(v, PARAMS)
    assert np.all(np.isfinite(pos))


def test_one_position_at_a_time_and_param_hold():
    v = make_view(T=1000)
    t = 800
    inject(v, t, 0, -0.05)
    inject(v, t + 4, 0, 0.08, buy_share=0.9)   # opposite event while in position: ignored
    ev = events(v, PARAMS)
    assert ev[t, 0] == 1 and ev[t + 4, 0] == -1
    pos = target_positions(v, PARAMS)
    assert np.all(pos[t:t + 8, 0] == 1.0 / 20)
    assert np.all(pos[t + 8:, 0] == 0)
    # with hold 4: the event at t+4 is decided while still in position (exit fills at the open of t+5): ignored;
    # the window t+2..t+5 still qualifies as a short event and opens a new short from row t+5
    assert ev[t + 5, 0] == -1
    pos4 = target_positions(v, dict(PARAMS, hold_candles=4))
    assert np.all(pos4[t:t + 4, 0] == 1.0 / 20)
    assert pos4[t + 4, 0] == 0
    assert np.all(pos4[t + 5:t + 9, 0] == -1.0 / 20)
    assert np.all(pos4[t + 9:, 0] == 0)
    pos16 = target_positions(v, dict(PARAMS, hold_candles=16))
    assert np.all(pos16[t:t + 16, 0] == 1.0 / 20)
    assert pos16[t + 16, 0] == 0


def test_new_position_after_exit():
    v = make_view(T=1000)
    t = 800
    inject(v, t, 0, -0.05)
    inject(v, t + 9, 0, 0.08, buy_share=0.9)
    pos = target_positions(v, PARAMS)
    assert np.all(pos[t:t + 8, 0] == 1.0 / 20)
    assert pos[t + 8, 0] == 0
    assert np.all(pos[t + 9:t + 17, 0] == -1.0 / 20)


def test_universe_exit_closes_position_early():
    v = make_view(T=1000)
    t = 800
    inject(v, t, 0, -0.05)
    v.universe[t + 3:t + 5, 0] = False            # leaves the universe at row t+3, back at t+5
    pos = target_positions(v, PARAMS)
    assert np.all(pos[t:t + 3, 0] == 1.0 / 20)
    assert np.all(pos[t + 3:, 0] == 0)              # forced exit at open t+4, not re-opened on re-entry
    assert np.all(pos[~v.universe] == 0)
    # leaving the universe after the scheduled exit changes nothing
    v2 = make_view(T=1000)
    inject(v2, t, 0, -0.05)
    v2.universe[t + 8:, 0] = False
    pos2 = target_positions(v2, PARAMS)
    assert np.all(pos2[t:t + 8, 0] == 1.0 / 20)
    assert np.all(pos2[t + 8:, 0] == 0)


def test_point_in_time_truncation():
    v = make_view(T=1000)
    inject(v, 800, 0, -0.05)
    full = target_positions(v, PARAMS)
    for cut in (700, 801, 805, 900):
        tv = FakeView(c=v.c[:cut], qv=v.qv[:cut], tbq=v.tbq[:cut], real=v.real[:cut], listed=v.listed[:cut],
                      universe=v.universe[:cut], instruments=v.instruments)
        assert np.array_equal(target_positions(tv, PARAMS), full[:cut])


def test_synthetic_gross_and_determinism():
    data = synthetic_view()
    a = target_positions(data, PARAMS)
    b = target_positions(data, PARAMS)
    assert a.shape == (len(data.times), len(data.instruments))
    assert np.array_equal(a, b)
    assert np.all(np.abs(a).sum(axis=1) <= 1.0 + 1e-12)
    assert set(np.unique(a)).issubset({-1.0 / 20, 0.0, 1.0 / 20})
    if data.universe is not None:
        assert np.all(a[~np.asarray(data.universe, dtype=bool)] == 0)
