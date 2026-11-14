import math

import numpy as np
import pytest

from strategy import PARAMS, find_trades, session_bands, target_positions, wilder_atr

BAR = 5
DAY0 = 20000 * 1440   # an arbitrary UTC midnight in minutes since epoch


def _minutes(start_minute_of_day, n):
    return (DAY0 + start_minute_of_day + BAR * np.arange(n)).astype(np.int64)


# ---------------------------------------------------------------- session bands

def test_session_vwap_and_sigma_by_hand():
    n = 14
    m = _minutes(0, n)
    c = np.full(n, 100.0)
    c[13] = 110.0
    h = c.copy()
    l = c.copy()
    qv = np.zeros(n)
    qv[12] = 100.0 * 1.0       # w = 1 at tp = 100
    qv[13] = 110.0 * 3.0       # w = 3 at tp = 110
    real = np.ones(n, dtype=bool)
    V, U, L, d = session_bands(m, h, l, c, qv, real, BAR)
    assert not d[:12].any()                    # warm-up: first 12 candles
    assert d[12] and d[13]
    assert V[12] == pytest.approx(100.0)
    assert U[12] == pytest.approx(100.0)
    assert V[13] == pytest.approx(107.5)
    assert U[13] - V[13] == pytest.approx(math.sqrt(18.75))
    assert V[13] - L[13] == pytest.approx(math.sqrt(18.75))


def test_session_resets_and_gap_rows_have_no_weight():
    n = 288 + 14
    m = _minutes(0, n)
    c = np.full(n, 100.0)
    c[288:] = 200.0
    c[289] = 300.0
    h = c.copy()
    l = c.copy()
    qv = c * 1.0
    real = np.ones(n, dtype=bool)
    real[289] = False                          # gap row: weight 0
    V, U, L, d = session_bands(m, h, l, c, qv, real, BAR)
    assert V[287] == pytest.approx(100.0)
    assert not d[288:300].any()                # warm-up of the new session
    assert V[300] == pytest.approx(200.0)      # previous session and gap row ignored
    assert U[300] == pytest.approx(200.0)


# ---------------------------------------------------------------- ATR

def test_wilder_atr():
    n = 20
    c = np.full(n, 100.0)
    h = c + 1.0
    l = c - 1.0
    h[15] = 115.0                               # TR = 16 at row 15
    real = np.ones(n, dtype=bool)
    a = wilder_atr(h, l, c, real, 14)
    assert np.isnan(a[:13]).all()
    assert a[13] == pytest.approx(2.0)
    assert a[14] == pytest.approx(2.0)
    assert a[15] == pytest.approx(2.0 + 14.0 / 14.0)
    real[16] = False                            # gap row carries the value
    a2 = wilder_atr(h, l, c, real, 14)
    assert a2[16] == pytest.approx(a2[15])


# ---------------------------------------------------------------- setup logic on injected bands

def _path(start_mod=60):
    # closes, highs, lows; V=100, U=101, L=99, ATR=1 everywhere
    rows = [
        (102.0, 102.2, 101.8),   # 0 outside above U
        (100.8, 101.0, 100.6),   # 1 re-entry
        (100.5, 100.7, 100.3),   # 2 inside, above V
        (100.4, 100.6, 100.05),  # 3 test candle b (low <= V + 0.1 ATR, close in (V, U])
        (100.5, 100.7, 100.3),   # 4 no confirmation yet
        (100.8, 100.9, 100.4),   # 5 close > h[b] = 100.6 -> signal s
        (100.9, 101.0, 100.7),   # 6
        (101.0, 101.1, 100.8),   # 7 close >= U -> band target exit
        (100.5, 100.6, 100.4),   # 8
        (99.7, 99.8, 99.6),      # 9 close <= stop 99.8 -> stop exit for 2R
        (99.7, 99.8, 99.6),      # 10
        (99.7, 99.8, 99.6),      # 11
    ]
    c = np.array([r[0] for r in rows])
    h = np.array([r[1] for r in rows])
    l = np.array([r[2] for r in rows])
    n = len(rows)
    return _minutes(start_mod, n), h, l, c


def _run(params, m, h, l, c, real=None):
    n = len(c)
    real = np.ones(n, dtype=bool) if real is None else real
    V = np.full(n, 100.0)
    U = np.full(n, 101.0)
    L = np.full(n, 99.0)
    d = np.ones(n, dtype=bool)
    atr = np.ones(n)
    member = np.ones(n, dtype=bool)
    return find_trades(m, h, l, c, real, V, U, L, d, atr, member, params["confirm_bars"], params["target"],
                       params["tol_atr"], BAR)


def test_primary_long_trade_band_target():
    tr = _run(PARAMS, *_path())
    assert len(tr) == 1
    t = tr[0]
    assert (t["test_row"], t["signal_row"], t["exit_row"], t["side"]) == (3, 5, 7, 1)
    assert t["stop"] == pytest.approx(100.05 - 0.25)
    assert t["target"] == pytest.approx(101.0)
    assert t["reason"] == "target"


def test_confirm_zero_enters_on_test_candle():
    tr = _run(dict(PARAMS, confirm_bars=0), *_path())
    assert len(tr) == 1
    assert (tr[0]["signal_row"], tr[0]["exit_row"]) == (3, 7)


def test_confirm_one_cancels_without_confirmation():
    assert _run(dict(PARAMS, confirm_bars=1), *_path()) == []


def test_close_below_v_cancels():
    m, h, l, c = _path()
    c[4] = 99.9                                 # close below V before the confirmation
    assert _run(PARAMS, m, h, l, c) == []


def test_tol_zero_no_test_candle():
    assert _run(dict(PARAMS, tol_atr=0.0), *_path()) == []


def test_two_r_target_and_stop_exit():
    tr = _run(dict(PARAMS, target="2R"), *_path())
    assert len(tr) == 1
    assert tr[0]["target"] == pytest.approx(100.8 + 2 * (100.8 - 99.8))
    assert tr[0]["exit_row"] == 9 and tr[0]["reason"] == "stop"


def test_band_target_skipped_when_close_beyond_band():
    m, h, l, c = _path()
    c[5] = 101.2
    h[5] = 101.3
    assert _run(PARAMS, m, h, l, c) == []
    assert len(_run(dict(PARAMS, target="2R"), m, h, l, c)) == 1


def test_gap_row_cannot_be_signal():
    m, h, l, c = _path()
    real = np.ones(len(c), dtype=bool)
    real[5] = False
    tr = _run(PARAMS, m, h, l, c, real)
    assert len(tr) == 1 and tr[0]["signal_row"] == 6


def test_reentry_must_be_within_24_candles():
    m, h, l, c = _path()
    pad = 24
    # insert 24 inside candles between the re-entry (row 1) and the rest: re-entry is then 25+ candles back
    c2 = np.concatenate([c[:2], np.full(pad, 100.5), c[2:]])
    h2 = np.concatenate([h[:2], np.full(pad, 100.7), h[2:]])
    l2 = np.concatenate([l[:2], np.full(pad, 100.3), l[2:]])
    m2 = _minutes(60, len(c2))
    assert _run(PARAMS, m2, h2, l2, c2) == []


def test_short_mirror():
    m, h, l, c = _path()
    c2, h2, l2 = 200.0 - c, 200.0 - l, 200.0 - h
    tr = _run(PARAMS, m, h2, l2, c2)
    assert len(tr) == 1
    t = tr[0]
    assert (t["signal_row"], t["exit_row"], t["side"]) == (5, 7, -1)
    assert t["stop"] == pytest.approx(200.0 - 100.05 + 0.25)
    assert t["target"] == pytest.approx(99.0)


def test_no_signal_in_last_hour():
    # signal row 5 at 23:00 open -> blocked; at 22:55 open -> allowed
    m, h, l, c = _path(start_mod=23 * 60 - 5 * BAR)
    assert _run(PARAMS, m, h, l, c) == []
    m, h, l, c = _path(start_mod=23 * 60 - 6 * BAR)
    assert len(_run(PARAMS, m, h, l, c)) == 1


def test_session_end_exit():
    # signal at 22:55; 2R target never reached, stop never hit -> exit decided on the 23:55 candle
    m, h, l, c = _path(start_mod=23 * 60 - 6 * BAR)
    n = 24
    c = np.concatenate([c[:7], np.full(n - 7, 100.5)])
    h = np.concatenate([h[:7], np.full(n - 7, 100.6)])
    l = np.concatenate([l[:7], np.full(n - 7, 100.4)])
    m = (m[0] + BAR * np.arange(n)).astype(np.int64)
    tr = _run(dict(PARAMS, target="2R"), m, h, l, c)
    assert len(tr) == 1
    e = tr[0]["exit_row"]
    assert (m[e] % 1440) == 1440 - BAR and tr[0]["reason"] == "session_end"


def test_new_trade_needs_test_after_exit():
    m, h, l, c = _path()
    # second setup right after the exit row 7 using test candles inside the trade does not count
    tr = _run(dict(PARAMS, confirm_bars=0), m, h, l, c)
    assert all(t["test_row"] > -1 for t in tr)
    assert len(tr) == 1


def test_open_trade_at_end_of_data():
    m, h, l, c = _path()
    tr = _run(PARAMS, m[:7], h[:7], l[:7], c[:7])
    assert len(tr) == 1 and tr[0]["exit_row"] == 7 and tr[0]["reason"] == "open"


# ---------------------------------------------------------------- full function on a synthetic view

def _view():
    from dataclasses import replace
    from lab.framework.intraday import synthetic_view
    v = synthetic_view()
    return replace(v, instruments=("BTCUSDT", "ETHUSDT") + tuple(v.instruments[2:]))


def test_positions_shape_gross_and_determinism():
    data = _view()
    for p in (PARAMS, dict(PARAMS, confirm_bars=0), dict(PARAMS, target="2R"), dict(PARAMS, tol_atr=0.25)):
        w = target_positions(data, p)
        assert w.shape == (len(data.times), len(data.instruments))
        assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)
        assert set(np.unique(w).tolist()) <= {-0.5, 0.0, 0.5}
        assert np.array_equal(w, target_positions(data, p))
