import numpy as np
import pytest

from strategy import (PARAMS, core_positions, days_from_civil, sessions, us_dst, vol_rank, weekday)

BARS_PER_DAY = 96


def _market(ndays=99, growing=True, n_inst=2):
    # 15m grid from a Monday midnight UTC (first Monday of January); alternating returns whose size grows
    # (or shrinks) every candle, so every session's 7-day rv is strictly above (below) all earlier ones.
    start = int(days_from_civil(2021, 1, 4))
    T = ndays * BARS_PER_DAY
    m = start * 1440 + np.arange(T, dtype=np.int64) * 15
    t = np.arange(T, dtype=np.float64)
    a = 0.0005 + 1e-7 * t if growing else 0.0005 + 1e-7 * (T - t)
    r = a * np.where(np.arange(T) % 2 == 0, 1.0, -1.0)
    c1 = 100.0 * np.exp(np.cumsum(r))
    c = np.repeat(c1[:, None], n_inst, axis=1)
    o = np.vstack([c[:1], c[:-1]])
    h = np.maximum(o, c) * 1.0001
    l = np.minimum(o, c) * 0.9999
    real = np.ones((T, n_inst), dtype=bool)
    return m, h, l, c, real


def _last_session_row930(ndays=99):
    # last day (2021-04-12, Monday) is under DST: 09:30 NY = 13:30 UTC = candle 54 of the day
    return (ndays - 1) * BARS_PER_DAY + 54


def _set_flat_session(h, l, c, r930):
    c[r930:] = 100.0
    h[r930:] = 100.2
    l[r930:] = 99.8
    h[r930] = 101.0
    l[r930] = 99.0


# ---------------------------------------------------------------------------- calendar
def test_days_from_civil_and_weekday():
    assert int(days_from_civil(1970, 1, 1)) == 0
    assert int(days_from_civil(2000, 3, 1)) == 11017
    assert int(weekday(days_from_civil(2021, 1, 4))) == 0      # Monday
    assert int(weekday(days_from_civil(2022, 11, 6))) == 6     # Sunday


@pytest.mark.parametrize("y,before,after_start,before_end,after_end", [
    (2020, (3, 6), (3, 9), (10, 30), (11, 2)),
    (2021, (3, 12), (3, 15), (11, 5), (11, 8)),
    (2022, (3, 11), (3, 14), (11, 4), (11, 7)),
])
def test_us_dst_transitions(y, before, after_start, before_end, after_end):
    assert not bool(us_dst(days_from_civil(y, *before)))
    assert bool(us_dst(days_from_civil(y, *after_start)))
    assert bool(us_dst(days_from_civil(y, *before_end)))
    assert not bool(us_dst(days_from_civil(y, *after_end)))
    # the transition Sundays themselves
    assert bool(us_dst(days_from_civil(y, 3, after_start[1] - 1)))
    assert not bool(us_dst(days_from_civil(y, 11, after_end[1] - 1)))


def test_sessions_open_time_and_weekdays():
    m, *_ = _market(ndays=99)
    days, t930 = sessions(m)
    assert np.all(weekday(days) < 5)
    assert len(days) == 99 - 2 * 14                                 # 14 weekends in 99 days from a Monday
    jan4 = int(days_from_civil(2021, 1, 4))
    apr12 = int(days_from_civil(2021, 4, 12))
    assert t930[days == jan4][0] == jan4 * 1440 + 14 * 60 + 30      # winter: 14:30 UTC
    assert t930[days == apr12][0] == apr12 * 1440 + 13 * 60 + 30    # summer: 13:30 UTC


# ---------------------------------------------------------------------------- rank
def test_vol_rank_strictly_below_previous_60():
    rv = np.concatenate([np.arange(60.0), [41.5]])[:, None]
    valid = np.ones_like(rv, dtype=bool)
    rank = vol_rank(rv, valid)
    assert np.all(np.isnan(rank[:60, 0]))                             # warm-up
    assert rank[60, 0] == pytest.approx(42 / 60)
    # ties are not "strictly below"
    rv2 = np.concatenate([np.full(60, 5.0), [5.0]])[:, None]
    assert vol_rank(rv2, np.ones_like(rv2, dtype=bool))[60, 0] == 0.0


def test_vol_rank_skips_invalid_sessions():
    rv = np.concatenate([np.arange(60.0), [np.nan], [100.0]])[:, None]
    valid = np.isfinite(rv)
    rank = vol_rank(rv, valid)
    assert np.isnan(rank[60, 0])
    assert rank[61, 0] == 1.0


# ---------------------------------------------------------------------------- full rule
def test_long_breakout_with_stop_and_short_with_time_exit():
    m, h, l, c, real = _market()
    r = _last_session_row930()
    _set_flat_session(h, l, c, r)
    # instrument 0: non-real close above the range at 10:00 cannot enter; real 10:15 close above -> long;
    # 11:00 close below the range low -> stop
    c[r + 2, 0] = 103.0
    real[r + 2, 0] = False
    c[r + 3, 0] = 101.5
    c[r + 6, 0] = 98.5
    # instrument 1: 10:30 close below -> short; non-real close above the high does not stop; time exit
    c[r + 4, 1] = 98.0
    c[r + 10, 1] = 102.0
    real[r + 10, 1] = False
    c[r + 26, 1] = 102.0     # 16:00 candle, after the exit
    pos = core_positions(m, h, l, c, real, PARAMS["vol_rank_min"])
    day = pos[r:, :]
    exp0 = np.zeros(len(day))
    exp0[3:6] = 0.5
    exp1 = np.zeros(len(day))
    exp1[4:25] = -0.5         # rows 10:30 .. 15:30 NY; the 15:45 row is flat (filled at the 16:00 open)
    np.testing.assert_array_equal(day[:, 0], exp0)
    np.testing.assert_array_equal(day[:, 1], exp1)
    assert np.all(np.abs(pos).sum(axis=1) <= 1.0 + 1e-12)
    assert set(np.unique(pos)) <= {-0.5, 0.0, 0.5}


def test_no_reentry_after_stop():
    m, h, l, c, real = _market()
    r = _last_session_row930()
    _set_flat_session(h, l, c, r)
    c[r + 3, 0] = 101.5
    c[r + 6, 0] = 98.5
    c[r + 8, 0] = 97.0       # second breakout (down) must not be traded
    c[r + 9, 0] = 102.0
    pos = core_positions(m, h, l, c, real, PARAMS["vol_rank_min"])
    assert np.all(pos[r + 6:, 0] == 0.0)


def test_entry_window_ends_at_1515():
    m, h, l, c, real = _market()
    r = _last_session_row930()
    _set_flat_session(h, l, c, r)
    c[r + 24, 0] = 101.5     # 15:30 candle: outside the entry window
    c[r + 23, 1] = 101.5     # 15:15 candle: last entry candle
    pos = core_positions(m, h, l, c, real, PARAMS["vol_rank_min"])
    assert np.all(pos[r:, 0] == 0.0)
    assert pos[r + 23, 1] == 0.5 and pos[r + 24, 1] == 0.5 and pos[r + 25, 1] == 0.0
    assert pos[r + 22, 1] == 0.0


def test_low_vol_regime_never_trades():
    m, h, l, c, real = _market(growing=False)
    r = _last_session_row930()
    _set_flat_session(h, l, c, r)
    c[r + 3, :] = 101.5
    pos = core_positions(m, h, l, c, real, 0.5)
    assert np.all(pos == 0.0)


def test_parameter_changes_output():
    m, h, l, c, real = _market()
    r = _last_session_row930()
    _set_flat_session(h, l, c, r)
    c[r + 3, :] = 101.5
    on = core_positions(m, h, l, c, real, 0.667)
    off = core_positions(m, h, l, c, real, 1.01)
    assert np.any(on[r:] != 0.0)
    assert np.all(off == 0.0)


def test_opening_range_candle_not_real_blocks_session():
    m, h, l, c, real = _market()
    r = _last_session_row930()
    _set_flat_session(h, l, c, r)
    c[r + 3, :] = 101.5
    real[r + 1, 0] = False
    pos = core_positions(m, h, l, c, real, PARAMS["vol_rank_min"])
    assert np.all(pos[r:, 0] == 0.0)
    assert pos[r + 3, 1] == 0.5


def test_rv_window_with_too_few_real_candles_blocks_session():
    m, h, l, c, real = _market()
    r = _last_session_row930()
    _set_flat_session(h, l, c, r)
    c[r + 3, :] = 101.5
    real[r - 1 - 70:r - 1, 0] = False      # 70 of the 672 window candles missing -> < 90 % real
    real[r - 1 - 60:r - 1, 1] = False      # 60 missing -> still >= 90 %, but rank may change; only check 0
    pos = core_positions(m, h, l, c, real, PARAMS["vol_rank_min"])
    assert np.all(pos[r:, 0] == 0.0)


def test_warm_up_needs_60_prior_valid_sessions():
    m, h, l, c, real = _market(ndays=70)
    pos = core_positions(m, h, l, c, real, 0.0)
    assert np.all(pos == 0.0)
    # sessions with a complete rv window start on 2021-01-11; the 61st of them is the first tradable one
    m, h, l, c, real = _market(ndays=99)
    days, t930 = sessions(m)
    first_valid = int(np.flatnonzero(days == int(days_from_civil(2021, 1, 11)))[0])
    rows = [(int(t930[first_valid + k]) - int(m[0])) // 15 for k in (59, 60)]
    for r in rows:
        c[r + 3, :] = max(h[r, 0], h[r + 1, 0]) * 1.01
    pos = core_positions(m, h, l, c, real, PARAMS["vol_rank_min"])
    assert np.all(pos[rows[0]:rows[0] + 30] == 0.0)
    assert np.all(pos[rows[1] + 3] == 0.5)
    assert np.all(pos[:rows[1] + 3] == 0.0)


def test_point_in_time_truncation():
    m, h, l, c, real = _market()
    days, t930 = sessions(m)
    for k, s in enumerate(range(len(days) - 10, len(days))):
        r = (int(t930[s]) - int(m[0])) // 15
        hi, lo = max(h[r, 0], h[r + 1, 0]), min(l[r, 0], l[r + 1, 0])
        c[r + 3 + k, 0] = hi * 1.01 if k % 2 == 0 else lo * 0.99
        c[r + 4 + k, 1] = lo * 0.99 if k % 3 == 0 else hi * 1.01
        if k % 4 == 1:
            c[r + 9 + k, :] = [lo * 0.98, hi * 1.02]          # stops
    full = core_positions(m, h, l, c, real, 0.5)
    assert (np.abs(full).sum(axis=1) > 0).sum() > 50                   # the cuts below cross live trades
    for cut in (5000, 7000, 8000, 8900, 9000, 9100, 9200, 9300, 9450):
        part = core_positions(m[:cut], h[:cut], l[:cut], c[:cut], real[:cut], 0.5)
        np.testing.assert_array_equal(part, full[:cut])
    rng = np.random.default_rng(0)
    cut = 8500
    c2 = c.copy()
    c2[cut:] = c2[cut:] * np.exp(rng.normal(0, 0.01, size=c2[cut:].shape))
    h2 = np.maximum(h, c2) * 1.001
    l2 = np.minimum(l, c2) * 0.999
    h2[:cut], l2[:cut] = h[:cut], l[:cut]
    noisy = core_positions(m, h2, l2, c2, real, 0.5)
    np.testing.assert_array_equal(noisy[:cut], full[:cut])
