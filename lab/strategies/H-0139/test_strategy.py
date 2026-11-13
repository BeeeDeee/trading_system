import dataclasses

import numpy as np
import pytest

from strategy import PARAMS, target_positions, days_from_civil, year_of_day, weekday, ny_offset_minutes
from strategy import time_frame, nr_eligibility, opening_range, session_trades


# ------------------------------------------------------------------------------------------ calendar
def test_days_from_civil_anchors():
    assert int(days_from_civil(1970, 1, 1)) == 0
    assert int(days_from_civil(2020, 1, 1)) == 18262
    assert int(days_from_civil(2022, 12, 31)) == 18262 + 366 + 365 + 364
    assert int(days_from_civil(2020, 3, 1)) == 18262 + 31 + 29
    assert int(weekday(days_from_civil(2020, 1, 1))) == 2          # Wednesday
    assert int(weekday(days_from_civil(2020, 1, 6))) == 0          # Monday
    days = np.arange(17000, 20500)
    y = year_of_day(days)
    assert np.all(days_from_civil(y, 1, 1) <= days)
    assert np.all(days_from_civil(y + 1, 1, 1) > days)


# (year, March DST-start Sunday, November DST-end Sunday) of the dev years
TRANSITIONS = [(2020, 8, 1), (2021, 14, 7), (2022, 13, 6)]


@pytest.mark.parametrize("year,mar,nov", TRANSITIONS)
def test_dst_transitions(year, mar, nov):
    start_day = int(days_from_civil(year, 3, mar))
    end_day = int(days_from_civil(year, 11, nov))
    assert int(weekday(start_day)) == 6 and int(weekday(end_day)) == 6
    m0 = start_day * 1440
    m1 = end_day * 1440
    # spring forward at 02:00 EST = 07:00 UTC
    assert int(ny_offset_minutes(m0 + 6 * 60 + 45)) == -300
    assert int(ny_offset_minutes(m0 + 7 * 60)) == -240
    # fall back at 02:00 EDT = 06:00 UTC
    assert int(ny_offset_minutes(m1 + 5 * 60 + 45)) == -240
    assert int(ny_offset_minutes(m1 + 6 * 60)) == -300
    # the Sunday a week earlier / later is outside / inside
    assert int(ny_offset_minutes(m0 - 7 * 1440 + 12 * 60)) == -300
    assert int(ny_offset_minutes(m1 - 7 * 1440 + 12 * 60)) == -240
    # 09:30 New York in summer = 13:30 UTC, in winter = 14:30 UTC
    assert int(ny_offset_minutes(int(days_from_civil(year, 7, 1)) * 1440 + 13 * 60 + 30)) == -240
    assert int(ny_offset_minutes(int(days_from_civil(year, 1, 15)) * 1440 + 14 * 60 + 30)) == -300


# ------------------------------------------------------------------------------------- fake market
@dataclasses.dataclass
class FakeView:
    times: np.ndarray
    instruments: tuple
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    real: np.ndarray
    tradable: np.ndarray
    universe: np.ndarray

    def col(self, name):
        return self.instruments.index(name)


START = 18267            # Monday 2020-01-06 (winter: New York = UTC-5, 09:30 NY = 14:30 UTC)
N_DAYS = 12              # Mon 6 .. Fri 17
# daily half-range set on the 03:00 UTC candle of each day: Mon6..Fri10, Sat, Sun, Mon13..Fri17
A = [10, 8, 6, 4, 6, 0.5, 0.5, 9, 3, 5, 7, 8]
R_OR1, R_OR2 = 58, 59    # 14:30, 14:45 UTC = 09:30, 09:45 NY
R_1000, R_1015, R_1030, R_1045, R_1100, R_1200 = 60, 61, 62, 63, 64, 68
R_1445, R_1500, R_1530, R_1545, R_1600 = 79, 80, 82, 83, 84
BTC, ETH = 0, 1


def make_view(btc_break_row=R_1015):
    T = N_DAYS * 96
    times = (START * 1440 + 15 * np.arange(T, dtype=np.int64)).astype("datetime64[m]")
    c = np.full((T, 2), 100.0)
    h = np.full((T, 2), 100.0)
    l = np.full((T, 2), 100.0)
    for d in range(N_DAYS):
        b = d * 96
        h[b + 12, :] = 100.0 + A[d]
        l[b + 12, :] = 100.0 - A[d]
        if weekday(START + d) >= 5:
            continue
        h[b + R_OR1, :], l[b + R_OR1, :] = 101.0, 99.0
        h[b + R_OR2, :], l[b + R_OR2, :] = 100.5, 99.5
        # BTC: 10:00 close inside, then a close above OR_H on btc_break_row, drifts to 102 at 12:00
        c[b + R_1000, BTC] = 100.5
        c[b + btc_break_row:b + 96, BTC] = 101.5
        if btc_break_row < R_1200:
            c[b + R_1200:b + 96, BTC] = 102.0
        # ETH: short at 10:30, stop at 11:00 (close above OR_H), later close below OR_L (no re-entry)
        c[b + R_1030, ETH] = 98.5
        c[b + R_1045, ETH] = 99.5
        c[b + R_1100, ETH] = 101.2
        c[b + R_1200:b + 96, ETH] = 98.0
    o = np.vstack([c[:1], c[:-1]])
    h = np.maximum(np.maximum(h, c), o)
    l = np.minimum(np.minimum(l, c), o)
    ones = np.ones((T, 2), dtype=bool)
    return FakeView(times, ("BTCUSDT", "ETHUSDT"), o, h, l, c, ones.copy(), ones.copy(), ones.copy())


def expected(view, sessions, btc_rows=(R_1015, R_1545), eth_rows=(R_1030, R_1100)):
    w = np.zeros((len(view.times), 2))
    for d in sessions:
        b = d * 96
        w[b + btc_rows[0]:b + btc_rows[1], BTC] = 0.5
        w[b + eth_rows[0]:b + eth_rows[1], ETH] = -0.5
    return w


def test_nr4_sessions_and_trades():
    v = make_view()
    w = target_positions(v, PARAMS)
    assert PARAMS["nr_n"] == 4
    # NR4 at Thu9 -> Fri10 session (day 4); NR4 at Tue14 -> Wed15 (day 9); weekends excluded so Mon13 is not
    np.testing.assert_array_equal(w, expected(v, [4, 9]))


def test_nr3_sessions():
    v = make_view()
    w = target_positions(v, {"nr_n": 3})
    # NR3 at Wed8, Thu9, Tue14 -> Thu9 (3), Fri10 (4), Wed15 (9)
    np.testing.assert_array_equal(w, expected(v, [3, 4, 9]))


def test_nr7_needs_seven_weekday_bars():
    v = make_view()
    w = target_positions(v, {"nr_n": 7})
    # first full window ends at Tue14: [10, 8, 6, 4, 6, 9, 3], 3 is the minimum -> Wed15 only
    np.testing.assert_array_equal(w, expected(v, [9]))


def test_entry_cutoff_last_candle_1445():
    v = make_view(btc_break_row=R_1445)
    w = target_positions(v, PARAMS)
    exp = expected(v, [4, 9], btc_rows=(R_1445, R_1545))
    np.testing.assert_array_equal(w, exp)


def test_no_entry_after_1500():
    v = make_view(btc_break_row=R_1500)
    w = target_positions(v, PARAMS)
    assert np.all(w[:, BTC] == 0.0)
    assert np.any(w[:, ETH] != 0.0)


def test_gap_in_opening_range_skips_session():
    v = make_view()
    v.real[4 * 96 + R_OR2, ETH] = False
    w = target_positions(v, PARAMS)
    exp = expected(v, [4, 9])
    exp[4 * 96:5 * 96, ETH] = 0.0
    np.testing.assert_array_equal(w, exp)


def test_gappy_daily_bar_makes_session_ineligible():
    v = make_view()
    v.real[2 * 96 + 30:2 * 96 + 35, BTC] = False         # 5 gaps on Wed8 (inside the Fri10 NR4 window)
    w = target_positions(v, PARAMS)
    exp = expected(v, [4, 9])
    exp[4 * 96:5 * 96, BTC] = 0.0
    np.testing.assert_array_equal(w, exp)
    v2 = make_view()
    v2.real[2 * 96 + 30:2 * 96 + 34, BTC] = False        # 4 gaps are allowed
    np.testing.assert_array_equal(target_positions(v2, PARAMS), expected(v2, [4, 9]))


def test_point_in_time_truncation():
    v = make_view()
    full = target_positions(v, PARAMS)
    for cut in (4 * 96 + R_1030, 4 * 96 + R_1200, 9 * 96 + 70, 7 * 96 + 5):
        t = cut + 1
        vt = FakeView(v.times[:t], v.instruments, v.o[:t], v.h[:t], v.l[:t], v.c[:t], v.real[:t],
                      v.tradable[:t], v.universe[:t])
        np.testing.assert_array_equal(target_positions(vt, PARAMS), full[:t])


def test_universe_and_gross():
    v = make_view()
    v.universe[:, ETH] = False
    w = target_positions(v, PARAMS)
    assert np.all(w[:, ETH] == 0.0)
    w = target_positions(make_view(), PARAMS)
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)


def test_missing_instrument_raises():
    v = make_view()
    v.instruments = ("BTCUSDT", "XRPUSDT")
    with pytest.raises(ValueError):
        target_positions(v, PARAMS)


def test_flag_and_trade_primitives():
    # the primitives the session diagnostic reuses: eligibility/definedness and the trade list on all sessions
    v = make_view()
    frame = time_frame(v.times)
    h, l, c, real = v.h[:, BTC], v.l[:, BTC], v.c[:, BTC], v.real[:, BTC]
    elig, defined = nr_eligibility(frame, h, l, real, 4)
    orh, orl, valid = opening_range(frame, h, l, real)
    base = START - 1                                   # calendar axis starts the day before the first row
    assert frame["n_cal"] == N_DAYS + 1
    assert sorted(np.flatnonzero(elig) + base - START) == [4, 9]
    # Mon6..Thu9 sessions have fewer than 4 earlier weekday bars: undefined
    assert sorted(np.flatnonzero(valid & ~defined) + base - START) == [0, 1, 2, 3]
    assert int(valid.sum()) == 10 and orh[START + 4 - base] == 101.0 and orl[START + 4 - base] == 99.0
    trades = session_trades(frame, c, orh, orl, valid)
    assert len(trades) == 10
    for entry, exit_row, side, s in trades:
        d = s + base - START
        assert (entry, exit_row, side) == (d * 96 + R_1015, d * 96 + R_1545, 1)
