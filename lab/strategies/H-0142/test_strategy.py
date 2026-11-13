import numpy as np
import pytest

from strategy import target_positions, PARAMS, et_clock

NAMES = ("BTCUSDT", "ETHUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT")
BTC, ETH, AAA, BBB, CCC, DDD, EEE = range(7)
# opening-range RVOL on the last day: BTC 10, ETH 1, AAA 10, BBB 5, CCC/DDD/EEE 1 -> median 1, threshold 2


def civil_days(y, m, d):
    # independent days-since-epoch for test dates (Gregorian, Hinnant algorithm)
    y = y - (1 if m <= 2 else 0)
    era = y // 400
    yoe = y - era * 400
    mp = m - 3 if m > 2 else m + 9
    doy = (153 * mp + 2) // 5 + d - 1
    return era * 146097 + yoe * 365 + yoe // 4 - yoe // 100 + doy - 719468


class FakeView:
    def __init__(self, start_day, n_days):
        T = n_days * 96
        N = len(NAMES)
        self.instruments = NAMES
        self.times = (start_day * 1440 + 15 * np.arange(T, dtype=np.int64)).astype("datetime64[m]")
        self.o = np.full((T, N), 100.0)
        self.c = np.full((T, N), 100.0)
        self.h = np.full((T, N), 100.2)
        self.l = np.full((T, N), 99.8)
        self.qv = np.ones((T, N))
        self.nt = np.ones((T, N))
        self.tbq = np.full((T, N), 0.5)
        self.funding = np.zeros((T, N))
        self.real = np.ones((T, N), dtype=bool)
        self.listed = np.ones((T, N), dtype=bool)
        self.tradable = np.ones((T, N), dtype=bool)
        self.universe = np.ones((T, N), dtype=bool)
        self.bars_per_day = 96

    def col(self, name):
        return self.instruments.index(name)


# February 2021 is EST (UTC-5): ET minute m opens at UTC minute m + 300.
START = civil_days(2021, 2, 1)      # a Monday
NDAYS = 36                          # last day: Monday 8 March 2021 (still EST), 25 weekday sessions before it
LAST = NDAYS - 1


def row(d, h, m):
    return d * 96 + (h * 60 + m + 300) // 15


def make(n_days=NDAYS):
    v = FakeView(START, n_days)
    d = n_days - 1
    i1, i2 = row(d, 9, 30), row(d, 9, 45)
    v.qv[i1:i2 + 1, AAA] = 10.0          # RVOL 10
    v.qv[i1:i2 + 1, BBB] = 5.0           # RVOL 5
    v.qv[i1:i2 + 1, BTC] = 10.0          # RVOL 10 but excluded
    v.c[row(d, 10, 30), AAA] = 102.0     # long breakout (candle closing 10:45)
    v.c[row(d, 11, 0), BBB] = 98.0       # short breakout
    v.c[row(d, 12, 0), BBB] = 101.0      # stop for the short
    v.c[row(d, 10, 15), BTC] = 102.0     # BTC breaks out, never traded
    return v


def test_clock_dst():
    cases = [((2021, 3, 12), 14 * 60 + 30), ((2021, 3, 15), 13 * 60 + 30),
             ((2021, 11, 5), 13 * 60 + 30), ((2021, 11, 8), 14 * 60 + 30),
             ((2022, 3, 11), 14 * 60 + 30), ((2022, 3, 14), 13 * 60 + 30),
             ((2020, 10, 30), 13 * 60 + 30), ((2020, 11, 2), 14 * 60 + 30)]
    mins = np.array([civil_days(*ymd) * 1440 + utc for ymd, utc in cases], dtype=np.int64)
    _, day, et = et_clock(mins.astype("datetime64[m]"))
    assert np.all(et == 570)
    assert np.all(day == mins // 1440)


def test_basic_book():
    v = make()
    w = target_positions(v, PARAMS)
    d = LAST
    k = 1.0 / 3
    a0, a1 = row(d, 10, 30), row(d, 15, 45)
    assert np.all(w[:a0, AAA] == 0)
    assert np.allclose(w[a0:a1, AAA], k)
    assert np.all(w[a1:, AAA] == 0)               # closed at the 15:45-16:00 candle's close
    b0, b1 = row(d, 11, 0), row(d, 12, 0)
    assert np.all(w[:b0, BBB] == 0)
    assert np.allclose(w[b0:b1, BBB], -k)
    assert np.all(w[b1:, BBB] == 0)               # stop, no re-entry
    assert np.all(w[:, BTC] == 0) and np.all(w[:, ETH] == 0) and np.all(w[:, CCC] == 0)
    assert np.abs(w).sum(axis=1).max() <= 1 + 1e-12


def test_top_k_and_rvol_mult():
    v = make()
    w = target_positions(v, {"rvol_mult": 2.0, "top_k": 1})
    assert np.allclose(w[row(LAST, 11, 0), AAA], 1.0)
    assert np.all(w[:, BBB] == 0)
    w = target_positions(v, {"rvol_mult": 6.0, "top_k": 3})
    assert np.allclose(w[row(LAST, 11, 0), AAA], 1 / 3)
    assert np.all(w[:, BBB] == 0)
    w = target_positions(v, {"rvol_mult": 11.0, "top_k": 3})
    assert np.all(w == 0)


def test_tie_broken_by_ov():
    v = make()
    d = LAST
    i1, i2 = row(d, 9, 30), row(d, 9, 45)
    v.qv[: i1, BBB] = 2.0                # BBB baseline OV 4
    v.qv[i1:i2 + 1, BBB] = 20.0          # RVOL 10 = AAA's, but OV 40 > AAA's 20
    w = target_positions(v, {"rvol_mult": 2.0, "top_k": 1})
    assert np.all(w[:, AAA] == 0)
    assert np.allclose(w[row(d, 11, 0), BBB], -1.0)


def test_market_wide_day_selects_nothing():
    v = make()
    d = LAST
    i1, i2 = row(d, 9, 30), row(d, 9, 45)
    v.qv[i1:i2 + 1, :] = 10.0            # median RVOL 10 -> threshold 20
    v.qv[i1:i2 + 1, AAA] = 15.0
    assert np.all(target_positions(v, PARAMS) == 0)


def test_gap_in_opening_range_blocks_coin():
    v = make()
    v.real[row(LAST, 9, 30), AAA] = False
    w = target_positions(v, PARAMS)
    assert np.all(w[:, AAA] == 0)
    assert np.allclose(w[row(LAST, 11, 0), BBB], -1 / 3)


def test_baseline_needs_15_sessions():
    # 15 weekday sessions before the last day -> ok; 14 -> no trade
    ok_days = 21 + 1                     # days 0..20 contain 15 weekdays (3 full weeks), last day = day 21 (Mon)
    v = make(ok_days)
    assert np.any(target_positions(v, PARAMS)[:, AAA] != 0)
    v = FakeView(START + 1, ok_days - 1)  # start Tuesday: only 14 weekday sessions before the last day
    d = ok_days - 2
    i1, i2 = row(d, 9, 30), row(d, 9, 45)
    v.qv[i1:i2 + 1, AAA] = 10.0
    v.c[row(d, 10, 30), AAA] = 102.0
    assert np.all(target_positions(v, PARAMS) == 0)


def test_gap_sessions_dropped_from_baseline():
    v = make()
    # 6 of the previous 20 sessions have a gap for AAA -> only 14 valid -> not eligible
    gaps = 0
    for d in range(LAST - 1, -1, -1):
        if (START + d + 3) % 7 < 5:
            v.real[row(d, 9, 45), AAA] = False
            gaps += 1
            if gaps == 6:
                break
    assert np.all(target_positions(v, PARAMS)[:, AAA] == 0)


def test_entry_window_and_first_breakout_only():
    v = make()
    d = LAST
    v.c[row(d, 10, 30), AAA] = 100.0
    v.c[row(d, 15, 0), AAA] = 102.0      # closes 15:15: ignored
    assert np.all(target_positions(v, PARAMS)[:, AAA] == 0)
    v.c[row(d, 14, 45), AAA] = 102.0     # closes 15:00: taken
    w = target_positions(v, PARAMS)
    assert np.allclose(w[row(d, 14, 45):row(d, 15, 45), AAA], 1 / 3)
    assert np.all(w[row(d, 15, 45):, AAA] == 0)
    v = make()
    v.c[row(d, 10, 0), AAA] = 102.0      # closes 10:15: earliest candle
    v.c[row(d, 10, 30), AAA] = 99.0      # stop
    v.c[row(d, 11, 0), AAA] = 103.0      # later breakout: no re-entry
    w = target_positions(v, PARAMS)
    assert np.allclose(w[row(d, 10, 0):row(d, 10, 30), AAA], 1 / 3)
    assert np.all(w[row(d, 10, 30):, AAA] == 0)
    v = make()
    v.c[row(d, 9, 45), AAA] = 102.0      # the opening-range candle itself is not a breakout candle
    v.c[row(d, 10, 30), AAA] = 100.0
    assert np.all(target_positions(v, PARAMS)[:, AAA] == 0)


def test_universe_exit_and_gap_hold():
    v = make()
    d = LAST
    v.universe[row(d, 13, 0):, AAA] = False
    w = target_positions(v, PARAMS)
    assert np.allclose(w[row(d, 12, 45), AAA], 1 / 3)
    assert np.all(w[row(d, 13, 0):, AAA] == 0)
    v = make()
    v.c[row(d, 11, 15), BBB] = 101.0     # stop-level close on a gap row is ignored
    v.real[row(d, 11, 15), BBB] = False
    w = target_positions(v, PARAMS)
    assert np.allclose(w[row(d, 11, 15), BBB], -1 / 3)
    assert np.all(w[row(d, 12, 0):, BBB] == 0)


def test_not_member_at_selection():
    v = make()
    v.universe[row(LAST, 9, 45), AAA] = False
    assert np.all(target_positions(v, PARAMS)[:, AAA] == 0)


def test_weekend_is_not_a_session():
    v = FakeView(START, NDAYS - 2)       # last day Saturday 6 March 2021
    d = NDAYS - 3
    i1, i2 = row(d, 9, 30), row(d, 9, 45)
    v.qv[i1:i2 + 1, AAA] = 10.0
    v.c[row(d, 10, 30), AAA] = 102.0
    assert np.all(target_positions(v, PARAMS) == 0)


def test_point_in_time_truncation():
    v = make()
    full = target_positions(v, PARAMS)
    for cut in (row(LAST, 9, 45), row(LAST, 10, 30), row(LAST, 11, 30), row(LAST, 15, 45)):
        t = FakeView(START, NDAYS)
        n = cut + 1
        t.times, t.o, t.h, t.l, t.c = v.times[:n], v.o[:n], v.h[:n], v.l[:n], v.c[:n]
        t.qv, t.nt, t.tbq, t.funding = v.qv[:n], v.nt[:n], v.tbq[:n], v.funding[:n]
        t.real, t.listed, t.tradable, t.universe = v.real[:n], v.listed[:n], v.tradable[:n], v.universe[:n]
        part = target_positions(t, PARAMS)
        assert np.array_equal(part, full[: cut + 1])


def test_missing_instrument_raises():
    v = make()
    v.instruments = ("XXXUSDT",) + NAMES[1:]
    with pytest.raises(ValueError):
        target_positions(v, PARAMS)
