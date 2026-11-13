import numpy as np
import pytest

from strategy import PARAMS, ny_utc_offset_minutes, orb_positions, target_positions, year_from_days

# integer day numbers (days since 1970-01-01), derived by hand
D_2020_01_01 = 18262
D_2020_03_08 = 18329        # Sunday, DST start 2020
D_2021_03_12 = 18698        # Friday, EST (UTC-5)
D_2021_03_14 = 18700        # Sunday, DST start 2021
D_2021_03_15 = 18701        # Monday, EDT (UTC-4)
D_2021_03_20 = 18706        # Saturday
D_2021_11_07 = 18938        # Sunday, DST end 2021


def k_utc(hh, mm):
    return hh * 4 + mm // 15


def make_day(day, K=1):
    tmin = day * 1440 + 15 * np.arange(96, dtype=np.int64)
    h = np.full((96, K), 100.1)
    l = np.full((96, K), 99.9)
    c = np.full((96, K), 100.0)
    real = np.ones((96, K), dtype=bool)
    return tmin, h, l, c, real


def run(tmin, h, l, c, real, rm=30, stop=True, anchor="us_open"):
    return orb_positions(tmin, h, l, c, real, 15, rm, stop, anchor)[:, 0]


def expected(n, lo, hi, v):
    e = np.zeros(n)
    e[lo:hi] = v
    return e


def test_calendar_helpers():
    assert year_from_days(np.array([D_2020_01_01]))[0] == 2020
    assert year_from_days(np.array([D_2020_01_01 - 1]))[0] == 2019
    assert (D_2021_03_15 + 3) % 7 == 0           # Monday
    t = np.array([D_2021_03_14 * 1440 + 7 * 60 - 1, D_2021_03_14 * 1440 + 7 * 60,
                  D_2021_11_07 * 1440 + 6 * 60 - 1, D_2021_11_07 * 1440 + 6 * 60,
                  D_2020_03_08 * 1440 + 7 * 60 - 1, D_2020_03_08 * 1440 + 7 * 60], dtype=np.int64)
    assert list(ny_utc_offset_minutes(t)) == [-300, -240, -240, -300, -300, -240]


def setup_long(day=D_2021_03_15, a=k_utc(13, 30)):
    tmin, h, l, c, real = make_day(day)
    h[a, 0] = 101.0                              # range high (09:30 NY candle)
    l[a + 1, 0] = 99.0                           # range low (09:45 NY candle)
    h[a + 6, 0] = 101.6
    c[a + 6, 0] = 101.5                          # 11:00 NY candle closes above H
    return tmin, h, l, c, real


def test_long_edt_holds_to_close():
    tmin, h, l, c, real = setup_long()
    a = k_utc(13, 30)
    # positions from the signal row (11:00 NY) through the 15:30 NY candle; 0 at the 15:45 candle (19:45 UTC)
    assert np.array_equal(run(tmin, h, l, c, real), expected(96, a + 6, k_utc(19, 45), 0.5))


def test_est_day_shifts_one_hour():
    a = k_utc(14, 30)                            # 09:30 EST = 14:30 UTC
    tmin, h, l, c, real = setup_long(D_2021_03_12, a)
    assert np.array_equal(run(tmin, h, l, c, real), expected(96, a + 6, k_utc(20, 45), 0.5))
    # a close far above at 14:15 UTC on an EST day is 09:15 NY, before the range: no trade
    tmin, h, l, c, real = make_day(D_2021_03_12)
    h[a, 0] = 101.0
    l[a + 1, 0] = 99.0
    c[a - 1, 0] = 105.0
    assert run(tmin, h, l, c, real).sum() == 0
    # on the EDT day the same 14:15 UTC candle is 10:15 NY, inside the entry window
    tmin, h, l, c, real = make_day(D_2021_03_15)
    h[k_utc(13, 30), 0] = 101.0
    l[k_utc(13, 45), 0] = 99.0
    c[a - 1, 0] = 105.0
    assert np.array_equal(run(tmin, h, l, c, real), expected(96, a - 1, k_utc(19, 45), 0.5))


def test_stop_opposite():
    tmin, h, l, c, real = setup_long()
    a = k_utc(13, 30)
    c[a + 11, 0] = 98.5                          # close below L after the long entry
    c[a + 14, 0] = 102.0                         # a later break above H: no re-entry
    assert np.array_equal(run(tmin, h, l, c, real, stop=True), expected(96, a + 6, a + 11, 0.5))
    assert np.array_equal(run(tmin, h, l, c, real, stop=False), expected(96, a + 6, k_utc(19, 45), 0.5))


def test_short_and_first_break_wins():
    tmin, h, l, c, real = make_day(D_2021_03_15)
    a = k_utc(13, 30)
    h[a, 0] = 101.0
    l[a + 1, 0] = 99.0
    c[a + 3, 0] = 98.9                           # first break: below L
    c[a + 5, 0] = 101.5                          # above H later: stop for the short, no reversal
    assert np.array_equal(run(tmin, h, l, c, real, stop=False), expected(96, a + 3, k_utc(19, 45), -0.5))
    assert np.array_equal(run(tmin, h, l, c, real, stop=True), expected(96, a + 3, a + 5, -0.5))


def test_break_on_range_candle_close_does_not_count():
    tmin, h, l, c, real = make_day(D_2021_03_15)
    a = k_utc(13, 30)
    h[a + 1, 0] = 101.0
    c[a + 1, 0] = 100.9
    l[a, 0] = 99.0
    assert run(tmin, h, l, c, real).sum() == 0


def test_entry_cutoff():
    a = k_utc(13, 30)
    for k, traded in ((k_utc(18, 30), True), (k_utc(18, 45), False)):   # 14:30 NY yes, 14:45 NY no
        tmin, h, l, c, real = make_day(D_2021_03_15)
        h[a, 0] = 101.0
        l[a + 1, 0] = 99.0
        c[k, 0] = 101.5
        p = run(tmin, h, l, c, real)
        if traded:
            assert np.array_equal(p, expected(96, k, k_utc(19, 45), 0.5))
        else:
            assert p.sum() == 0


def test_gap_in_range_skips_session():
    tmin, h, l, c, real = setup_long()
    real[k_utc(13, 45), 0] = False
    assert run(tmin, h, l, c, real).sum() == 0


def test_weekend_not_traded():
    tmin, h, l, c, real = setup_long(D_2021_03_20)
    assert run(tmin, h, l, c, real).sum() == 0


def test_range_minutes_changes_output():
    tmin, h, l, c, real = make_day(D_2021_03_15)
    a = k_utc(13, 30)
    h[a, 0] = 100.5
    h[a + 1, 0] = 101.0
    c[a + 1, 0] = 100.6
    l[a + 1, 0] = 99.0
    c[a + 6, 0] = 101.5
    assert np.array_equal(run(tmin, h, l, c, real, rm=15, stop=False), expected(96, a + 1, k_utc(19, 45), 0.5))
    assert np.array_equal(run(tmin, h, l, c, real, rm=30, stop=False), expected(96, a + 6, k_utc(19, 45), 0.5))
    # 60 minutes: range includes the 10:00/10:15 candles; c=101.5 at 11:00 still above H=101
    h[a + 2, 0] = 101.2
    assert np.array_equal(run(tmin, h, l, c, real, rm=60, stop=False), expected(96, a + 6, k_utc(19, 45), 0.5))


def test_utc_midnight_anchor():
    tmin, h, l, c, real = make_day(D_2021_03_15)
    h[0, 0] = 101.0
    l[1, 0] = 99.0
    c[4, 0] = 101.5
    # exit decision after the candle starting at 06:15 UTC (ends 06:30)
    assert np.array_equal(run(tmin, h, l, c, real, anchor="utc_midnight"), expected(96, 4, 25, 0.5))
    assert run(tmin, h, l, c, real, anchor="us_open").sum() == 0
    tmin, h, l, c, real = make_day(D_2021_03_15)
    h[0, 0] = 101.0
    l[1, 0] = 99.0
    c[21, 0] = 101.5                             # starts 05:15 UTC = anchor + 5h15m: too late
    assert run(tmin, h, l, c, real, anchor="utc_midnight").sum() == 0
    with pytest.raises(ValueError):
        run(tmin, h, l, c, real, anchor="london")


class FakeView:
    def __init__(self, days, instruments):
        T = 96 * len(days)
        self.instruments = tuple(instruments)
        self.times = (np.concatenate([d * 1440 + 15 * np.arange(96) for d in days])).astype("datetime64[m]")
        self.bars_per_day = 96
        N = len(instruments)
        self.h = np.full((T, N), 100.1, dtype=np.float32)
        self.l = np.full((T, N), 99.9, dtype=np.float32)
        self.c = np.full((T, N), 100.0, dtype=np.float32)
        self.o = self.c.copy()
        self.real = np.ones((T, N), dtype=bool)
        self.tradable = self.real.copy()
        self.universe = None

    def col(self, name):
        return self.instruments.index(name)


def test_wrapper_columns_and_gross():
    v = FakeView([D_2021_03_15, D_2021_03_15 + 1], ["ETHUSDT", "BTCUSDT"])
    a = k_utc(13, 30)
    for j, sgn in ((0, 1), (1, -1)):
        v.h[a, j] = 101.0
        v.l[a + 1, j] = 99.0
        v.c[a + 6, j] = 101.5 if sgn > 0 else 98.5
    w = target_positions(v, PARAMS)
    assert w.shape == (192, 2)
    assert np.array_equal(w[:96, 0], expected(96, a + 6, k_utc(19, 45), 0.5))
    assert np.array_equal(w[:96, 1], expected(96, a + 6, k_utc(19, 45), -0.5))
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)
    assert np.all(w[96:] == 0)
    v2 = FakeView([D_2021_03_15], ["BTCUSDT"])
    with pytest.raises(Exception):
        target_positions(v2, PARAMS)                # ETHUSDT missing: fail loudly


def test_point_in_time_truncation():
    v = FakeView([D_2021_03_15], ["BTCUSDT", "ETHUSDT"])
    a = k_utc(13, 30)
    v.h[a, 0] = 101.0
    v.l[a + 1, 0] = 99.0
    v.c[a + 6, 0] = 101.5
    v.c[a + 9, 0] = 98.0
    full = target_positions(v, PARAMS)
    for cut in (a + 3, a + 6, a + 8, a + 9, a + 20):
        t = FakeView([D_2021_03_15], ["BTCUSDT", "ETHUSDT"])
        t.times = v.times[: cut + 1]
        t.h, t.l, t.c, t.real = v.h[: cut + 1], v.l[: cut + 1], v.c[: cut + 1], v.real[: cut + 1]
        part = target_positions(t, PARAMS)
        assert np.array_equal(part, full[: cut + 1])
