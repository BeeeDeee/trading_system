import numpy as np
import pytest
from dataclasses import dataclass

from strategy import target_positions, PARAMS, ny_utc_offset_minutes

# day numbers since 1970-01-01, derived by hand:
# 2021-01-01 = 51*365 + 13 leap days = 18628 (Friday)
JAN04_2021 = 18631      # Monday, EST
FEB15_2021 = 18673      # Monday, EST
MAR14_2021 = 18700      # Sunday, DST starts (second Sunday of March)
MAR15_2021 = 18701      # Monday, EDT
EPS = 0.05


@dataclass
class FakeView:
    times: np.ndarray
    instruments: tuple
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    real: np.ndarray
    tradable: np.ndarray
    universe: object
    bars_per_day: int = 96

    def col(self, name):
        return self.instruments.index(name)


def hand_offset(day):
    # hand-derived for the test windows (Jan-Mar 2021)
    return 240 if day >= MAR14_2021 else 300


def weekdays(start, ndays):
    return [d for d in range(start, start + ndays) if (d + 3) % 7 < 5]


def build(start, ndays, r_by_day, breakout=None, unreal=None):
    """Price path: 100 everywhere except from the 09:15 ET candle to the 15:30 ET candle of each
    weekday session, where it is 100*exp(r). breakout = (day, inst, scan_index, side) puts the close
    of that scan candle (and later scan candles) at P + side*1. unreal = list of (day, minute_utc, inst)."""
    T = ndays * 96
    tm = start * 1440 + 15 * np.arange(T)
    px = np.full((T, 2), 100.0)
    for d, r in r_by_day.items():
        t930 = d * 1440 + 570 + hand_offset(d)
        a = (t930 - 15 - start * 1440) // 15
        b = (t930 + 375 - start * 1440) // 15
        px[a:b, :] = 100.0 * np.exp(r)
    if breakout is not None:
        d, j, si, side = breakout
        t930 = d * 1440 + 570 + hand_offset(d)
        a = (t930 + 30 + 15 * si - start * 1440) // 15
        b = (t930 + 375 - start * 1440) // 15
        px[a:b, j] = px[a, j] + side * 1.0
    c = px.copy()
    o = np.vstack([px[:1], px[:-1]])
    h = np.maximum(o, c) + EPS
    lo = np.minimum(o, c) - EPS
    real = np.ones((T, 2), dtype=bool)
    if unreal:
        for d_min, j in unreal:
            real[(d_min - start * 1440) // 15, j] = False
    return FakeView(times=tm.astype("datetime64[m]"), instruments=("BTCUSDT", "ETHUSDT"),
                    o=o, h=h, l=lo, c=c, real=real, tradable=real.copy(), universe=None)


def alt_history(days):
    return {d: (0.01 if i % 2 == 0 else -0.01) for i, d in enumerate(days)}


def row_of(start, utc_minute):
    return (utc_minute - start * 1440) // 15


def jan_case(r_test=0.02, side=1.0, si=3, inst=0, **kw):
    days = weekdays(JAN04_2021, 29)
    assert len(days) == 21
    hist = alt_history(days[:20])
    test_day = days[20]
    hist[test_day] = r_test
    v = build(JAN04_2021, 29, hist, breakout=(test_day, inst, si, side), **kw)
    return v, test_day


def test_dst_offsets():
    off = ny_utc_offset_minutes(np.array([JAN04_2021, MAR14_2021 - 1, MAR14_2021, MAR15_2021,
                                          18938, 18939]))
    # 2021-11-06 = 18937 + ... : 2021-11-07 (first Sunday of November) = 18938
    assert list(off) == [300, 300, 240, 240, 300, 300]
    # 2021-11-06 (Saturday) is still EDT
    assert int(ny_utc_offset_minutes(np.array([18937]))[0]) == 240


def test_aligned_long_breakout_hand_rows():
    v, d = jan_case()
    w = target_positions(v, PARAMS)
    t930 = d * 1440 + 570 + 300                       # 14:30 UTC in winter
    b = row_of(JAN04_2021, t930 + 30 + 15 * 3)         # breakout candle opens 10:45 ET
    e = row_of(JAN04_2021, t930 + 375)                 # candle opening 15:45 ET -> exit decision
    assert np.all(w[b:e, 0] == 0.5)
    assert w[b - 1, 0] == 0.0 and w[e, 0] == 0.0
    assert np.count_nonzero(w[:, 0]) == e - b
    assert np.count_nonzero(w[:, 1]) == 0              # ETH flat, no breakout
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)


def test_side_modes_partition():
    v, d = jan_case()
    w_with = target_positions(v, PARAMS, side_mode="with")
    w_against = target_positions(v, PARAMS, side_mode="against")
    w_all = target_positions(v, PARAMS, side_mode="all")
    assert np.count_nonzero(w_with) > 0
    assert np.count_nonzero(w_against) == 0
    assert np.array_equal(w_all, w_with)
    # breakout against a positive pre-session move: only 'against' and 'all' trade, short
    v2, _ = jan_case(side=-1.0)
    assert np.count_nonzero(target_positions(v2, PARAMS, side_mode="with")) == 0
    wa = target_positions(v2, PARAMS, side_mode="against")
    assert wa.min() == -0.5 and wa.max() == 0.0
    assert np.array_equal(wa, target_positions(v2, PARAMS, side_mode="all"))
    with pytest.raises(ValueError):
        target_positions(v, PARAMS, side_mode="both")


def test_k_threshold():
    # s = sqrt(20 * 0.01^2 / 19) = 0.010260; r = 0.004 -> z = 0.39
    v, _ = jan_case(r_test=0.004)
    assert np.count_nonzero(target_positions(v, {"k": 0.25})) > 0
    assert np.count_nonzero(target_positions(v, {"k": 0.5})) == 0
    assert np.count_nonzero(target_positions(v, {"k": 0.0})) > 0
    # all mode ignores k
    assert np.count_nonzero(target_positions(v, {"k": 0.5}, side_mode="all")) > 0


def test_short_with_negative_premove():
    v, d = jan_case(r_test=-0.02, side=-1.0, inst=1, si=0)
    w = target_positions(v, PARAMS)
    t930 = d * 1440 + 570 + 300
    b = row_of(JAN04_2021, t930 + 30)
    e = row_of(JAN04_2021, t930 + 375)
    assert np.all(w[b:e, 1] == -0.5)
    assert w[b - 1, 1] == 0.0 and w[e, 1] == 0.0
    assert np.count_nonzero(w[:, 0]) == 0


def test_last_scan_candle_holds_one_bar():
    v, d = jan_case(si=22)                              # breakout closes 15:45 ET
    w = target_positions(v, PARAMS)
    t930 = d * 1440 + 570 + 300
    b = row_of(JAN04_2021, t930 + 30 + 15 * 22)
    assert np.count_nonzero(w[:, 0]) == 1 and w[b, 0] == 0.5


def test_only_first_breakout_counts():
    # first breakout is below (against the positive pre-move), a later one above: no trade in 'with'
    days = weekdays(JAN04_2021, 29)
    hist = alt_history(days[:20])
    d = days[20]
    hist[d] = 0.02
    v = build(JAN04_2021, 29, hist, breakout=(d, 0, 2, -1.0))
    t930 = d * 1440 + 570 + 300
    a = row_of(JAN04_2021, t930 + 30 + 15 * 6)
    e = row_of(JAN04_2021, t930 + 375)
    v.c[a:e, 0] = v.c[a - 1, 0] + 3.0                  # well above the range later
    v.o[1:, 0] = v.c[:-1, 0]
    v.h[:, 0] = np.maximum(v.o[:, 0], v.c[:, 0]) + EPS
    v.l[:, 0] = np.minimum(v.o[:, 0], v.c[:, 0]) - EPS
    assert np.count_nonzero(target_positions(v, PARAMS)) == 0
    assert target_positions(v, PARAMS, side_mode="against").min() == -0.5


def test_missing_candles_block_session():
    _, d = jan_case()
    t930 = d * 1440 + 570 + 300
    for m in (t930, t930 + 15, d * 1440):
        v, _ = jan_case(unreal=[(m, 0)])
        assert np.count_nonzero(target_positions(v, PARAMS)[:, 0]) == 0


def test_too_little_history():
    days = weekdays(JAN04_2021, 29)
    hist = alt_history(days[:20])
    d = days[20]
    hist[d] = 0.02
    # knock out 6 of the 20 previous sessions' 00:00 UTC candles -> 14 valid < 15
    unreal = [(dd * 1440, 0) for dd in days[14:20]]
    v = build(JAN04_2021, 29, hist, breakout=(d, 0, 3, 1.0), unreal=unreal)
    assert np.count_nonzero(target_positions(v, PARAMS)[:, 0]) == 0
    # 5 knocked out -> 15 valid: trades
    unreal = [(dd * 1440, 0) for dd in days[15:20]]
    v = build(JAN04_2021, 29, hist, breakout=(d, 0, 3, 1.0), unreal=unreal)
    assert np.count_nonzero(target_positions(v, PARAMS)[:, 0]) > 0


def test_dst_session_times():
    days = weekdays(FEB15_2021, 29)
    assert days[20] == MAR15_2021
    hist = alt_history(days[:20])
    hist[MAR15_2021] = 0.02
    v = build(FEB15_2021, 29, hist, breakout=(MAR15_2021, 0, 3, 1.0))
    w = target_positions(v, PARAMS)
    t930 = MAR15_2021 * 1440 + 570 + 240               # 13:30 UTC in summer time
    b = row_of(FEB15_2021, t930 + 30 + 45)
    e = row_of(FEB15_2021, t930 + 375)
    assert np.all(w[b:e, 0] == 0.5)
    assert np.count_nonzero(w[:, 0]) == e - b


def test_point_in_time_truncation():
    v, d = jan_case()
    full = target_positions(v, PARAMS)
    T = len(v.times)
    for cut in (T - 1, T - 20, T - 40, T - 60, T - 200):
        tv = FakeView(times=v.times[:cut], instruments=v.instruments, o=v.o[:cut], h=v.h[:cut],
                      l=v.l[:cut], c=v.c[:cut], real=v.real[:cut], tradable=v.tradable[:cut],
                      universe=None)
        assert np.array_equal(target_positions(tv, PARAMS), full[:cut])


def test_universe_mask():
    v, _ = jan_case()
    v.universe = np.ones(v.real.shape, dtype=bool)
    v.universe[:, 0] = False
    assert np.count_nonzero(target_positions(v, PARAMS)) == 0
