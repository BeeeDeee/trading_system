import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS
from strategy import decision_days, signals, _civil_from_days, _easter


# --------------------------------------------------------------------------- helpers

def weekday_dates(start_day, n):
    """n consecutive weekdays (Mon-Fri) starting at or after integer day start_day (days since epoch)."""
    days = np.arange(start_day, start_day + 2 * n + 10)
    days = days[(days + 3) % 7 < 5][:n]
    return days.astype("datetime64[D]")


def friday_ending_dates(n, start_day=17000):
    """n weekdays whose last row is a Friday (no scheduled holidays in this stretch of 2016-2017)."""
    d = weekday_dates(start_day, n + 10)
    wd = (d.astype(np.int64) + 3) % 7
    last_fri = np.flatnonzero(wd == 4)[-1]
    return d[last_fri - n + 1: last_fri + 1]


def make_view(ret_oc, dates=None, tradable=None, listed=None, universe=None, names=None):
    T, N = ret_oc.shape
    if dates is None:
        dates = friday_ending_dates(T)
    if names is None:
        names = tuple("E%d" % (1000 + i) for i in range(N))
    if tradable is None:
        tradable = np.ones((T, N), dtype=bool)
    if listed is None:
        listed = np.ones((T, N), dtype=bool)
    close = np.where(listed, 10.0, np.nan)
    return DataView(
        dates=dates,
        instruments=tuple(names),
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=np.zeros((T, N)),
        ret_oc=ret_oc,
        tradable=tradable,
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.where(listed, 1e6, np.nan),
        universe=universe,
    )


SMALL = {"fast_days": 2, "slow_days": 25}   # slow window = rows t-24 .. t-21 (4 rows)


def known_book_returns(N=100, T=30):
    """IR_slow_i = i/100 (one day at t-22), IR_fast_i = (i % 4)/100 (one day at t)."""
    r = np.zeros((T, N))
    i = np.arange(N)
    r[T - 1 - 22] = np.expm1(i / 100.0)
    r[T - 1] = np.expm1((i % 4) / 100.0)
    return r


# --------------------------------------------------------------------------- calendar

def test_civil_from_days_matches_numpy():
    days = np.arange(9000, 21000)
    y, m, d = _civil_from_days(days)
    dt = days.astype("datetime64[D]")
    ny = dt.astype("datetime64[Y]").astype(np.int64) + 1970
    nm = dt.astype("datetime64[M]").astype(np.int64) % 12 + 1
    nd = (dt - dt.astype("datetime64[M]")).astype(np.int64) + 1
    assert np.array_equal(y, ny) and np.array_equal(m, nm) and np.array_equal(d, nd)


def test_easter_known_years():
    years = np.array([2000, 2008, 2011, 2015, 2019, 2024])
    month, day = _easter(years)
    assert list(zip(month.tolist(), day.tolist())) == [(4, 23), (3, 23), (4, 24), (4, 5), (4, 21), (3, 31)]


def test_decision_days_2015():
    # all weekdays of 2015 (2015-01-01 is day 16436, a Thursday)
    d = weekday_dates(16436, 400)
    y, _, _ = _civil_from_days(d.astype(np.int64))
    d = d[y == 2015]
    dec = decision_days(d)
    wd = (d.astype(np.int64) + 3) % 7
    assert dec[wd == 4].all()                      # every Friday
    assert not dec[wd < 3].any()                   # never Mon-Wed
    _, m, dd = _civil_from_days(d[dec & (wd == 3)].astype(np.int64))
    # Thursday before Good Friday (Apr 3), before observed Jul 3, before Christmas (Fri Dec 25),
    # before New Year (Fri 2016-01-01)
    assert list(zip(m.tolist(), dd.tolist())) == [(4, 2), (7, 2), (12, 24), (12, 31)]


def test_decision_days_observed_holidays():
    # 2010: Good Friday Apr 2; Jul 4 is a Sunday (Monday observed, no effect);
    #       Dec 25 is a Saturday -> Fri Dec 24 closed -> Thu Dec 23 decides
    # 2020: Good Friday Apr 10; Jul 4 is a Saturday -> Fri Jul 3 closed -> Thu Jul 2;
    #       Christmas Fri Dec 25; New Year Fri 2021-01-01
    for start, year, expected in [(14610, 2010, [(4, 1), (12, 23)]),
                                  (18262, 2020, [(4, 9), (7, 2), (12, 24), (12, 31)])]:
        d = weekday_dates(start, 400)
        y, _, _ = _civil_from_days(d.astype(np.int64))
        d = d[y == year]
        dec = decision_days(d)
        wd = (d.astype(np.int64) + 3) % 7
        _, m, dd = _civil_from_days(d[dec & (wd == 3)].astype(np.int64))
        assert list(zip(m.tolist(), dd.tolist())) == expected


# --------------------------------------------------------------------------- construction

def test_known_book():
    r = known_book_returns()
    w = target_weights(make_view(r), SMALL)
    last = w[-1]
    # W = ids 50..99 (top IR_slow); lowest IR_fast = i%4==0 -> 52,56,...,96 (12 names), first 10 by id
    longs = [52, 56, 60, 64, 68, 72, 76, 80, 84, 88]
    # L = ids 0..49; highest IR_fast = i%4==3 -> 3,7,...,47 (12 names), first 10 by id
    shorts = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
    exp = np.zeros(100)
    exp[longs] = 0.05
    exp[shorts] = -0.05
    assert np.allclose(last, exp)
    assert abs(np.abs(last).sum() - 1.0) < 1e-12 and abs(last.sum()) < 1e-12


def test_ties_follow_instrument_id_not_column_order():
    r = known_book_returns()
    names = ["E%d" % (1000 + i) for i in range(100)]
    names[52], names[92] = names[92], names[52]    # column 52 now has the larger id E1092
    w = target_weights(make_view(r, names=names), SMALL)[-1]
    assert w[52] == 0.0 and w[92] == pytest.approx(0.05)


def test_non_decision_rows_nan_and_warmup_flat():
    r = known_book_returns()
    view = make_view(r)
    w = target_weights(view, SMALL)
    dec = decision_days(view.dates)
    assert np.isnan(w[~dec]).all()
    assert not np.isnan(w[dec]).any()
    # decision rows before slow_days of history: flat
    early = np.flatnonzero(dec)
    early = early[early < SMALL["slow_days"] - 1]
    assert early.size > 0 and np.all(w[early] == 0.0)


def test_min_leg_size_flat():
    r = known_book_returns(N=99)       # half = 49, round(9.8) = 10 -> trades
    assert np.count_nonzero(target_weights(make_view(r), SMALL)[-1]) == 20
    r = known_book_returns(N=95)       # half = 47, round(9.4) = 9 < 10 -> flat
    assert np.all(target_weights(make_view(r), SMALL)[-1] == 0.0)


def test_leg_rounding_and_odd_median():
    r = known_book_returns(N=107)      # half = 53 (median name 53 in neither), round(10.6) = 11
    w = target_weights(make_view(r), SMALL)[-1]
    assert np.count_nonzero(w > 0) == 11 and np.count_nonzero(w < 0) == 11
    assert w[53] == 0.0
    assert np.allclose(w[w > 0], 0.5 / 11) and np.allclose(w[w < 0], -0.5 / 11)
    assert np.all(np.flatnonzero(w > 0) >= 54) and np.all(np.flatnonzero(w < 0) <= 52)


def test_eligibility_filters():
    T, N = 30, 110
    r = known_book_returns(N=N, T=T)
    t = T - 1
    tradable = np.ones((T, N), dtype=bool)
    listed = np.ones((T, N), dtype=bool)
    universe = np.ones((T, N), dtype=bool)
    universe[t, 100] = False                  # not a member at t
    tradable[t, 101] = False                  # not tradable at t
    tradable[t - 1, 102] = False              # 1 of 2 fast days missing (< 90 %)
    tradable[t - 23, 103] = False             # 3 of 4 slow days (< 90 %)
    listed[:6, 104] = False                   # listed for 24 < slow_days rows
    r[:6, 104] = 0.0
    view = make_view(r, tradable=tradable, listed=listed, universe=universe)
    ir_fast, ir_slow, elig = signals(view, SMALL)
    assert not elig[t, 100:105].any()
    assert elig[t, :100].all() and elig[t, 105:].all()
    # 105 eligible -> half 52, round(10.4) = 10 per leg
    w = target_weights(view, SMALL)[-1]
    assert np.count_nonzero(w > 0) == 10 and np.count_nonzero(w < 0) == 10
    assert np.all(w[100:105] == 0.0)


def test_signal_sums_and_missing_days_zero():
    T, N = 30, 3
    r = np.zeros((T, N))
    t = T - 1
    r[t, 0] = 0.10
    r[t - 1, 0] = -0.05
    r[t - 2, 0] = 0.50                        # outside the fast window (fast = 2)
    r[t - 21, 1] = 0.20                       # inside the slow window (t-24 .. t-21)
    r[t - 20, 1] = 0.30                       # skipped month: not in IR_slow
    r[t - 25, 1] = 0.40                       # before the slow window
    tradable = np.ones((T, N), dtype=bool)
    r[t - 22, 2] = 0.70
    tradable[t - 22, 2] = False               # invalid day counts as 0
    ir_fast, ir_slow, _ = signals(make_view(r, tradable=tradable), SMALL)
    assert ir_fast[t, 0] == pytest.approx(np.log(1.10) + np.log(0.95))
    assert ir_slow[t, 1] == pytest.approx(np.log(1.20))
    assert ir_slow[t, 2] == 0.0
    assert np.isnan(ir_slow[: 25 - 1]).all()  # warm-up rows


def random_view(T=700, N=150, seed=1):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.02, size=(T, N))
    return make_view(r, dates=weekday_dates(16000, T))


def test_params_change_output_and_gross():
    view = random_view()
    base = target_weights(view, PARAMS)
    dec = ~np.isnan(base).all(axis=1)
    assert np.all(np.abs(np.nan_to_num(base)).sum(axis=1) <= 1.0 + 1e-12)
    active = dec & (np.abs(np.nan_to_num(base)).sum(axis=1) > 0)
    assert active.sum() > 50
    assert np.allclose(np.nansum(base[active], axis=1), 0.0)
    for key, val in [("fast_days", 3), ("fast_days", 10), ("slow_days", 126), ("slow_days", 504)]:
        p = dict(PARAMS)
        p[key] = val
        other = target_weights(view, p)
        assert not np.array_equal(np.nan_to_num(other, nan=9.0), np.nan_to_num(base, nan=9.0)), key


def test_point_in_time_truncation():
    view = random_view(T=400)
    full = target_weights(view, {"fast_days": 5, "slow_days": 126})
    cut = 300
    short = make_view(np.asarray(view.ret_oc)[:cut], dates=view.dates[:cut])
    part = target_weights(short, {"fast_days": 5, "slow_days": 126})
    assert np.array_equal(np.nan_to_num(part, nan=9.0), np.nan_to_num(full[:cut], nan=9.0))
