from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import _ymd, _ymd_days, _nyse_holidays, _calendar_flags, _dow
from strategy import events, diag_calendar_flags as diag_flags


@dataclass
class FakeView:
    dates: Any
    instruments: tuple
    ret_co: Any
    ret_oc: Any
    tradable: Any
    listed: Any
    close: Any
    universe: Any
    asset_class: tuple = ()
    delisting: Any = None
    dollar_volume: Any = None
    extras: dict = field(default_factory=dict)
    series: dict = field(default_factory=dict)
    cash_ret: Any = None

    def col(self, name):
        return self.instruments.index(name)


def sessions(y0, y1):
    """Weekday sessions of the years y0..y1 minus the rule-based NYSE holidays."""
    start = int(_ymd_days(y0, 1, 1))
    end = int(_ymd_days(y1 + 1, 1, 1))
    days = np.arange(start, end).astype("datetime64[D]")
    hol = _nyse_holidays(np.arange(y0 - 1, y1 + 2))
    return days[np.is_busday(days, holidays=hol)]


def make_view(T_years=(2015, 2015), n=10, seed=0, noisy=(0, 1), betas=None, msd=0.01):
    dates = sessions(*T_years)
    T = dates.size
    rng = np.random.default_rng(seed)
    m = rng.normal(0, msd, T)
    if betas is None:
        betas = np.linspace(0.6, 1.5, n)
    sd = np.full(n, 0.004)
    sd[list(noisy)] = 0.04
    ret = betas[None, :] * m[:, None] + rng.normal(0, 1, (T, n)) * sd[None, :]
    ret_co = 0.3 * ret
    ret_oc = (1 + ret) / (1 + ret_co) - 1
    close = 100 * np.cumprod(1 + ret, axis=0)
    ones = np.ones((T, n), bool)
    return FakeView(dates=dates, instruments=tuple("E%d" % i for i in range(n)),
                    ret_co=ret_co, ret_oc=ret_oc, tradable=ones.copy(), listed=ones.copy(),
                    close=close, universe=ones.copy(), asset_class=("us_equity",) * n)


def row(dates, y, m, d):
    hit = np.flatnonzero(dates == _ymd(y, m, d))
    assert hit.size == 1
    return int(hit[0])


# ---------------------------------------------------------------- calendar

def test_holidays_rules():
    hol = set(_nyse_holidays(np.arange(2014, 2025)).tolist())
    assert _ymd(2015, 4, 3).item() in hol           # Good Friday 2015
    assert _ymd(2024, 3, 29).item() in hol          # Good Friday 2024
    assert _ymd(2019, 4, 19).item() in hol          # Good Friday 2019
    assert _ymd(2015, 7, 3).item() in hol           # July 4 on Saturday -> Friday
    assert _ymd(2021, 7, 5).item() in hol           # July 4 on Sunday -> Monday
    assert _ymd(2021, 12, 31).item() not in hol     # New Year on Saturday: no closure
    assert _ymd(2022, 6, 20).item() in hol          # Juneteenth observed Monday
    assert _ymd(2021, 6, 18).item() not in hol      # no Juneteenth before 2022
    assert _ymd(2015, 1, 19).item() in hol          # MLK
    assert _ymd(2015, 5, 25).item() in hol          # Memorial Day
    assert _ymd(2015, 9, 7).item() in hol           # Labor Day
    assert _ymd(2015, 11, 26).item() in hol         # Thanksgiving
    assert _ymd(2015, 11, 27).item() not in hol     # day after Thanksgiving is a session
    assert all(_dow(np.array(sorted(hol), dtype="datetime64[D]")) < 5)


def test_calendar_flags_2015():
    dates = sessions(2015, 2015)
    a, b = _calendar_flags(dates)
    # normal week: Thursday -> A, Friday -> B, others nothing
    assert a[row(dates, 2015, 3, 26)] and not b[row(dates, 2015, 3, 26)]
    assert b[row(dates, 2015, 3, 27)] and not a[row(dates, 2015, 3, 27)]
    for d in (23, 24, 25):
        assert not a[row(dates, 2015, 3, d)] and not b[row(dates, 2015, 3, d)]
    # Good Friday: F = Thursday, so A after Wednesday, B after Thursday
    assert a[row(dates, 2015, 4, 1)]
    assert b[row(dates, 2015, 4, 2)]
    assert not a[row(dates, 2015, 4, 2)]
    # MLK Monday: B after Friday (M = Tuesday), Tuesday row is flat (decision C)
    assert b[row(dates, 2015, 1, 16)]
    assert not a[row(dates, 2015, 1, 20)] and not b[row(dates, 2015, 1, 20)]
    # Thanksgiving: Friday is a session, so A after Wednesday
    assert a[row(dates, 2015, 11, 25)]
    assert not a[row(dates, 2015, 11, 24)]
    # July 3 holiday (Friday)
    assert a[row(dates, 2015, 7, 1)] and b[row(dates, 2015, 7, 2)]
    # counts: one A and one B per week
    assert abs(a.sum() - 52) <= 1 and abs(b.sum() - 52) <= 1
    assert not np.any(a & b)
    da, db = diag_flags(dates)
    assert np.array_equal(da, a) and np.array_equal(db, b)


def test_one_session_week_skipped():
    # custom holidays: Monday-Thursday of the week of 2015-03-09 closed -> Friday 13th is the only session
    hol = _ymd(2015, 3, np.array([9, 10, 11, 12]))
    days = (int(_ymd_days(2015, 3, 1)) + np.arange(31)).astype("datetime64[D]")
    dates = days[np.is_busday(days, holidays=hol)]
    a, b = _calendar_flags(dates, holidays=hol)
    fri6 = row(dates, 2015, 3, 6)
    fri13 = row(dates, 2015, 3, 13)
    assert not a[fri6] and not b[fri6]       # next session is M and F at once: skipped
    assert b[fri13]                          # next week's M book is still formed after its close
    assert a[row(dates, 2015, 3, 5)]         # previous week normal


def test_calendar_truncation_invariant():
    dates = sessions(2014, 2016)
    a, b = _calendar_flags(dates)
    for cut in (10, 250, 500, dates.size - 3):
        a2, b2 = _calendar_flags(dates[:cut])
        assert np.array_equal(a2, a[:cut]) and np.array_equal(b2, b[:cut])


# ---------------------------------------------------------------- weights

def ols_beta(y, x):
    X = np.column_stack([np.ones_like(x), x])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return coef[1]


def test_books_on_known_rows():
    v = make_view()
    p = {"ivol_lookback_days": 20, "leg_quantile": 0.2}
    w = target_weights(v, p)
    a_rows, b_rows = _calendar_flags(v.dates)
    t = row(v.dates, 2015, 3, 26)             # decision A
    assert a_rows[t]
    S = np.flatnonzero(w[t] > 0)
    H = np.flatnonzero(w[t] < 0)
    assert list(S) == [0, 1]                  # noisy stocks = speculative leg (2 of 10)
    assert H.size == 8
    # independent beta check
    r = (1 + v.ret_co) * (1 + v.ret_oc) - 1
    win = r[t - 19:t + 1]
    m = win.mean(axis=1)
    betas = np.array([ols_beta(win[:, j], m) for j in range(10)])
    bS, bH = betas[S].mean(), betas[H].mean()
    a = bH / (bS + bH)
    b = bS / (bS + bH)
    assert np.allclose(w[t, S], a / 2)
    assert np.allclose(w[t, H], -b / 8)
    assert np.isclose(np.abs(w[t]).sum(), 1.0)
    # decision B on Friday: mirror book
    tb = row(v.dates, 2015, 3, 27)
    assert np.all(w[tb, [0, 1]] < 0) and np.all(np.delete(w[tb], [0, 1]) > 0)
    assert np.isclose(np.abs(w[tb]).sum(), 1.0)
    # all other rows flat
    other = ~(a_rows | b_rows)
    assert np.all(w[other] == 0)
    assert np.all(np.abs(w).sum(axis=1) <= 1 + 1e-12)
    assert not np.any(np.isnan(w))


def test_rows_before_lookback_flat():
    v = make_view()
    p = {"ivol_lookback_days": 20, "leg_quantile": 0.2}
    w = target_weights(v, p)
    assert np.all(w[:15] == 0)                # need >= 16 valid returns (80 % of 20)
    # 16th valid return: row 16 (row 0 has no previous close)
    a_rows, b_rows = _calendar_flags(v.dates)
    first = np.flatnonzero((a_rows | b_rows) & (np.arange(v.dates.size) >= 16))[0]
    assert np.abs(w[first]).sum() > 0.99


def test_eligibility_universe_tradable_missing():
    v = make_view()
    p = {"ivol_lookback_days": 20, "leg_quantile": 0.2}
    t = row(v.dates, 2015, 3, 26)
    v.universe[t, 5] = False                  # not a member at t
    v.tradable[t, 6] = False                  # not tradable at t
    v.close[t - 10:t - 5, 7] = np.nan         # 6 invalid returns of 20 -> < 80 %
    v.close[t - 3, 8] = np.nan                # 2 invalid returns of 20 -> still eligible
    w = target_weights(v, p)
    assert w[t, 5] == 0 and w[t, 6] == 0 and w[t, 7] == 0
    assert w[t, 8] != 0
    assert np.isclose(np.abs(w[t]).sum(), 1.0)
    # removal is for that decision only
    tb = row(v.dates, 2015, 3, 27)
    assert w[tb, 5] != 0 and w[tb, 6] != 0


def test_negative_beta_fallback():
    betas = np.array([-1.5, -1.2, 1, 1, 1, 1, 1, 1, 1, 1.0])
    v = make_view(betas=betas, msd=0.03)
    p = {"ivol_lookback_days": 20, "leg_quantile": 0.2}
    w = target_weights(v, p)
    t = row(v.dates, 2015, 3, 26)
    assert np.allclose(w[t, [0, 1]], 0.25)
    assert np.allclose(w[t, 2:], -0.5 / 8)


def test_params_change_output():
    v = make_view(n=20, noisy=(0, 1, 2, 3, 4, 5))
    t = row(v.dates, 2015, 6, 25)
    w1 = target_weights(v, PARAMS)
    w2 = target_weights(v, dict(PARAMS, leg_quantile=0.3))
    assert (w1[t] > 0).sum() == 4 and (w2[t] > 0).sum() == 6
    w3 = target_weights(v, dict(PARAMS, ivol_lookback_days=126))
    first1 = np.flatnonzero(np.abs(w1).sum(axis=1) > 0)[0]
    first3 = np.flatnonzero(np.abs(w3).sum(axis=1) > 0)[0]
    assert first3 > first1
    assert not np.allclose(w1[t], w3[t])


def test_point_in_time_truncation():
    v = make_view(T_years=(2014, 2015))
    p = {"ivol_lookback_days": 42, "leg_quantile": 0.2}
    w = target_weights(v, p)
    for cut in (120, 301, 400):
        vt = FakeView(dates=v.dates[:cut], instruments=v.instruments, ret_co=v.ret_co[:cut],
                      ret_oc=v.ret_oc[:cut], tradable=v.tradable[:cut], listed=v.listed[:cut],
                      close=v.close[:cut], universe=v.universe[:cut])
        assert np.array_equal(target_weights(vt, p), w[:cut])


def test_missing_universe_raises():
    v = make_view()
    v.universe = None
    with pytest.raises(ValueError):
        target_weights(v, PARAMS)


def test_diagnostic_matches_speculative_leg():
    v = make_view()
    p = {"ivol_lookback_days": 20, "leg_quantile": 0.2}
    w = target_weights(v, p)
    mask, side = events(v, p)
    a_rows, b_rows = _calendar_flags(v.dates)
    assert np.array_equal(mask[a_rows], w[a_rows] > 0)
    assert np.array_equal(mask[b_rows], w[b_rows] < 0)
    assert np.all(side[a_rows][mask[a_rows]] == 1)
    assert np.all(side[b_rows][mask[b_rows]] == -1)
    assert not mask[~(a_rows | b_rows)].any()
