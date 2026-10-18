from dataclasses import replace

import numpy as np
import pytest

from strategy import (target_weights, PARAMS, decision_rows, monthly_relative_returns, seasonal_score,
                      book_row, month_ids)
from lab.framework.data import synthetic

DAY_2004_01_01 = 12418          # days since 1970-01-01 (34 * 365 + 8 leap days)


def weekdays(start_day, n_days):
    d = (np.arange(n_days) + start_day).astype("datetime64[D]")
    return d[np.is_busday(d)]


# ---------------------------------------------------------------- calendar
def test_decision_rows_last_weekday_of_each_month():
    d = weekdays(DAY_2004_01_01, 366)      # all of 2004 (leap year), weekdays only
    rows, target = decision_rows(d)
    mid = month_ids(d)
    last_of_month = np.flatnonzero(np.r_[mid[1:] != mid[:-1], False])
    # 2004-12-31 is a Friday: the next weekday is in January, so it is a decision too
    expected = np.r_[last_of_month, len(d) - 1]
    assert np.array_equal(rows, expected)
    assert np.array_equal(target, mid[rows] + 1)
    assert len(rows) == 12


def test_decision_rows_holiday_on_last_weekday_falls_back_to_first_session():
    d = weekdays(DAY_2004_01_01, 366)
    mid = month_ids(d)
    # drop the last weekday of March (a Good-Friday-like holiday on the month's last weekday)
    march = np.flatnonzero(mid == mid[0] + 2)
    keep = np.ones(len(d), dtype=bool)
    keep[march[-1]] = False
    d2 = d[keep]
    rows, target = decision_rows(d2)
    mid2 = month_ids(d2)
    first_april = np.flatnonzero(mid2 == mid2[0] + 3)[0]
    assert first_april in rows
    assert target[list(rows).index(first_april)] == mid2[first_april]      # targets April itself
    assert not np.any(np.isin(rows, np.flatnonzero(mid2 == mid2[0] + 2)))  # no decision in March
    assert len(rows) == 12


def test_decision_rows_point_in_time_under_truncation():
    d = weekdays(DAY_2004_01_01, 366)
    rows, target = decision_rows(d)
    for cut in (20, 41, 42, 100, 200):
        r2, t2 = decision_rows(d[:cut])
        assert np.array_equal(r2, rows[rows < cut])
        assert np.array_equal(t2, target[rows < cut])


# ---------------------------------------------------------------- monthly returns
def test_monthly_relative_returns_by_hand():
    d = weekdays(DAY_2004_01_01, 60)       # January and February 2004
    T = len(d)
    mid = month_ids(d)
    jan = np.flatnonzero(mid == mid[0])
    rco = np.zeros((T, 3))
    roc = np.zeros((T, 3))
    valid = np.ones((T, 3), dtype=bool)
    rco[jan[3], 0] = 0.10
    roc[jan[3], 0] = 0.10                  # (1.1 * 1.1) - 1 = 0.21 for stock 0 in January
    roc[jan[5], 1] = -0.05                 # stock 1: -0.05
    rco[jan[6], 2] = 0.50                  # stock 2 has a return on 10 January sessions only
    valid[jan[10:], 2] = False
    first, R = monthly_relative_returns(d, rco, roc, valid)
    assert first == mid[0]
    assert R.shape == (2, 3)
    mean = (0.21 - 0.05) / 2
    assert R[0, 0] == pytest.approx(0.21 - mean)
    assert R[0, 1] == pytest.approx(-0.05 - mean)
    assert np.isnan(R[0, 2])               # fewer than 15 sessions: does not count
    assert np.allclose(R[1], 0.0)          # February: all zero returns, all count


def test_monthly_invalid_sessions_do_not_compound():
    d = weekdays(DAY_2004_01_01, 31)
    T = len(d)
    rco = np.zeros((T, 2))
    roc = np.zeros((T, 2))
    valid = np.ones((T, 2), dtype=bool)
    rco[0, 0] = 0.30
    valid[0, 0] = False                    # a return on a session that does not count is ignored
    rco[1, 1] = np.nan                     # a NaN return is a missing session, not a zero
    first, R = monthly_relative_returns(d, rco, roc, valid)
    assert np.allclose(R[0], 0.0)


# ---------------------------------------------------------------- score
def test_seasonal_score_by_hand():
    # target month 30, Y = 2: annual lags 12, 24 -> months 18, 6; other lags 13..23 -> months 17..7
    R = np.zeros((30, 4))
    R[18, 0] = 0.04
    R[6, 0] = 0.02                         # annual mean 0.03
    R[7:18, 0] = 0.01                      # other mean 0.01 -> score 0.02
    R[12, 0] = 0.5                         # month 12 = lag 18: other lag, include it
    oth0 = (0.01 * 10 + 0.5) / 11
    R[:, 1] = 0.0
    R[18, 1] = np.nan                      # one annual lag missing -> no score
    R[7:10, 2] = np.nan                    # 3 of 11 other lags missing -> 8/11 < 75 % -> no score
    R[7:9, 3] = np.nan                     # 2 of 11 missing -> 9/11 >= 75 % -> scored
    R[5, :] = 9.0                          # month 5 = lag 25: outside the window, never used
    R[19:, :] = 9.0                        # lags 1..11 (the last year) never used
    s = seasonal_score(R, 0, 30, 2)
    assert s[0] == pytest.approx(0.03 - oth0)
    assert np.isnan(s[1])
    assert np.isnan(s[2])
    assert s[3] == pytest.approx(0.0)


def test_seasonal_score_needs_full_history():
    R = np.zeros((40, 2))
    first = 100
    # target 100 + 40: Y = 3 needs month 140 - 36 = 104 (exists); Y = 4 needs 92 (before the data)
    assert np.all(np.isfinite(seasonal_score(R, first, 140, 3)))
    assert np.all(np.isnan(seasonal_score(R, first, 140, 4)))


def test_years_back_changes_the_lags():
    rng = np.random.default_rng(1)
    R = rng.normal(size=(120, 50))
    s3 = seasonal_score(R, 0, 110, 3)
    s5 = seasonal_score(R, 0, 110, 5)
    assert np.all(np.isfinite(s3)) and np.all(np.isfinite(s5))
    assert not np.allclose(s3, s5)


# ---------------------------------------------------------------- book
def test_book_row_legs():
    score = np.arange(60, dtype=float)
    elig = np.ones(60, dtype=bool)
    elig[:10] = False                      # 50 eligible: 10 per leg at 0.2
    score[59] = np.nan                     # no score: not eligible -> 49 eligible
    w = book_row(score, elig, 0.2)
    assert np.allclose(w, 0.0)             # floor(0.2 * 49) = 9 < 10: hold nothing
    score[59] = 59.0
    w = book_row(score, elig, 0.2)
    assert np.allclose(w[50:60], 0.05)
    assert np.allclose(w[10:20], -0.05)
    assert np.allclose(w[20:50], 0.0) and np.allclose(w[:10], 0.0)
    assert np.abs(w).sum() == pytest.approx(1.0) and w.sum() == pytest.approx(0.0)
    w3 = book_row(score, elig, 0.3)
    assert (w3 > 0).sum() == 15 and (w3 < 0).sum() == 15
    w1 = book_row(score, elig, 0.1)
    assert np.allclose(w1, 0.0)            # 5 per leg < 10


# ---------------------------------------------------------------- end to end
# synthetic() has ~31 instruments and no universe: give it full membership, and use leg quantiles large
# enough for the card's minimum of 10 names per leg (0.2 * 31 = 6 would hold nothing).
P_E2E = {"leg_quantile": 0.4, "years_back": 5}


@pytest.fixture(scope="module")
def market():
    m = synthetic()
    shape = (len(m.dates), len(m.instruments))
    return replace(m, universe=np.ones(shape, dtype=bool))


def test_synthetic_has_no_universe_and_strategy_fails_loudly():
    m = synthetic()
    assert m.universe is None
    with pytest.raises(ValueError):
        target_weights(m, PARAMS)


def test_primary_quantile_on_small_universe_holds_nothing(market):
    W = target_weights(market, PARAMS)     # 0.2 * ~31 < 10 names per leg
    assert np.all(np.nan_to_num(W) == 0.0)


def test_end_to_end_shape_gross_and_decisions(market):
    W = target_weights(market, P_E2E)
    assert W.shape == (len(market.dates), len(market.instruments))
    rows, _ = decision_rows(market.dates)
    dec = ~np.all(np.isnan(W), axis=1)
    assert np.all(dec[rows])
    gross = np.nansum(np.abs(W), axis=1)
    assert np.all(gross <= 1.0 + 1e-9)
    nonzero = rows[np.abs(W[rows]).sum(axis=1) > 0]
    assert len(nonzero) > 0
    assert np.allclose(np.nansum(W[nonzero], axis=1), 0.0)
    # non-decision rows only ever close positions
    other = np.setdiff1d(np.arange(len(W)), rows)
    vals = W[other][np.isfinite(W[other])]
    assert np.all(vals == 0.0)
    # held names are universe members and tradable at the decision
    u = np.asarray(market.universe, dtype=bool)
    tr = np.asarray(market.tradable, dtype=bool)
    held = np.abs(np.nan_to_num(W[rows])) > 0
    assert np.all((u[rows] & tr[rows])[held])


def test_end_to_end_first_decision_needs_years_back(market):
    rows, _ = decision_rows(market.dates)
    for Y in (3, 5):
        W = target_weights(market, {"leg_quantile": 0.4, "years_back": Y})
        traded = rows[np.abs(np.nan_to_num(W[rows])).sum(axis=1) > 0]
        months_in = month_ids(market.dates[traded[0]]) - month_ids(market.dates[0])
        # target month = decision month + 1 needs month (target - 12Y) inside the data
        assert months_in + 1 - 12 * Y >= 0
        assert months_in + 1 - 12 * Y <= 1   # first full month of data is used as soon as possible


def test_end_to_end_params_change_output(market):
    base = target_weights(market, P_E2E)
    for name, value in (("leg_quantile", 0.45), ("years_back", 3)):
        p = dict(P_E2E)
        p[name] = value
        other = target_weights(market, p)
        assert not np.array_equal(np.nan_to_num(base, nan=7.0), np.nan_to_num(other, nan=7.0)), name


def test_end_to_end_deterministic(market):
    a = target_weights(market, P_E2E)
    b = target_weights(market, P_E2E)
    assert np.array_equal(np.nan_to_num(a, nan=7.0), np.nan_to_num(b, nan=7.0))


def test_exit_when_leaving_universe_between_decisions(market):
    W = target_weights(market, P_E2E)
    rows, _ = decision_rows(market.dates)
    traded = rows[np.abs(np.nan_to_num(W[rows])).sum(axis=1) > 0]
    t = traded[len(traded) // 2]
    nxt = rows[rows > t][0]
    ok = np.asarray(market.tradable, dtype=bool)[t + 1:nxt].all(axis=0)
    j = int(np.flatnonzero((W[t] > 0) & ok)[0])
    k = int(np.flatnonzero((W[t] < 0) & ok)[0])
    out = t + 3
    assert out < nxt
    u = np.ones((len(market.dates), len(market.instruments)), dtype=bool)
    u[out, j] = False                      # long name leaves the universe for one day
    W2 = target_weights(replace(market, universe=u), P_E2E)
    assert np.all(np.isnan(W2[t + 1:out, j]))
    assert np.all(W2[out:nxt, j] == 0.0)   # closed and stays closed until the next decision
    assert np.all(np.isnan(W2[t + 1:nxt, k]))   # other positions keep (NaN = hold)
    assert np.array_equal(W2[t], W[t])
    # not-tradable also closes the position
    tr = np.asarray(market.tradable, dtype=bool).copy()
    tr[out, k] = False
    W3 = target_weights(replace(market, tradable=tr), P_E2E)
    assert np.all(W3[out:nxt, k] == 0.0)
    assert np.all(np.isnan(W3[t + 1:out, k]))
