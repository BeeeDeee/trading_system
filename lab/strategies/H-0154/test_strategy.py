import numpy as np
import pytest

from strategy import PARAMS, target_weights  # noqa: F401
from strategy import (cum_return, decision_rows, fill_dividend_yield, level_change, member_mean,
                      rate_beta, sleeve_book, top_names)

# day numbers since 1970-01-01 (no date strings): 18628 = 2021-01-01 (Friday)
APR26_2021 = 18743   # Monday
APR30_2021 = 18747   # Friday, last weekday of April
MAY28_2021 = 18775   # Friday, last trading day of May (May 31 is Memorial Day)
MAY31_2021 = 18778   # Monday, holiday
JUN01_2021 = 18779   # Tuesday
JUN30_2021 = 18808   # Wednesday
JUL02_2021 = 18810


def _trading_days(lo, hi, holidays=()):
    d = np.arange(lo, hi + 1).astype("datetime64[D]")
    d = d[np.is_busday(d)]
    hol = np.array(holidays, dtype=np.int64)
    keep = ~np.isin(d.astype(np.int64), hol)
    return d[keep]


# ------------------------------------------------------------------ calendar
def test_decision_rows_month_ends_and_holiday_fallback():
    d = _trading_days(APR26_2021, JUL02_2021, holidays=(MAY31_2021,))
    dec = decision_rows(d)
    got = d[dec].astype(np.int64).tolist()
    # April 30 (month end), May 28 is missed by weekday arithmetic (May 31 is a weekday) -> June 1 late,
    # June 30 month end
    assert got == [APR30_2021, JUN01_2021, JUN30_2021]
    assert not dec[d.astype(np.int64) == MAY28_2021].any()


def test_decision_rows_point_in_time_under_truncation():
    d = _trading_days(APR26_2021, JUL02_2021, holidays=(MAY31_2021,))
    full = decision_rows(d)
    for cut in range(1, d.shape[0]):
        assert np.array_equal(decision_rows(d[:cut]), full[:cut])


def test_decision_rows_empty():
    assert decision_rows(np.zeros(0, dtype="datetime64[D]")).shape == (0,)


# ------------------------------------------------------------------ dividend yield
def test_fill_dividend_yield_rules():
    dates = (np.arange(6) * 100 + 18000).astype("datetime64[D]")
    nan = np.nan
    dy = np.array([
        [0.03, nan, 0.02, nan],
        [0.03, nan, nan, nan],
        [0.03, nan, nan, 0.01],
        [0.03, nan, nan, nan],
        [0.03, nan, nan, 0.01],
        [0.03, nan, nan, nan],
    ])
    out = fill_dividend_yield(dy, dates, recent_days=400)
    # present values are kept
    assert np.allclose(out[:, 0], 0.03)
    # never filed -> not scored
    assert np.isnan(out[:, 1]).all()
    # filing visible on row 0: 0 while it is less than 400 days old, NaN from 400 days on
    assert out[0, 2] == 0.02
    assert out[1, 2] == 0.0 and out[2, 2] == 0.0 and out[3, 2] == 0.0
    assert np.isnan(out[4, 2]) and np.isnan(out[5, 2])
    # late first filing on row 2; NaN gaps after it are 0
    assert np.isnan(out[0, 3]) and np.isnan(out[1, 3])
    assert out[2, 3] == 0.01 and out[3, 3] == 0.0 and out[4, 3] == 0.01 and out[5, 3] == 0.0


def test_fill_dividend_yield_repeat_value_does_not_refresh():
    # an unchanged value on a later row is not a new filing (its age counts from the first appearance)
    dates = (np.arange(6) * 100 + 18000).astype("datetime64[D]")
    nan = np.nan
    dy = np.array([[0.02], [0.02], [0.02], [nan], [nan], [nan]])
    out = fill_dividend_yield(dy, dates, recent_days=400)
    assert out[3, 0] == 0.0          # 300 days after the visible filing
    assert np.isnan(out[4, 0])       # 400 days after


# ------------------------------------------------------------------ helpers
def test_member_mean_and_cum_return():
    r = np.array([[0.01, 0.03, 0.5], [np.nan, 0.02, 0.5], [0.0, 0.0, 0.0]])
    uni = np.array([[True, True, False], [True, True, False], [False, False, False]])
    m = member_mean(r, uni)
    assert np.allclose(m[:2], [0.02, 0.02]) and np.isnan(m[2])

    s = np.array([0.01, 0.01, 0.01, 0.01, np.nan, 0.01, 0.01, 0.01])
    c = cum_return(s, 3)
    assert np.isnan(c[:2]).all()
    assert np.allclose(c[2:4], 1.01 ** 3 - 1)
    assert np.isnan(c[4:7]).all()            # windows containing the missing day
    assert np.isclose(c[7], 1.01 ** 3 - 1)


def test_level_change():
    lv = np.array([1.0, 2.0, 4.0, 7.0])
    out = level_change(lv, 2)
    assert np.isnan(out[:2]).all() and np.allclose(out[2:], [3.0, 5.0])
    assert np.isnan(level_change(lv, 10)).all()


# ------------------------------------------------------------------ rate beta
def _beta_inputs(T=60, seed=1):
    rng = np.random.default_rng(seed)
    level = 4.0 + np.cumsum(rng.normal(0, 0.05, T))
    # aligned FRED: row t holds the value dated t-1 -> the change of day s is level[s+1] - level[s]
    chg_day = np.full(T, np.nan)
    chg_day[:-1] = level[1:] - level[:-1]
    mkt = rng.normal(0, 0.01, T)
    r = np.zeros((T, 3))
    r[:, 0] = mkt + 2.0 * np.nan_to_num(chg_day)
    r[:, 1] = mkt - 0.5 * np.nan_to_num(chg_day) + 0.001
    r[:, 2] = mkt + 1.0 * np.nan_to_num(chg_day)
    r[::2, 2] = np.nan                      # half of the pairs missing
    return r, mkt, level


def test_rate_beta_slope_and_minimum_pairs():
    r, mkt, level = _beta_inputs()
    b = rate_beta(r, mkt, level, window=20, min_pairs=15)
    # row t uses the pairs of days max(0, t-20)..t-1: t pairs while t <= 20
    assert np.isnan(b[:15, :2]).all()
    assert np.allclose(b[15:, 0], 2.0, atol=1e-6)
    assert np.allclose(b[15:, 1], -0.5, atol=1e-6)
    # column 2 has only about 10 valid pairs in any 20-day window -> never scored
    assert np.isnan(b[:, 2]).all()
    b2 = rate_beta(r, mkt, level, window=20, min_pairs=8)
    assert np.allclose(b2[np.isfinite(b2[:, 2]), 2], 1.0, atol=1e-6)
    assert np.isfinite(b2[30:, 2]).all()


def test_rate_beta_uses_only_days_before_t():
    r, mkt, level = _beta_inputs()
    base = rate_beta(r, mkt, level, window=20, min_pairs=15)
    for t in (25, 40, 59):
        r2 = r.copy()
        r2[t:] = 0.3                         # change the stock return of day t and later
        m2 = mkt.copy()
        m2[t:] = -0.2
        b = rate_beta(r2, m2, level, window=20, min_pairs=15)
        assert np.allclose(b[:t + 1], base[:t + 1], equal_nan=True)


def test_rate_beta_short_view():
    r, mkt, level = _beta_inputs(T=5)
    assert np.isnan(rate_beta(r, mkt, level)).all()


# ------------------------------------------------------------------ book
def test_top_names_rounding_and_ties():
    s = np.array([0.5, 0.9, np.nan, 0.9, 0.1, 0.3])
    mem = np.array([True, True, True, True, True, False])
    m, n = top_names(s, mem, 0.5)
    assert n == 4
    assert m.tolist() == [False, True, False, True, False, False]
    m, n = top_names(s, mem, 0.2)          # 0.8 -> 1 name, tie at 0.9 broken by column order
    assert m.tolist() == [False, True, False, False, False, False]
    m, n = top_names(np.full(6, np.nan), mem, 0.2)
    assert n == 0 and not m.any()


def _book_inputs():
    T, N = 10, 6
    uni = np.ones((T, N), dtype=bool)
    score = np.full((T, N), np.nan)
    score[3:] = np.arange(N)[None, :]      # model exists from row 3; column 5 best, then 4, 3 ...
    decision = np.zeros(T, dtype=bool)
    decision[[0, 3, 6, 9]] = True
    return score, uni, decision


def test_sleeve_book_warmup_rotation_and_exit():
    score, uni, decision = _book_inputs()
    uni[5, 4] = False                       # name 4 leaves the universe on row 5
    uni[6:, 4] = False
    w = sleeve_book(score, uni, decision, top=0.34, min_scored=3)
    # warm-up: all three sleeves hold the equal-weight universe
    assert np.allclose(w[0], 1 / 6)
    assert np.isnan(w[1:3]).all()
    # row 3: sleeve 1 rebuilt with the top 2 (names 4, 5), each 1/6; the other sleeves keep 1/18 each
    assert np.allclose(w[3, [4, 5]], 2 / 18 + 1 / 6)
    assert np.allclose(w[3, :4], 2 / 18)
    assert np.isclose(np.nansum(np.abs(w[3])), 1.0)
    # row 5: name 4 leaves -> sold; every other target is unchanged (full row, same values)
    assert w[5, 4] == 0.0
    assert np.allclose(np.delete(w[5], 4), np.delete(w[3], 4))
    assert np.isnan(w[4]).all()
    # row 6: sleeve 2 rebuilt (top 2 among 5 members = names 5, 3); sleeve 0 still EW minus name 4 (cash)
    # sleeve 1 holds name 5 only (name 4's share is cash)
    exp = np.zeros(6)
    exp[:] += 1 / 18                        # sleeve 0
    exp[4] = 0.0
    exp[5] += 1 / 6                         # sleeve 1
    exp[[5, 3]] += 1 / 6                    # sleeve 2
    assert np.allclose(w[6], exp)
    assert np.isclose(w[6].sum(), 14 / 18)  # name 4's share (1/18 + 1/6) stays in cash
    assert np.isnan(w[7:9]).all()
    # row 9: sleeve 0 rebuilt (names 5, 3 at 1/6) -> the cash of name 4 in sleeve 0 is reinvested
    exp9 = np.zeros(6)
    exp9[5] = 3 / 6
    exp9[3] = 2 / 6
    assert np.allclose(w[9], exp9)
    assert np.isclose(w[9].sum(), 5 / 6)    # sleeve 1 still carries name 4's cash


def test_sleeve_book_gross_and_long_only():
    rng = np.random.default_rng(3)
    T, N = 300, 40
    uni = rng.random((T, N)) > 0.2
    score = rng.normal(size=(T, N))
    score[:50] = np.nan
    decision = np.zeros(T, dtype=bool)
    decision[::21] = True
    w = sleeve_book(score, uni, decision, top=0.2, min_scored=10)
    # reconstruct the held targets (NaN = keep) and check gross <= 1, long only, members only
    pos = np.zeros(N)
    for t in range(T):
        fin = np.isfinite(w[t])
        pos[fin] = w[t, fin]
        assert (pos >= 0).all()
        assert pos.sum() <= 1.0 + 1e-9
        assert not (pos[~uni[t]] > 0).any()


def test_sleeve_book_fewer_than_min_scored_is_equal_weight():
    score, uni, decision = _book_inputs()
    score[3:, :3] = np.nan                   # only 3 scored members
    w = sleeve_book(score, uni, decision, top=0.34, min_scored=4)
    # row 3: sleeve 1 rebuilt as the equal-weight universe -> same targets as the warm-up book
    assert np.allclose(w[3], 1 / 6)
    w2 = sleeve_book(score, uni, decision, top=0.34, min_scored=3)
    # with 3 scored allowed: top 0.34 of 3 -> 1 name (5) gets the whole sleeve
    assert np.isclose(w2[3, 5], 2 / 18 + 1 / 3)
    assert np.allclose(w2[3, :5], 2 / 18)


def test_sleeve_book_top_param_changes_output():
    score, uni, decision = _book_inputs()
    a = sleeve_book(score, uni, decision, top=0.2, min_scored=3)
    b = sleeve_book(score, uni, decision, top=0.5, min_scored=3)
    assert not np.array_equal(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1))


def test_sleeve_book_point_in_time():
    rng = np.random.default_rng(5)
    T, N = 200, 30
    uni = rng.random((T, N)) > 0.1
    score = rng.normal(size=(T, N))
    score[:40] = np.nan
    decision = np.zeros(T, dtype=bool)
    decision[::21] = True
    full = sleeve_book(score, uni, decision, top=0.2, min_scored=10)
    for cut in (30, 63, 105, 150):
        part = sleeve_book(score[:cut], uni[:cut], decision[:cut], top=0.2, min_scored=10)
        assert np.array_equal(np.nan_to_num(part, nan=-1), np.nan_to_num(full[:cut], nan=-1))


def test_params_are_the_card_primary_values():
    assert PARAMS == {"n_estimators": 100, "top": 0.2}
