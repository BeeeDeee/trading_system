import math

import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import PARAMS, target_weights
from strategy import (
    COUNTRY_KEY, DATASET, bond_scores, decision_rows, last_session_of_month, monthly_samples,
    select_legs, _good_friday_day, _month_index,
)

TICKERS = sorted(COUNTRY_KEY)
N = len(TICKERS)


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def _mi(y, m):
    return (y - 1970) * 12 + (m - 1)


def _dn(y, m, d):
    """Day number of y-m-d."""
    return int(np.array([_mi(y, m)]).astype("datetime64[M]").astype("datetime64[D]").astype(np.int64)[0]) + d - 1


def _days(start_year, start_month, n_months):
    """Weekdays (Mon-Fri) from the first of a month for n_months, as datetime64[D]."""
    a = _dn(start_year, start_month, 1)
    m_end = _mi(start_year, start_month) + n_months
    b = int(np.array([m_end]).astype("datetime64[M]").astype("datetime64[D]").astype(np.int64)[0])
    allday = np.arange(a, b).astype("datetime64[D]")
    return allday[np.is_busday(allday)]


def monthly_yields(n_data_months, seed=0):
    """(M, 15) monthly yields: trend + alternating noise, all distinct month to month."""
    rng = np.random.default_rng(seed)
    slopes = rng.normal(0, 0.05, size=N)
    noise = rng.uniform(0.05, 0.3, size=(n_data_months, N)) * np.where(
        np.arange(n_data_months)[:, None] % 2 == 0, 1.0, -1.0)
    return 5.0 + np.cumsum(slopes[None, :] + noise, axis=0)


def step_series(dates, ym, first_data_month):
    """Daily step series: value of data month D usable from the first session of month D+2."""
    mon = _month_index(np.asarray(dates).astype("datetime64[D]"))
    D = mon - 2 - first_data_month
    Y = np.full((len(dates), N), np.nan)
    ok = (D >= 0) & (D < len(ym))
    Y[ok] = ym[D[ok]]
    return Y


def make_view(dates, yields, close=None, ret=None, universe=None, seed=1):
    T = len(dates)
    rng = np.random.default_rng(seed)
    if ret is None:
        ret = rng.normal(0, 0.01, size=(T, N)) * (1.0 + np.arange(N) / 5.0)
    if close is None:
        close = 100.0 * np.cumprod(1 + ret, axis=0)
    listed = np.isfinite(close)
    ret = np.where(listed, ret, 0.0)
    series = {DATASET + "." + COUNTRY_KEY[tk]: yields[:, j].astype(np.float32) for j, tk in enumerate(TICKERS)}
    return DataView(
        dates=np.asarray(dates).astype("datetime64[D]"),
        instruments=tuple(TICKERS),
        asset_class=tuple("us_etf" for _ in TICKERS),
        ret_co=np.zeros((T, N)),
        ret_oc=ret,
        tradable=listed.copy(),
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.where(listed, 1e8, np.nan),
        universe=universe,
        extras={},
        series=series,
        cash_ret=np.zeros(T),
    )


def expected_score(ym_valid, D0, L):
    """Independent score from a monthly array (NaN = missing), D0 index of y(0)."""
    y0 = ym_valid[D0]
    yl = ym_valid[D0 - L] if D0 - L >= 0 else np.nan
    lo = max(0, D0 - 35)
    w = ym_valid[lo:D0 + 1]
    ch = np.diff(w, axis=0)             # NaN where either month is missing
    out = np.full(N, np.nan)
    for j in range(N):
        c = ch[:, j][np.isfinite(ch[:, j])]
        if len(c) < 24 or not (np.isfinite(y0[j]) and np.isfinite(yl[j] if np.ndim(yl) else yl)):
            continue
        out[j] = -(y0[j] - yl[j]) / (np.std(c, ddof=1) * math.sqrt(L))
    return out


# ------------------------------------------------------------------------------------------------
# calendar
# ------------------------------------------------------------------------------------------------
def test_good_friday():
    assert _good_friday_day([2018])[0] == _dn(2018, 3, 30)
    assert _good_friday_day([2024])[0] == _dn(2024, 3, 29)
    assert _good_friday_day([2021])[0] == _dn(2021, 4, 2)
    assert _good_friday_day([2000])[0] == _dn(2000, 4, 21)


def test_last_session_of_month():
    d = np.array([_dn(2018, 3, 5), _dn(2021, 5, 3), _dn(2020, 1, 2), _dn(2015, 2, 2), _dn(2024, 3, 1),
                  _dn(2010, 5, 3), _dn(2019, 6, 3)]).astype("datetime64[D]")
    got = last_session_of_month(d)
    assert list(got) == [_dn(2018, 3, 29),      # Good Friday 3-30
                         _dn(2021, 5, 28),      # Memorial Day Monday 5-31
                         _dn(2020, 1, 31),
                         _dn(2015, 2, 27),      # 2-28 is a Saturday
                         _dn(2024, 3, 28),      # Good Friday 3-29
                         _dn(2010, 5, 28),      # Memorial Day Monday 5-31
                         _dn(2019, 6, 28)]


def test_decision_rows_one_per_month_and_fallback():
    d = _days(2018, 1, 12)
    dec = decision_rows(d)
    mon = _month_index(d)
    assert dec.sum() == 12
    for m in np.unique(mon):
        assert dec[mon == m].sum() == 1
    dec_days = d[dec].astype(np.int64)
    assert _dn(2018, 3, 29) in dec_days and _dn(2018, 3, 30) not in dec_days
    assert _dn(2018, 4, 30) in dec_days
    # unforeseen closure on the last weekday of January 2018: the first February row decides
    d2 = d[d.astype(np.int64) != _dn(2018, 1, 31)]
    dec2 = decision_rows(d2)
    assert _dn(2018, 2, 1) in d2[dec2].astype(np.int64)
    assert dec2.sum() == 12
    # point in time: decisions on a truncated calendar equal the full ones
    for cut in (17, 40, 61, 100):
        assert np.array_equal(decision_rows(d[:cut]), dec[:cut])


# ------------------------------------------------------------------------------------------------
# bond score
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("L", [9, 12, 15])
def test_bond_score_matches_hand_formula(L):
    fdm = _mi(1997, 11)                      # first data month; usable from 1998-01
    d = _days(1998, 1, 50)
    ym = monthly_yields(60)
    Y = step_series(d, ym, fdm)
    mon = _month_index(d)
    m0, samp = monthly_samples(Y, mon)
    dec = np.flatnonzero(decision_rows(d))
    for t in dec[[10, 24, 25, 40, 49]]:
        D0 = int(mon[t]) - 2 - fdm
        got = bond_scores(Y, mon, samp, m0, t, L)
        exp = expected_score(ym[:D0 + 1], D0, L)
        assert np.allclose(got, exp, equal_nan=True, rtol=1e-9)
    # first defined score: 24 changes need 25 months, i.e. D0 = 24
    t24 = dec[24]
    assert np.all(np.isfinite(bond_scores(Y, mon, samp, m0, t24, L)))
    assert np.all(np.isnan(bond_scores(Y, mon, samp, m0, dec[23], L)))


def test_missing_month_no_filling():
    fdm = _mi(1997, 11)
    d = _days(1998, 1, 50)
    ym = monthly_yields(60)
    L = 12
    mon = _month_index(d)
    dec = np.flatnonzero(decision_rows(d))
    t = dec[45]
    D0 = int(mon[t]) - 2 - fdm
    # month D0 not published for country 0: the step series carries the previous value
    ym2 = ym.copy()
    ym2[D0, 0] = ym2[D0 - 1, 0]
    Y = step_series(d, ym2, fdm)
    m0, samp = monthly_samples(Y, mon)
    s = bond_scores(Y, mon, samp, m0, t, L)
    assert np.isnan(s[0]) and np.all(np.isfinite(s[1:]))
    # month D0 - L missing for country 3 -> no score
    ym3 = ym.copy()
    ym3[D0 - L, 3] = ym3[D0 - L - 1, 3]
    Y = step_series(d, ym3, fdm)
    m0, samp = monthly_samples(Y, mon)
    s = bond_scores(Y, mon, samp, m0, t, L)
    assert np.isnan(s[3])
    # a missing month inside the window only removes its two changes
    ym4 = ym.copy()
    ym4[D0 - 5, 2] = ym4[D0 - 6, 2]
    Y = step_series(d, ym4, fdm)
    m0, samp = monthly_samples(Y, mon)
    s = bond_scores(Y, mon, samp, m0, t, L)
    v = ym.copy()
    v[D0 - 5, 2] = np.nan
    exp = expected_score(v[:D0 + 1], D0, L)
    assert np.isclose(s[2], exp[2])
    # NaN yields (series not started) -> no score
    ym5 = ym.copy()
    ym5[: D0 - 10, 4] = np.nan
    Y = step_series(d, ym5, fdm)
    m0, samp = monthly_samples(Y, mon)
    assert np.isnan(bond_scores(Y, mon, samp, m0, t, L)[4])


# ------------------------------------------------------------------------------------------------
# cohort selection
# ------------------------------------------------------------------------------------------------
def test_select_legs():
    names = TICKERS[:10]
    sc = {n: float(i) for i, n in enumerate(names)}
    lo, sh = select_legs(sc, TICKERS)
    assert lo == [names[9], names[8]] and sh == [names[0], names[1]]     # k = 10 // 4 = 2
    sc15 = {n: float(i) for i, n in enumerate(TICKERS)}
    lo, sh = select_legs(sc15, TICKERS)
    assert len(lo) == 3 and len(sh) == 3
    assert select_legs({n: 1.0 for n in TICKERS[:7]}, TICKERS) == ([], [])
    # ties: equal scores -> alphabetical
    tie = {n: 0.0 for n in TICKERS[:8]}
    lo, sh = select_legs(tie, TICKERS)
    assert lo == TICKERS[:2] and sh == TICKERS[:2]
    tie2 = {n: float(i) for i, n in enumerate(TICKERS[:8])}
    tie2[TICKERS[0]] = 7.0
    lo, _ = select_legs(tie2, TICKERS)
    assert lo == [TICKERS[0], TICKERS[7]]


# ------------------------------------------------------------------------------------------------
# full strategy
# ------------------------------------------------------------------------------------------------
def _setup(n_months=60, seed=0):
    fdm = _mi(1997, 11)
    d = _days(1998, 1, n_months)
    ym = monthly_yields(n_months + 5, seed=seed)
    Y = step_series(d, ym, fdm)
    return d, ym, Y, fdm


def _expected_cohort(v, ym_valid, fdm, t, H, L, eligible_mask=None):
    mon = _month_index(np.asarray(v.dates).astype("datetime64[D]"))
    D0 = int(mon[t]) - 2 - fdm
    sc = expected_score(ym_valid[:D0 + 1], D0, L)
    lr = np.log1p(np.asarray(v.ret_co)) + np.log1p(np.asarray(v.ret_oc))
    close = np.asarray(v.close)
    elig = {}
    sig = {}
    for j, tk in enumerate(TICKERS):
        bars = np.flatnonzero(np.isfinite(close[: t + 1, j]))
        if not np.isfinite(close[t, j]) or len(bars) < 63 or not np.isfinite(sc[j]):
            continue
        if eligible_mask is not None and not eligible_mask[j]:
            continue
        elig[tk] = sc[j]
        sig[tk] = np.std(lr[bars[-63:], j], ddof=1)
    c = np.zeros(N)
    if len(elig) < 8:
        return c
    k = len(elig) // 4
    order = sorted(elig, key=lambda n: -elig[n])
    for leg, tot in ((order[:k], 0.5 / H), (order[::-1][:k], -0.5 / H)):
        inv = np.array([1 / sig[n] for n in leg])
        for n, w in zip(leg, tot * inv / inv.sum()):
            c[TICKERS.index(n)] += w
    return c


def test_weights_book_and_schedule():
    d, ym, Y, fdm = _setup()
    v = make_view(d, Y)
    H, L = PARAMS["hold_months"], PARAMS["lookback_months"]
    W = target_weights(v, PARAMS)
    dec = decision_rows(d)
    assert W.shape == (len(d), N)
    assert np.all(np.isnan(W[~dec]))                       # no decision between month-ends
    assert np.all(np.isfinite(W[dec]))
    assert np.all(np.abs(W[dec]).sum(axis=1) <= 1.0 + 1e-12)
    dec_rows = np.flatnonzero(dec)
    # no cohort before 25 monthly values exist (decision index 24)
    assert np.all(W[dec_rows[:24]] == 0.0)
    cohorts = [_expected_cohort(v, ym, fdm, t, H, L) for t in dec_rows]
    for i, t in enumerate(dec_rows):
        exp = np.sum(cohorts[max(0, i - H + 1): i + 1], axis=0)
        assert np.allclose(W[t], exp, atol=1e-12), i
    # first cohort: 15 eligible -> k = 3, legs +0.5/H and -0.5/H, dollar neutral
    w0 = W[dec_rows[24]]
    assert (w0 > 0).sum() == 3 and (w0 < 0).sum() == 3
    assert np.isclose(w0[w0 > 0].sum(), 0.5 / H) and np.isclose(w0[w0 < 0].sum(), -0.5 / H)
    # after H cohorts the book is fully invested and dollar neutral
    wf = W[dec_rows[24 + H + 2]]
    assert np.isclose(np.abs(wf).sum(), 1.0, atol=1e-9) or np.abs(wf).sum() < 1.0
    assert np.isclose(wf.sum(), 0.0, atol=1e-12)


def test_inverse_vol_within_leg():
    d, ym, Y, fdm = _setup()
    v = make_view(d, Y)
    W = target_weights(v, PARAMS)
    t = np.flatnonzero(decision_rows(d))[24]
    w = W[t]
    lr = np.asarray(v.ret_oc)
    sig = np.std(np.log1p(lr[t - 62: t + 1]), axis=0, ddof=1)
    longs = np.flatnonzero(w > 0)
    prod = w[longs] * sig[longs]
    assert np.allclose(prod, prod[0])


def test_params_change_output():
    d, ym, Y, fdm = _setup()
    v = make_view(d, Y)
    base = target_weights(v, PARAMS)
    for key, val in (("hold_months", 3), ("hold_months", 12), ("lookback_months", 9), ("lookback_months", 15)):
        p = dict(PARAMS)
        p[key] = val
        assert not np.allclose(np.nan_to_num(base), np.nan_to_num(target_weights(v, p)))


def test_missing_bar_zeroes_existing_cohorts_and_late_listing():
    d, ym, Y, fdm = _setup()
    rng = np.random.default_rng(1)
    ret = rng.normal(0, 0.01, size=(len(d), N)) * (1.0 + np.arange(N) / 5.0)
    close = 100.0 * np.cumprod(1 + ret, axis=0)
    dec_rows = np.flatnonzero(decision_rows(d))
    v = make_view(d, Y, close=close.copy(), ret=ret)
    W = target_weights(v, PARAMS)
    t1 = dec_rows[26]
    j = int(np.flatnonzero(W[t1] != 0)[0])
    # remove the bar of instrument j on the decision row t2 = next month-end
    t2 = dec_rows[27]
    close2 = close.copy()
    close2[t2, j] = np.nan
    v2 = make_view(d, Y, close=close2, ret=ret)
    W2 = target_weights(v2, PARAMS)
    assert np.array_equal(W2[: t2], W[: t2], equal_nan=True)
    assert W2[t2, j] == 0.0
    # later: cohorts formed up to t2 keep 0 for j; only cohorts formed after t2 can hold it
    H = PARAMS["hold_months"]
    t3 = dec_rows[28]
    c3 = _expected_cohort(v2, ym, fdm, t3, H, PARAMS["lookback_months"])
    assert np.isclose(W2[t3, j], c3[j])
    # late listing: instrument 5 without prices before month 40 is ineligible until 63 bars
    close3 = close.copy()
    start = int(np.flatnonzero(_month_index(d) >= _month_index(d)[0] + 40)[0])
    close3[:start, 5] = np.nan
    v3 = make_view(d, Y, close=close3, ret=ret)
    W3 = target_weights(v3, PARAMS)
    assert np.all(W3[dec_rows[:42], 5] == 0.0)
    for t in dec_rows[40:42]:                # about 22 and 44 bars: not eligible
        assert _expected_cohort(v3, ym, fdm, t, H, PARAMS["lookback_months"])[5] == 0.0
    # from month 42 (>= 63 bars) the strategy follows the eligibility rule
    for i in range(42, 50):
        t = dec_rows[i]
        exp = np.sum([_expected_cohort(v3, ym, fdm, dec_rows[q], H, PARAMS["lookback_months"])
                      for q in range(i - H + 1, i + 1)], axis=0)
        assert np.allclose(W3[t], exp, atol=1e-12)


def test_fewer_than_8_eligible_no_new_cohort():
    d, ym, Y, fdm = _setup()
    dec_rows = np.flatnonzero(decision_rows(d))
    Y2 = Y.copy()
    Y2[:, :8] = np.nan                       # only 7 countries with yields
    v = make_view(d, Y2)
    W = target_weights(v, PARAMS)
    assert np.all(W[dec_rows] == 0.0)


def test_point_in_time_truncation():
    d, ym, Y, fdm = _setup()
    v = make_view(d, Y)
    W = target_weights(v, PARAMS)
    rng = np.random.default_rng(5)
    for cut in rng.integers(300, len(d), size=6):
        cut = int(cut)
        vt = make_view(d[:cut], Y[:cut], close=np.asarray(v.close)[:cut], ret=np.asarray(v.ret_oc)[:cut])
        Wt = target_weights(vt, PARAMS)
        assert np.array_equal(Wt, W[:cut], equal_nan=True)


def test_missing_series_raises():
    d, ym, Y, fdm = _setup(30)
    v = make_view(d, Y)
    del v.series[DATASET + ".US"]
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)
