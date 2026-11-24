import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import PARAMS, target_weights, last_session_flag, decisions, monthly_states

INSTR = ("SPY", "IEF", "TLT", "SHY", "LQD")


def ymd(y, m, d):
    base = np.array([(y - 1970) * 12 + (m - 1)]).astype("datetime64[M]").astype("datetime64[D]")
    return (base.astype(np.int64) + (d - 1)).astype("datetime64[D]")[0]


def weekdays(y0, m0, d0, n):
    start = ymd(y0, m0, d0)
    return np.busday_offset(start, np.arange(n), roll="forward").astype("datetime64[D]")


def make_view(dates, series, close=None):
    T, N = len(dates), len(INSTR)
    if close is None:
        close = np.full((T, N), 100.0)
    listed = np.isfinite(close)
    z = np.zeros((T, N))
    return DataView(
        dates=dates,
        instruments=INSTR,
        asset_class=tuple("us_etf" for _ in INSTR),
        ret_co=z.copy(),
        ret_oc=z.copy(),
        tradable=listed.copy(),
        listed=listed.copy(),
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.where(listed, 1e9, np.nan),
        universe=None,
        extras={},
        series={"fred_macro.T10Y3M": np.asarray(series, dtype=float)},
        cash_ret=np.zeros(T),
    )


def month_ids(dates):
    return dates.astype("datetime64[M]").astype(np.int64)


def const_by_month(dates, values):
    mid = month_ids(dates)
    mid0 = mid[0]
    return np.array([values[int(m - mid0)] for m in mid], dtype=float)


# ------------------------------------------------------------------ calendar
def test_last_session_flag_holidays_and_plain_months():
    d = np.array([ymd(2018, 3, 28), ymd(2018, 3, 29), ymd(2010, 5, 27), ymd(2010, 5, 28),
                  ymd(2019, 1, 30), ymd(2019, 1, 31), ymd(2013, 3, 28), ymd(2002, 3, 28),
                  ymd(2005, 5, 27), ymd(2005, 5, 31), ymd(2020, 2, 28)])
    f = last_session_flag(d)
    # Good Friday 2018-03-30 -> 03-29 is the last session; Memorial Day 2010-05-31 -> 05-28
    # 2013 and 2002: Good Friday on 03-29 -> 03-28 is the last session
    # 2005: Memorial Day 05-30, 05-31 is a session; 2020-02-28 is a Friday, Feb 29 a Saturday
    assert f.tolist() == [False, True, False, True, False, True, True, True, False, True, True]


def test_one_decision_per_month_at_last_session():
    d = weekdays(2005, 1, 3, 130)
    dec = decisions(d)
    mid = month_ids(d)
    months = [m for _, m in dec]
    assert months == sorted(set(months))
    for t, m in dec:
        assert mid[t] == m
        if t + 1 < len(d):
            assert mid[t + 1] != m


def test_fallback_when_month_end_row_missing():
    d = weekdays(2005, 1, 3, 80)
    mid = month_ids(d)
    # drop the last session of January -> January is decided at the first row of February
    last_jan = int(np.nonzero(mid == mid[0])[0][-1])
    d2 = np.delete(d, last_jan)
    dec = decisions(d2)
    assert dec[0] == (last_jan, int(mid[0]))  # first February row now sits at index last_jan
    assert month_ids(d2)[last_jan] == mid[0] + 1


# ------------------------------------------------------------------ signal and book
def _view6():
    d = weekdays(2005, 1, 3, 150)   # Jan..Jul 2005
    s = const_by_month(d, [1.0, 2.0, 3.0, 0.0, 5.0, 4.0, 9.0, 9.0])
    return d, s


def test_monthly_means_and_states_by_hand():
    d, s = _view6()
    v = make_view(d, s)
    dec, xs, st = monthly_states(v, 3)
    m0 = int(month_ids(d)[0])
    assert [xs[m0 + k] for k in range(6)] == [1.0, 2.0, 3.0, 0.0, 5.0, 4.0]
    assert [st[m0 + k] for k in range(6)] == [0, 0, 1, -1, 1, -1]   # tie (4 vs median 4) is flat


def test_weights_by_hand_small_params():
    d, s = _view6()
    v = make_view(d, s)
    w = target_weights(v, {"cohort_months": 2, "median_months": 3})
    dec = decisions(d)
    exp = [(0.0, 0.4, 0.0), (0.0, 0.4, 0.0), (0.2, 0.2, 0.0), (0.2, 0.0, 0.2), (0.2, 0.0, 0.2), (0.2, 0.0, 0.2)]
    for (t, _), (tlt, ief, shy) in zip(dec[:6], exp):
        assert w[t, 0] == pytest.approx(0.6)
        assert w[t, 2] == pytest.approx(tlt)
        assert w[t, 1] == pytest.approx(ief)
        assert w[t, 3] == pytest.approx(shy)
        assert w[t, 4] == 0.0
    dec_rows = {t for t, _ in dec}
    other = [r for r in range(len(d)) if r not in dec_rows]
    assert np.all(np.isnan(w[other]))


def test_daily_mean_excludes_first_row_and_skips_nan():
    d = weekdays(2005, 1, 3, 45)
    mid = month_ids(d)
    s = np.full(len(d), 7.0)
    jan = np.nonzero(mid == mid[0])[0]
    s[jan[0]] = 1000.0        # value dated in December: excluded
    s[jan[3]] = np.nan        # missing day: skipped
    s[jan[5]] = 1.0
    v = make_view(d, s)
    _, xs, _ = monthly_states(v, 1)
    vals = [7.0] * (len(jan) - 1)
    vals[4] = 1.0            # jan[5] is the 5th contributing row
    vals.pop(2)              # jan[3] is NaN
    assert xs[int(mid[0])] == pytest.approx(np.mean(vals))


def test_missing_month_carries_state_and_blocks_window():
    d = weekdays(2005, 1, 3, 150)
    s = const_by_month(d, [1.0, 2.0, 3.0, np.nan, 5.0, 4.0, 9.0, 9.0])
    v = make_view(d, s)
    _, _, st = monthly_states(v, 2)
    m0 = int(month_ids(d)[0])
    # m1: 2 > median(1,2)=1.5 -> +1; m2: 3 > 2.5 -> +1; m3 missing -> carry +1;
    # m4: window (nan, 5) incomplete -> carry +1; m5: 4 > median(5,4)=4.5? no -> -1
    assert [st[m0 + k] for k in range(6)] == [0, 1, 1, 1, 1, -1]


def test_default_params_warmup_is_benchmark():
    d = weekdays(2005, 1, 3, 300)
    rng = np.random.default_rng(0)
    v = make_view(d, rng.normal(1.5, 1.0, len(d)))
    w = target_weights(v, PARAMS)
    for t, _ in decisions(d):
        assert np.allclose(w[t], [0.6, 0.4, 0.0, 0.0, 0.0], rtol=0, atol=1e-12)


def test_late_listing_no_position_before_start():
    d, s = _view6()
    close = np.full((len(d), 5), 100.0)
    close[:30, 2] = np.nan     # TLT lists on row 30
    v = make_view(d, s, close)
    w = target_weights(v, {"cohort_months": 2, "median_months": 3})
    assert np.all(w[:30] == 0.0)
    dec = decisions(d)
    t1 = [t for t, _ in dec if t >= 30][0]
    assert w[t1, 0] == pytest.approx(0.6)


def test_gross_and_long_only_and_params_matter():
    d = weekdays(2003, 1, 1, 2600)
    rng = np.random.default_rng(1)
    s = np.cumsum(rng.normal(0, 0.05, len(d))) + 1.5
    v = make_view(d, s)
    outs = {}
    for H in (6, 12, 24):
        for L in (36, 60, 84):
            w = target_weights(v, {"cohort_months": H, "median_months": L})
            rows = ~np.all(np.isnan(w), axis=1)
            assert np.all(w[rows] >= 0)
            assert np.all(np.abs(w[rows]).sum(axis=1) <= 1.0 + 1e-12)
            assert np.allclose(np.abs(w[rows]).sum(axis=1)[np.abs(w[rows]).sum(axis=1) > 0], 1.0)
            outs[(H, L)] = w
    base = outs[(12, 60)]
    for k, w in outs.items():
        if k != (12, 60):
            assert not np.array_equal(np.nan_to_num(w, nan=-1), np.nan_to_num(base, nan=-1))


def test_point_in_time_truncation():
    d = weekdays(2003, 1, 1, 1400)
    rng = np.random.default_rng(2)
    s = np.cumsum(rng.normal(0, 0.05, len(d))) + 1.5
    full = target_weights(make_view(d, s), {"cohort_months": 6, "median_months": 36})
    for cut in (900, 1000, 1100, 1250, 1399):
        part = target_weights(make_view(d[:cut], s[:cut]), {"cohort_months": 6, "median_months": 36})
        assert np.array_equal(np.nan_to_num(part, nan=-9), np.nan_to_num(full[:cut], nan=-9))


def test_missing_series_raises():
    d, s = _view6()
    v = make_view(d, s)
    v.series.pop("fred_macro.T10Y3M")
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)
