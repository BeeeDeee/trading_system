import math

import numpy as np
import pytest

from strategy import PARAMS, target_weights
from strategy import (auction_offerings, base_supply, cohort_counts, month_end_rows, supply_state,
                      supply_surge, trailing_supply, nyse_month_end_holidays, surge_signal)

INSTR = ("SPY", "SHY", "IEI", "IEF", "TLH", "TLT")
DS = "treasury_auctions_1d"


def td(k):
    return np.asarray(k, dtype=np.int64).astype("timedelta64[D]")


def day(y, m, d):
    return (np.array([(y - 1970) * 12 + m - 1]).astype("datetime64[M]").astype("datetime64[D]") + td(d - 1))[0]


def sessions(start, end, closed=()):
    x = np.arange(start, end + td(1), td(1))
    x = x[np.is_busday(x)]
    if closed:
        x = x[~np.isin(x, np.array(closed, dtype="datetime64[D]"))]
    return x


class FakeData:
    def __init__(self, dates, close, series, tradable=None):
        self.dates = dates
        self.instruments = INSTR
        self.close = close
        self.series = series
        T, N = close.shape
        self.tradable = np.isfinite(close) if tradable is None else tradable
        self.listed = np.isfinite(close)
        self.universe = None

    def col(self, name):
        return self.instruments.index(name)


def auction_series(T, every=4, first=2, rate=1.8e-4, jump_at=None, jump=1.0):
    """Auction on rows first, first+every, ...; every auction announced well ahead."""
    rows = np.arange(T)
    auc = np.arange(first, T + every * 2, every)
    offer = np.exp(rate * auc) * 1e10
    if jump_at is not None:
        offer = np.where(auc >= jump_at, offer * jump, offer)
    last = np.full(T, np.nan)
    nxt = np.full(T, np.nan)
    noff = np.full(T, np.nan)
    for r in rows:
        past = auc[auc <= r]
        if past.size:
            last[r] = r - past[-1]
        k = np.searchsorted(auc, r, side="right")
        nxt[r] = auc[k] - r
        noff[r] = offer[k]
    return {DS + ".last_coupon5_sessions": last, DS + ".next_coupon5_sessions": nxt,
            DS + ".next_coupon5_offering_usd": noff}, auc, offer


def make_data(T=1800, **kw):
    hol = nyse_month_end_holidays(np.array([day(2003, 1, 2), day(2012, 12, 31)]))
    dates = sessions(day(2003, 1, 2), day(2012, 12, 31), closed=list(hol))[:T]
    T = dates.shape[0]
    close = np.full((T, len(INSTR)), 100.0)
    series, auc, offer = auction_series(T, **kw)
    return FakeData(dates, close, series), auc, offer


# ----------------------------------------------------------------------------- calendar
def test_good_friday_and_memorial_day():
    hol = set(nyse_month_end_holidays(np.array([day(2010, 1, 4), day(2019, 1, 2)])).tolist())
    for d in (day(2013, 3, 29), day(2018, 3, 30), day(2019, 4, 19), day(2010, 4, 2),
              day(2010, 5, 31), day(2013, 5, 27), day(2018, 5, 28)):
        assert d.tolist() in hol


def test_month_end_memorial_day_on_31st():
    d = sessions(day(2010, 5, 3), day(2010, 6, 30), closed=[day(2010, 5, 31)])
    me = month_end_rows(d)
    assert set(d[me].tolist()) == {day(2010, 5, 28).tolist(), day(2010, 6, 30).tolist()}


def test_month_end_good_friday_last_weekday():
    d = sessions(day(2013, 3, 1), day(2013, 4, 30), closed=[day(2013, 3, 29)])
    me = month_end_rows(d)
    assert set(d[me].tolist()) == {day(2013, 3, 28).tolist(), day(2013, 4, 30).tolist()}


def test_month_end_fallback_unscheduled_closure():
    # 2014-01-31 (Friday) closed without notice: the first session of February decides for January
    d = sessions(day(2014, 1, 2), day(2014, 2, 28), closed=[day(2014, 1, 31)])
    me = month_end_rows(d)
    assert set(d[me].tolist()) == {day(2014, 2, 3).tolist(), day(2014, 2, 28).tolist()}


def test_month_end_point_in_time():
    hol = nyse_month_end_holidays(np.array([day(2004, 1, 2), day(2006, 12, 29)]))
    d = sessions(day(2004, 1, 2), day(2006, 12, 29), closed=list(hol))
    full = month_end_rows(d)
    assert full.sum() == 36
    for k in (1, 2, 17, 21, 22, 300, 500):
        assert np.array_equal(month_end_rows(d[:k]), full[:k])


# ----------------------------------------------------------------------------- supply signal
def test_auction_offerings_and_missing():
    last = np.array([5, 0, 1, 0, 0, 2, 0], float)
    nxt = np.array([1, 3, 2, 1, 1, np.nan, np.nan])
    noff = np.array([10, 99, 99, 20, 30, np.nan, np.nan])
    off, miss = auction_offerings(last, nxt, noff)
    assert np.array_equal(off, [0, 10, 0, 0, 20, 0, 0])
    assert miss == 2
    # auction on the first row has no previous session: missing
    off, miss = auction_offerings(np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([5.0, 5.0]))
    assert np.array_equal(off, [0, 0]) and miss == 1


def test_trailing_supply_window():
    S = trailing_supply(np.ones(300))
    assert np.all(np.isnan(S[:251])) and np.all(S[251:] == 252)


def test_base_and_surge_hand_case():
    T = 1300
    off = np.ones(T)
    off[T - 252:] = 2.0
    S = trailing_supply(off)
    B = base_supply(S, 2)
    g = supply_surge(S, B)
    assert np.isnan(g[1006]) and np.isfinite(g[1007])          # by=2: needs a full S window at t-756
    assert g[1007] == pytest.approx(0.0)
    assert g[T - 1] == pytest.approx(math.log(2.0))
    B3 = base_supply(S, 3)
    assert np.isnan(B3[1258]) and np.isfinite(B3[1259])
    st = supply_state(g, 0.1)
    assert st[T - 1] == 1 and st[1007] == 0 and st[1000] == 0
    off2 = np.ones(T)
    off2[T - 252:] = 0.5
    g2 = supply_surge(trailing_supply(off2), base_supply(trailing_supply(off2), 2))
    assert supply_state(g2, 0.1)[T - 1] == -1


def test_base_points():
    # B is the mean of S at t-252, t-273, ..., t-252*(1+by): check with an S that is the row index
    T = 2000
    S = np.arange(T, dtype=float)
    B = base_supply(S, 3)
    t = 1500
    pts = [t - 252 - 21 * k for k in range(37)]
    assert pts[-1] == t - 1008
    assert B[t] == pytest.approx(np.mean(pts))


def test_cohort_counts():
    st = np.array([1, 1, 0, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1])
    h, l, n = cohort_counts(st)
    assert (h[0], l[0], n[0]) == (1, 0, 11)                       # 11 pre-start cohorts count as IEF
    assert (h[3], l[3], n[3]) == (2, 1, 9)
    assert (h[11], l[11], n[11]) == (10, 1, 1)
    assert (h[12], l[12], n[12]) == (10, 1, 1)                    # first cohort (HIGH) expired, new HIGH
    assert (h[14], l[14], n[14]) == (10, 2, 0)                    # live cohorts 3..14: two LOW, ten HIGH
    assert np.all(h + l + n == 12)


# ----------------------------------------------------------------------------- book
def test_weights_book():
    data, _, _ = make_data()
    W = target_weights(data, PARAMS)
    me = month_end_rows(data.dates)
    dec = np.all(np.isfinite(W), axis=1)
    assert np.array_equal(dec, me)
    assert np.all(np.isnan(W[~dec]))
    Wd = W[dec]
    assert np.all(Wd >= 0) and np.allclose(Wd.sum(axis=1), 1.0)
    assert np.allclose(Wd[:, data.col("SPY")], 0.6)
    assert np.all(Wd[:, data.col("IEI")] == 0) and np.all(Wd[:, data.col("TLH")] == 0)
    # g undefined before 252*(2+3)-1 rows: pure 60/40 SPY/IEF
    rows = np.flatnonzero(dec)
    early = rows < 1259
    assert np.allclose(W[rows[early], data.col("IEF")], 0.4)
    # constant growth -> g about 0.11 > 0.1: HIGH from the first defined month on, cohorts fill up one per month
    late = rows[~early]
    tlt = W[late, data.col("TLT")]
    k = np.arange(1, late.size + 1)
    assert np.allclose(tlt, 0.4 * np.minimum(k, 12) / 12)
    assert np.allclose(W[late, data.col("SHY")], 0.0)


def test_surge_value_direct():
    data, auc, offer = make_data()
    g = surge_signal(data, PARAMS)
    t = len(data.dates) - 1

    def S_at(s):
        sel = (auc <= s) & (auc > s - 252)
        return offer[sel].sum()

    B = np.mean([S_at(t - 252 - 21 * k) for k in range(37)])
    assert g[t] == pytest.approx(math.log(S_at(t) / B))
    assert 0.1 < g[t] < 0.15


def test_params_change_output():
    data, _, _ = make_data()
    W0 = target_weights(data, PARAMS)
    for p in ({"base_years": 2, "supply_cut": 0.1}, {"base_years": 4, "supply_cut": 0.1},
              {"base_years": 3, "supply_cut": 0.15}):
        W1 = target_weights(data, p)
        assert not np.array_equal(np.nan_to_num(W0, nan=-1), np.nan_to_num(W1, nan=-1)), p
        d = np.all(np.isfinite(W1), axis=1)
        assert np.all(W1[d] >= 0) and np.all(W1[d].sum(axis=1) <= 1 + 1e-12)
    # base_years 2: g about 0.09, NEUTRAL at cut 0.10 but HIGH at cut 0.05
    Wa = target_weights(data, {"base_years": 2, "supply_cut": 0.1})
    Wb = target_weights(data, {"base_years": 2, "supply_cut": 0.05})
    d = np.all(np.isfinite(Wa), axis=1)
    assert np.allclose(Wa[d, data.col("TLT")], 0.0) and Wb[d, data.col("TLT")].max() == pytest.approx(0.4)


def test_late_listing_and_truncation():
    data, _, _ = make_data()
    data.close[:300, data.col("TLT")] = np.nan
    W = target_weights(data, PARAMS)
    rows = np.flatnonzero(np.all(np.isfinite(W), axis=1))
    assert rows[0] >= 300
    assert np.all(np.isnan(W[:300]))
    for k in (400, 1260, 1500, 1700):
        sub = FakeData(data.dates[:k], data.close[:k], {n: v[:k] for n, v in data.series.items()})
        Wk = target_weights(sub, PARAMS)
        assert np.array_equal(np.nan_to_num(Wk, nan=-9), np.nan_to_num(W[:k], nan=-9))


def test_missing_series_raises():
    data, _, _ = make_data(T=300)
    del data.series[DS + ".next_coupon5_offering_usd"]
    with pytest.raises(KeyError):
        target_weights(data, PARAMS)


# ----------------------------------------------------------------------------- diagnostic
def test_low_state_moves_sleeve_to_shy():
    # offering halved for the last 300 rows -> LOW at the last month-ends: new cohorts go to SHY
    data, _, _ = make_data(jump_at=1800 - 300, jump=0.5)
    g = surge_signal(data, PARAMS)
    assert g[-1] < -0.1
    W = target_weights(data, PARAMS)
    rows = np.flatnonzero(np.all(np.isfinite(W), axis=1))
    assert W[rows[-1], data.col("SHY")] > 0
    assert W[rows[-1], data.col("TLT")] + W[rows[-1], data.col("SHY")] + W[rows[-1], data.col("IEF")] == pytest.approx(0.4)
    # the cohorts opened in LOW months add 0.4/12 SHY each month
    st = supply_state(g, 0.1)[rows]
    n_low = np.convolve(st == -1, np.ones(12), mode="full")[:rows.size]
    assert np.allclose(W[rows, data.col("SHY")], 0.4 * n_low / 12)
