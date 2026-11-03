import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import PARAMS, cohorts, decision_rows, target_weights

# ETF order inside the test market deliberately differs from the strategy's PAIRS order
TICKERS = ("USO", "SPY", "FXE", "QQQ", "GLD", "IWM", "SLV", "EEM", "GDX", "EWZ", "XLE", "FXI")
KEYS = {
    "SPY": "fred_macro.VIXCLS", "QQQ": "fred_macro.VXNCLS", "IWM": "cboe_etf_vol_1d.RVXCLS",
    "EEM": "cboe_etf_vol_1d.VXEEMCLS", "EWZ": "cboe_etf_vol_1d.VXEWZCLS", "FXI": "cboe_etf_vol_1d.VXFXICLS",
    "XLE": "cboe_etf_vol_1d.VXXLECLS", "GDX": "cboe_etf_vol_1d.VXGDXCLS", "GLD": "fred_macro.GVZCLS",
    "SLV": "cboe_etf_vol_1d.VXSLVCLS", "USO": "fred_macro.OVXCLS", "FXE": "cboe_etf_vol_1d.EVZCLS",
}


def weekdays(n, start=14613):            # 14613 = Monday 2010-01-04 as days since the epoch
    d = np.arange(start, start + 2 * n + 10).astype("datetime64[D]")
    return d[np.is_busday(d)][:n]


def make_view(T=330, seed=0, per_instrument_noise=False, dates=None):
    dates = weekdays(T) if dates is None else dates
    T = len(dates)
    N = len(TICKERS)
    sign = np.where(np.arange(T) % 2 == 0, 1.0, -1.0)
    ret_oc = np.repeat((0.01 * sign)[:, None], N, axis=1)
    rng = np.random.default_rng(seed)
    if per_instrument_noise:
        ret_oc = ret_oc + rng.normal(0, 0.01, (T, N))
    ret_co = np.zeros((T, N))
    close = 100.0 * np.cumprod(1.0 + ret_oc, axis=0)
    noise = rng.normal(0, 1, T)
    series = {}
    for j, tk in enumerate(TICKERS):
        nz = rng.normal(0, 1, T) if per_instrument_noise else noise
        series[KEYS[tk]] = 20.0 * np.exp(0.01 * nz)
    return DataView(
        dates=dates, instruments=TICKERS, asset_class=tuple("us_etf" for _ in TICKERS),
        ret_co=ret_co, ret_oc=ret_oc, tradable=np.ones((T, N), bool), listed=np.ones((T, N), bool),
        delisting=np.zeros((T, N), bool), close=close, dollar_volume=np.full((T, N), 1e9),
        universe=None, extras={}, series=series, cash_ret=np.zeros(T),
    )


def last_decision(data):
    rows = np.nonzero(decision_rows(data.dates))[0]
    return rows[-1]


def day_of_month(d):
    return (d - d.astype("datetime64[M]")).astype(int) + 1


# ---------------------------------------------------------------- calendar
def test_month_end_rows_are_last_weekday_of_month():
    d = weekdays(300)
    flag = decision_rows(d)
    mon = d.astype("datetime64[M]")
    expected = np.zeros(len(d), bool)
    expected[:-1] = mon[1:] != mon[:-1]
    # the last row of the sample is a month end only if it is the last weekday
    assert np.array_equal(flag[:-1], expected[:-1])


def test_holiday_on_last_weekday_moves_decision_to_next_session():
    d = weekdays(300)
    m = d.astype("datetime64[M]")
    may = (m - d.astype("datetime64[Y]")).astype(int) == 4            # month index 4 = May
    holiday = may & (day_of_month(d) == 31)                            # 2010-05-31 Memorial Day (Monday)
    assert holiday.sum() == 1
    d2 = d[~holiday]
    flag = decision_rows(d2)
    i_fri = np.nonzero(may[~holiday])[0][-1]                           # Friday May 28
    assert not flag[i_fri]
    assert flag[i_fri + 1]                                             # June 1: one session late
    # June still gets its own month-end
    jun_rows = np.nonzero((d2.astype("datetime64[M]") - d2.astype("datetime64[Y]")).astype(int) == 5)[0]
    assert flag[jun_rows[-1]]
    assert flag[jun_rows].sum() == 2


# ---------------------------------------------------------------- warm-up / eligibility
@pytest.mark.parametrize("rv_days,first_row", [(21, 217), (10, 208)])
def test_first_cohort_after_200_valid_x(rv_days, first_row):
    # RV valid from row rv_min (row 0 has no return), IV session = t-1, x valid from rv_min+1,
    # 200 valid x in the 252-row window -> first z at rv_min + 200
    data = make_view(per_instrument_noise=True)
    rows, W, formed = cohorts(data, {"n_per_leg": 3, "rv_days": rv_days})
    assert np.array_equal(formed, rows >= first_row)


def test_ranking_and_inverse_vol_weights():
    data = make_view()
    t = last_decision(data)
    bump = {tk: 0.02 * j for j, tk in enumerate(TICKERS)}
    for tk in TICKERS:
        data.series[KEYS[tk]][t] *= np.exp(bump[tk])
    w = target_weights(data, PARAMS)
    expected = np.zeros(len(TICKERS))
    expected[-3:] = 1.0 / 18.0          # highest z: XLE EWZ FXI ... last three of TICKERS
    expected[:3] = -1.0 / 18.0          # lowest z
    rows, W, formed = cohorts(data, PARAMS)
    assert formed[-1]
    assert np.allclose(W[-1], expected)
    assert np.allclose(w[t], W[-3:].sum(axis=0))


def test_nonpositive_iv_is_ineligible():
    data = make_view()
    t = last_decision(data)
    data.series[KEYS["FXE"]][t] = -5.0
    rows, W, formed = cohorts(data, PARAMS)
    assert not formed[-1]


def test_ties_broken_by_ticker_order():
    data = make_view()             # identical z for all instruments
    rows, W, formed = cohorts(data, PARAMS)
    w = W[-1]
    alpha = sorted(TICKERS)
    longs = sorted(TICKERS[j] for j in np.nonzero(w > 0)[0])
    shorts = sorted(TICKERS[j] for j in np.nonzero(w < 0)[0])
    assert longs == alpha[:3]
    assert shorts == alpha[-3:]


def test_n_per_leg_changes_leg_size_and_threshold():
    data = make_view()
    rows, W2, f2 = cohorts(data, {"n_per_leg": 2, "rv_days": 21})
    assert ((W2[-1] > 0).sum(), (W2[-1] < 0).sum()) == (2, 2)
    assert np.isclose(W2[-1][W2[-1] > 0].sum(), 1 / 6) and np.isclose(W2[-1][W2[-1] < 0].sum(), -1 / 6)
    rows, W4, f4 = cohorts(data, {"n_per_leg": 4, "rv_days": 21})
    assert not f4.any()             # 4 per leg needs 15 eligible, only 12 exist


@pytest.mark.parametrize("frozen,eligible", [(4, True), (5, False)])
def test_stale_iv_makes_instrument_ineligible(frozen, eligible):
    # freezing the last `frozen` rows: the value on row t comes from session t - frozen - 1
    data = make_view()
    t = last_decision(data)
    s = data.series[KEYS["GLD"]]
    s[t - frozen + 1:t + 1] = s[t - frozen]
    rows, W, formed = cohorts(data, PARAMS)
    assert formed[-1] == eligible            # 12 eligible needed for n_per_leg = 3
    rows, W2, f2 = cohorts(data, {"n_per_leg": 2, "rv_days": 21})
    assert f2[-1]
    if not eligible:
        assert W2[-1][TICKERS.index("GLD")] == 0


def test_not_tradable_on_decision_day_excluded():
    data = make_view()
    t = last_decision(data)
    data.tradable[t, TICKERS.index("SPY")] = False
    rows, W, formed = cohorts(data, PARAMS)
    assert not formed[-1]


# ---------------------------------------------------------------- book mechanics
def test_book_is_sum_of_last_three_cohorts_and_constant_between():
    data = make_view(per_instrument_noise=True, seed=3)
    w = target_weights(data, PARAMS)
    rows, W, formed = cohorts(data, PARAMS)
    assert formed.sum() >= 4
    for k, t in enumerate(rows):
        exp = W[max(0, k - 2):k + 1].sum(axis=0)
        assert np.allclose(w[t], exp)
        assert np.abs(w[t]).sum() <= 1 + 1e-12
    dec = np.zeros(len(data.dates), bool)
    dec[rows] = True
    assert np.all(np.isnan(w[~dec]))
    for k in np.nonzero(formed)[0]:
        assert np.isclose(W[k][W[k] > 0].sum(), 1 / 6)
        assert np.isclose(W[k][W[k] < 0].sum(), -1 / 6)
        assert (W[k] != 0).sum() == 6


def test_inverse_vol_proportions():
    data = make_view(per_instrument_noise=True, seed=5)
    rows, W, formed = cohorts(data, PARAMS)
    k = np.nonzero(formed)[0][-1]
    t = rows[k]
    r = np.log1p(data.ret_oc[t - 62:t + 1])
    sig = r.std(axis=0, ddof=1)
    lg = W[k] > 0
    exp = (1 / sig[lg]) / (1 / sig[lg]).sum() / 6
    assert np.allclose(W[k][lg], exp)


def test_non_tradable_held_instrument_goes_to_zero_until_cohort_end():
    data = make_view(per_instrument_noise=True, seed=3)
    rows, W, formed = cohorts(data, PARAMS)
    k = np.nonzero(formed)[0][0]
    t0 = rows[k]
    j = int(np.nonzero(W[k] > 0)[0][0])
    data.tradable[t0 + 3, j] = False
    w = target_weights(data, PARAMS)
    rows2, W2, formed2 = cohorts(data, PARAMS)
    assert np.allclose(W2[k], W[k])
    assert np.isclose(w[t0, j], W[k][j])
    assert not np.isnan(w[t0 + 3, j])
    # from t0+3 on, cohort k no longer contributes to j while alive
    end = rows[k + 3] if k + 3 < len(rows) else len(data.dates)
    for kk in range(k + 1, min(k + 3, len(rows))):
        t = rows2[kk]
        if t < end:
            other = W2[max(0, kk - 2):kk + 1].sum(axis=0)[j] - W2[k][j]
            assert np.isclose(w[t, j], other)


def test_rv_days_changes_output():
    data = make_view(per_instrument_noise=True, seed=7)
    a = target_weights(data, {"n_per_leg": 3, "rv_days": 10})
    b = target_weights(data, {"n_per_leg": 3, "rv_days": 42})
    assert not np.allclose(np.nan_to_num(a), np.nan_to_num(b))


def test_point_in_time_truncation():
    data = make_view(per_instrument_noise=True, seed=11)
    full = target_weights(data, PARAMS)
    cut = 290
    short = DataView(
        dates=data.dates[:cut], instruments=data.instruments, asset_class=data.asset_class,
        ret_co=data.ret_co[:cut], ret_oc=data.ret_oc[:cut], tradable=data.tradable[:cut],
        listed=data.listed[:cut], delisting=data.delisting[:cut], close=data.close[:cut],
        dollar_volume=data.dollar_volume[:cut], universe=None, extras={},
        series={k: v[:cut] for k, v in data.series.items()}, cash_ret=data.cash_ret[:cut],
    )
    part = target_weights(short, PARAMS)
    assert np.array_equal(np.isnan(part), np.isnan(full[:cut]))
    assert np.allclose(np.nan_to_num(part), np.nan_to_num(full[:cut]))


def test_missing_inputs_raise():
    data = make_view()
    del data.series["cboe_etf_vol_1d.VXGDXCLS"]
    with pytest.raises(KeyError):
        target_weights(data, PARAMS)
