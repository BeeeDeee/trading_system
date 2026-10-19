import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS, compute_z, weekly_decision_rows, events

INSTR = ("SPY", "IEF", "TLT", "AGG", "SHY")
OIL = "fred_macro.DCOILWTICO"


def _bdays(n, start=10959):  # day 10959 = Monday 2000-01-03
    d = np.arange(start, start + n * 2 + 10)
    d = d[(d + 3) % 7 < 5][:n]
    return d.astype("datetime64[D]")


def make_view(T=1300, seed=0, ief_start=0, oil=None, dates=None):
    rng = np.random.default_rng(seed)
    N = len(INSTR)
    dates = _bdays(T) if dates is None else dates
    T = len(dates)
    ret_co = rng.normal(0, 0.003, (T, N))
    ret_oc = rng.normal(0, 0.008, (T, N))
    listed = np.ones((T, N), dtype=bool)
    listed[:ief_start, 1] = False
    ret_co[~listed] = 0.0
    ret_oc[~listed] = 0.0
    close = 100.0 * np.cumprod((1 + ret_co) * (1 + ret_oc), axis=0)
    close[~listed] = np.nan
    tradable = listed.copy()
    if oil is None:
        spy = (1 + ret_co[:, 0]) * (1 + ret_oc[:, 0]) - 1
        o_ret = 1.5 * spy + rng.normal(0, 0.02, T)
        oil = 30.0 * np.cumprod(1 + o_ret)
        # framework alignment (clock next_morning): row t holds the value of session t-1
        oil = np.concatenate(([np.nan], oil[:-1]))
    return DataView(
        dates=dates,
        instruments=INSTR,
        asset_class=tuple("us_etf" for _ in INSTR),
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=tradable,
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.where(listed, 1e9, np.nan),
        universe=None,
        extras={},
        series={OIL: np.asarray(oil, dtype=np.float64)},
        cash_ret=np.zeros(T),
    )


def reference_z(data, L):
    """Brute-force loop implementation of the card's z (unclipped)."""
    T = len(data.dates)
    p = np.asarray(data.series[OIL], dtype=float)
    pf = np.full(T, np.nan)
    last = np.nan
    for t in range(T):
        if np.isfinite(p[t]) and p[t] > 0:
            last = p[t]
        pf[t] = last
    o = np.full(T, np.nan)
    s = np.full(T, np.nan)
    for t in range(1, T):
        o[t] = pf[t] / pf[t - 1] - 1
        s[t] = (1 + data.ret_co[t - 1, 0]) * (1 + data.ret_oc[t - 1, 0]) - 1
    shock = np.full(T, np.nan)
    for t in range(T):
        if t - 251 < 0:
            continue
        xs, ys = s[t - 251:t + 1], o[t - 251:t + 1]
        if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
            continue
        beta = np.polyfit(xs, ys, 1)[0]
        ro = np.prod(1 + o[t - L + 1:t + 1]) - 1
        rs = np.prod(1 + s[t - L + 1:t + 1]) - 1
        shock[t] = ro - beta * rs
    z = np.full(T, np.nan)
    for t in range(T):
        if t - 755 < 0:
            continue
        w = shock[t - 755:t + 1]
        if np.all(np.isfinite(w)):
            z[t] = shock[t] / np.sqrt(np.mean(w * w))
    return z


def test_z_matches_reference():
    d = make_view(T=1100, seed=1)
    z, _, _ = compute_z(d, PARAMS)
    zr = reference_z(d, PARAMS["lookback_days"])
    assert np.array_equal(np.isfinite(z), np.isfinite(zr))
    assert np.isfinite(z).sum() > 50
    m = np.isfinite(z)
    assert np.allclose(z[m], zr[m], rtol=1e-6, atol=1e-8)
    # first defined z: oil row 0 is NaN (next_morning alignment), so the first oil return is row 2;
    # beta needs rows 2..253, then 756 shocks -> row 253 + 755
    assert np.argmax(np.isfinite(z)) == 253 + 755


def test_missing_and_nonpositive_oil_carry_forward():
    d0 = make_view(T=1100, seed=2)
    oil = np.array(d0.series[OIL], dtype=float)
    oil[400] = np.nan
    oil[401] = -5.0
    oil[700] = 0.0
    d = make_view(T=1100, seed=2, oil=oil)
    z, _, _ = compute_z(d, PARAMS)
    zr = reference_z(d, PARAMS["lookback_days"])
    m = np.isfinite(zr)
    assert np.array_equal(np.isfinite(z), m)
    assert np.allclose(z[m], zr[m], rtol=1e-6, atol=1e-8)


def test_weekly_decision_rows_fridays_and_holiday_fallback():
    dates = _bdays(30)
    dec = weekly_decision_rows(dates)
    wd = (dates.astype(np.int64) + 3) % 7
    assert np.array_equal(dec, wd == 4)
    # remove the second Friday (a holiday): the decision moves to the following Monday
    fridays = np.where(wd == 4)[0]
    keep = np.ones(30, dtype=bool)
    keep[fridays[1]] = False
    d2 = dates[keep]
    dec2 = weekly_decision_rows(d2)
    wd2 = (d2.astype(np.int64) + 3) % 7
    exp = wd2 == 4
    exp[fridays[1]] = True  # the Monday now sits at the old Friday's index
    assert wd2[fridays[1]] == 0
    assert np.array_equal(dec2, exp)
    assert dec2[fridays[1] - 1] == False  # the Thursday itself is not known to be last


def test_weights_shape_and_rules():
    d = make_view(T=1300, seed=3)
    w = target_weights(d, PARAMS)
    assert w.shape == (1300, 5)
    dec = weekly_decision_rows(d.dates)
    assert np.all(np.isnan(w[~dec]))
    assert np.all(np.isfinite(w[dec]))
    assert np.all(w[dec] >= 0)
    assert np.all(np.abs(w[dec]).sum(axis=1) <= 1 + 1e-12)
    assert np.all(w[dec][:, 2:] == 0)
    z, _, _ = compute_z(d, PARAMS)
    live = dec & np.isfinite(z)
    zc = np.clip(z[live], -2, 2)
    exp = np.clip(0.60 - 0.15 * zc, 0.2, 1.0)
    assert np.allclose(w[live, 0], exp)
    assert np.allclose(w[live, 1], 1 - exp)
    # before z is defined: flat
    pre = dec & (np.arange(1300) < 253 + 755)
    assert pre.sum() > 100
    assert np.all(w[pre] == 0)


def test_warmup_waits_for_ief_and_z_nan_later_is_6040():
    d = make_view(T=1300, seed=4, ief_start=1150)
    w = target_weights(d, PARAMS)
    dec = weekly_decision_rows(d.dates)
    rows = np.where(dec)[0]
    first_live = rows[rows >= 1150][0]
    assert np.all(w[rows[rows < first_live]] == 0)
    assert w[first_live].sum() == pytest.approx(1.0)
    # an SPY NaN return makes z undefined for a while after the start: hold 60/40
    d2 = make_view(T=1300, seed=4)
    d2.ret_oc[1200, 0] = np.nan
    w2 = target_weights(d2, PARAMS)
    nxt = rows[(rows > 1202) & (rows < 1250)]
    assert len(nxt) > 0
    assert np.allclose(w2[nxt, 0], 0.6)
    assert np.allclose(w2[nxt, 1], 0.4)


def test_params_change_output():
    d = make_view(T=1300, seed=5)
    w = target_weights(d, PARAMS)
    w_t = target_weights(d, dict(PARAMS, tilt_per_z=0.2))
    w_l = target_weights(d, dict(PARAMS, lookback_days=42))
    dec = np.isfinite(w[:, 0]) & (w[:, 0] + w[:, 1] > 0)
    assert not np.allclose(w[dec, 0], w_t[dec, 0])
    assert not np.allclose(w[dec, 0], w_l[dec, 0])


def test_point_in_time_truncation():
    d = make_view(T=1300, seed=6)
    w = target_weights(d, PARAMS)
    cut = 1150
    oil = np.array(d.series[OIL], dtype=float)
    oil2 = oil.copy()
    oil2[cut + 1:] = np.random.default_rng(9).uniform(-10, 100, 1300 - cut - 1)
    d2 = make_view(T=1300, seed=6, oil=oil2)
    d2.ret_oc[cut + 1:] = np.random.default_rng(10).normal(0, 0.05, (1300 - cut - 1, 5))
    w2 = target_weights(d2, PARAMS)
    a, b = w[:cut + 1], w2[:cut + 1]
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])


def test_diagnostic_events():
    d = make_view(T=1300, seed=7)
    mask, side = events(d, PARAMS)
    z, _, _ = compute_z(d, PARAMS)
    dec = weekly_decision_rows(d.dates)
    exp = dec & np.isfinite(z) & (np.abs(np.nan_to_num(z)) >= 1.0)
    assert np.array_equal(mask[:, 0], exp)
    assert not mask[:, 1:].any()
    assert exp.sum() > 0
    assert np.all(side[exp, 0] == -np.sign(z[exp]))
