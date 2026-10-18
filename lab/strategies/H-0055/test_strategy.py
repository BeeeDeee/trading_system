import numpy as np
import pytest
from lab.framework.data import DataView

from strategy import target_weights, PARAMS
from strategy import decision_schedule, spy_realized_variance, usable_vix, lagged_vrp, percentile_ranks

INSTR = ("SPY", "IEF", "TLT", "SHY", "AGG")
MONDAY0 = 4 + 7 * 1800          # day 4 since the epoch is a Monday


def weekdays(n_rows, drop=()):
    """n_rows weekdays starting on a Monday, minus the weekday indices in `drop` (holidays)."""
    out = []
    k = 0
    while len(out) < n_rows:
        d = MONDAY0 + k
        if (d + 3) % 7 < 5 and k not in drop:
            out.append(d)
        k += 1
    return np.array(out, dtype=np.int64).astype("datetime64[D]")


def make_view(T, vix, spy_oc=0.001, close_nan_rows=(), untradable=None, drop=()):
    N = len(INSTR)
    dates = weekdays(T, drop=drop)
    ret_co = np.zeros((T, N))
    ret_oc = np.full((T, N), 0.0005)
    ret_oc[:, 0] = spy_oc
    close = np.full((T, N), 100.0)
    tradable = np.ones((T, N), dtype=bool)
    for r in close_nan_rows:
        close[r, 0] = np.nan
        ret_co[r, 0] = 0.0
        ret_oc[r, 0] = 0.0
        if r + 1 < T:
            tradable[r + 1, 0] = False
    if untradable:
        for (r, j) in untradable:
            tradable[r, j] = False
    return DataView(
        dates=dates, instruments=INSTR, asset_class=tuple("us_etf" for _ in INSTR),
        ret_co=ret_co, ret_oc=ret_oc, tradable=tradable, listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool), close=close, dollar_volume=np.full((T, N), 1e9),
        universe=None, extras={}, series={"fred_macro.VIXCLS": np.asarray(vix, dtype=float)},
        cash_ret=np.zeros(T),
    )


def rising_vix(T):
    return 10.0 + 0.01 * np.arange(T)


# ---------------------------------------------------------------- calendar

def test_schedule_fridays_and_holiday_fallback():
    # 3 weeks; weekday index 11 (= Friday of week 2) is a holiday
    dates = weekdays(14, drop=(11,))
    dec, src = decision_schedule(dates)
    # rows: week1 Mon..Fri = 0..4, week2 Mon..Thu = 5..8, week3 Mon.. = 9..13
    assert list(dec) == [4, 9, 13]
    assert list(src) == [4, 8, 13]


def test_schedule_is_point_in_time():
    dates = weekdays(40, drop=(11, 25))
    dec, src = decision_schedule(dates)
    for cut in range(1, 40):
        d2, s2 = decision_schedule(dates[:cut])
        keep = dec < cut
        assert list(d2) == list(dec[keep]) and list(s2) == list(src[keep])


# ---------------------------------------------------------------- inputs

def test_realized_variance_full_partial_and_missing():
    T = 40
    v = make_view(T, rising_vix(T), spy_oc=0.01)
    rv = spy_realized_variance(v)
    sq = (100 * np.log1p(0.01)) ** 2
    assert np.all(np.isnan(rv[:14]))                  # fewer than 15 returns
    assert rv[14] == pytest.approx(sq * 21)           # 15 present, rescaled 21/15
    assert rv[30] == pytest.approx(sq * 21)
    # 3 missing bars in the window ending at row 30: still 21/18 rescaled to the same value
    v2 = make_view(T, rising_vix(T), spy_oc=0.01, close_nan_rows=(25, 26, 27))
    rv2 = spy_realized_variance(v2)
    assert rv2[30] == pytest.approx(sq * 21)
    # 7 missing in the window -> 14 present -> missing
    v3 = make_view(T, rising_vix(T), spy_oc=0.01, close_nan_rows=tuple(range(20, 27)))
    assert np.isnan(spy_realized_variance(v3)[30])
    assert np.isnan(spy_realized_variance(v3)[39])     # window 19..39 still holds the 7 gaps
    v4 = make_view(50, rising_vix(50), spy_oc=0.01, close_nan_rows=tuple(range(20, 27)))
    assert spy_realized_variance(v4)[47] == pytest.approx(sq * 21)   # window 27..47 complete


def test_realized_variance_uses_co_and_oc():
    T = 30
    v = make_view(T, rising_vix(T), spy_oc=0.01)
    v.ret_co[:, 0] = 0.02
    sq = (100 * (np.log1p(0.02) + np.log1p(0.01))) ** 2
    assert spy_realized_variance(v)[25] == pytest.approx(21 * sq)


def test_vix_carry_at_most_five_sessions():
    T = 30
    vix = rising_vix(T)
    vix[10:15] = np.nan                               # 5 missing -> carried
    vix[20:26] = np.nan                               # 6 missing -> 6th is missing
    v = make_view(T, vix)
    u = usable_vix(v)
    assert np.all(u[10:15] == vix[9])
    assert np.all(u[20:25] == vix[19]) and np.isnan(u[25])
    assert u[26] == vix[26]
    # a forward-filled series (repeated value) is treated the same way
    vix2 = rising_vix(T)
    vix2[20:26] = vix2[19]
    u2 = usable_vix(make_view(T, vix2))
    assert np.all(u2[20:25] == vix2[19]) and np.isnan(u2[25])


def test_missing_vix_series_raises():
    T = 30
    v = make_view(T, rising_vix(T))
    v.series.clear()
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)


def test_vrp_alignment():
    T = 60
    vix = rising_vix(T)
    v = make_view(T, vix, spy_oc=0.01)
    w = lagged_vrp(v)
    rv = spy_realized_variance(v)
    assert np.isnan(w[0])
    for t in (20, 40, 59):
        assert w[t] == pytest.approx(vix[t] ** 2 / 12 - rv[t - 1])


# ---------------------------------------------------------------- percentile

def test_percentile_rank_hand_cases():
    w = np.arange(400, dtype=float)
    assert percentile_ranks(w, [399], 300)[0] == pytest.approx(1.0)
    wd = -np.arange(400, dtype=float)
    assert percentile_ranks(wd, [399], 300)[0] == pytest.approx(1 / 300)
    assert percentile_ranks(wd, [399], 1000)[0] == pytest.approx(1 / 400)   # expanding
    assert np.isnan(percentile_ranks(wd, [250], 1000)[0])                    # 251 values < 252
    assert percentile_ranks(wd, [251], 1000)[0] == pytest.approx(1 / 252)
    wn = wd.copy()
    wn[:200] = np.nan
    assert np.isnan(percentile_ranks(wn, [399], 1000)[0])                    # 200 valid only
    wn[399] = np.nan
    assert np.isnan(percentile_ranks(wd * np.nan, [399], 300)[0])
    # ties count as <=
    wt = np.zeros(300)
    assert percentile_ranks(wt, [299], 300)[0] == pytest.approx(1.0)


# ---------------------------------------------------------------- weights

def test_weights_warmup_and_tilt():
    T = 600
    v = make_view(T, rising_vix(T))
    w = target_weights(v, PARAMS)
    spy, ief = 0, 1
    assert w[0, spy] == pytest.approx(0.6) and w[0, ief] == pytest.approx(0.4)
    dec, src = decision_schedule(v.dates)
    # rising VRP -> p = 1 -> SPY = 0.6 + tilt once 252 valid values exist (valid from row 15)
    late = dec[src >= 15 + 251]
    early = dec[(src < 15 + 251) & (dec > 0)]
    assert np.all(np.isnan(w[early]))                 # warm-up: hold 60/40
    assert np.allclose(w[late, spy], 0.9) and np.allclose(w[late, ief], 0.1)
    assert np.all(w[late][:, 2:] == 0)
    nondec = np.setdiff1d(np.arange(1, T), dec)
    assert np.all(np.isnan(w[nondec]))
    rows = ~np.all(np.isnan(w), axis=1)
    assert np.all(np.abs(w[rows]).sum(axis=1) <= 1 + 1e-12)
    assert np.all(w[rows] >= 0)
    # tilt changes the weights
    w2 = target_weights(v, {"rank_days": 756, "tilt": 0.2})
    assert np.allclose(w2[late, spy], 0.8)
    # falling VRP -> p small -> SPY near 0.6 - tilt
    vf = make_view(T, 40.0 - 0.01 * np.arange(T))
    wf = target_weights(vf, PARAMS)
    assert np.all(wf[late, spy] < 0.31) and np.all(wf[late, spy] >= 0.3)


def test_rank_days_changes_output():
    T = 1100
    t = np.arange(T)
    vix = np.where(t < 500, 10 + 0.02 * t, 20 - 0.02 * (t - 500))
    v = make_view(T, vix)
    a = target_weights(v, {"rank_days": 504, "tilt": 0.3})
    b = target_weights(v, {"rank_days": 1008, "tilt": 0.3})
    both = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
    assert np.any(np.abs(a[both, 0] - b[both, 0]) > 1e-9)


def test_missing_vrp_holds():
    T = 600
    vix = rising_vix(T)
    vix[490:510] = np.nan                             # > 5 sessions: VRP missing
    v = make_view(T, vix)
    w = target_weights(v, PARAMS)
    dec, src = decision_schedule(v.dates)
    miss = dec[(src >= 496) & (src < 510)]
    assert len(miss) > 0 and np.all(np.isnan(w[miss]))


def test_trade_deferred_when_bar_missing():
    T = 600
    v0 = make_view(T, rising_vix(T))
    dec, src = decision_schedule(v0.dates)
    r = int(dec[dec > 400][0])
    v = make_view(T, rising_vix(T), untradable=[(r + 1, 1), (r + 2, 0)])
    w = target_weights(v, PARAMS)
    assert w[r, 0] == pytest.approx(0.9)
    assert w[r + 1, 0] == pytest.approx(0.9) and w[r + 2, 0] == pytest.approx(0.9)
    assert np.all(np.isnan(w[r + 3]))


def test_holiday_week_decision_on_monday_uses_thursday_signal():
    T = 600
    drop = (7 * 60 + 4,)                              # a Friday holiday
    v = make_view(T, rising_vix(T), drop=drop)
    dec, src = decision_schedule(v.dates)
    k = np.nonzero(src != dec)[0]
    assert len(k) == 1 and src[k[0]] == dec[k[0]] - 1
    w = target_weights(v, PARAMS)
    assert w[dec[k[0]], 0] == pytest.approx(0.9)

