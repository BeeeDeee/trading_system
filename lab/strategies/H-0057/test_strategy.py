import math

import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS
from strategy import CFTC_CODE, DATASET, report_rows, market_z, select_weights
from strategy import events

ETFS = list(CFTC_CODE.keys())
SMALL = {"avg_reports": 2, "z_window_weeks": 8}


def _weekday(day):
    return (day - 4) % 7          # 0 = Monday (day 4 = 1970-01-05 was a Monday)


def make_view(n_cal_days=700, seed=0, ffill=True, listing=None, gap=None, drop_key=None):
    """18 card ETFs (shuffled order) plus one non-universe column, weekly COT reports released on Fridays."""
    rng = np.random.default_rng(seed)
    all_days = 13149 + np.arange(n_cal_days)            # 13149 = 2006-01-01
    days = all_days[_weekday(all_days) < 5]
    dates = days.astype("datetime64[D]")
    T = dates.shape[0]
    names = ETFS + ["MKT"]
    perm = rng.permutation(len(names))
    instruments = tuple(names[i] for i in perm)
    N = len(instruments)
    ret_co = rng.normal(0.0, 0.003, (T, N))
    ret_oc = rng.normal(0.0, 0.01, (T, N))
    listed = np.ones((T, N), dtype=bool)
    if listing:
        for name, row in listing.items():
            j = instruments.index(name)
            listed[:row, j] = False
    ret_co[~listed] = 0.0
    ret_oc[~listed] = 0.0
    close = 50.0 * np.cumprod((1 + ret_co) * (1 + ret_oc), axis=0)
    close[~listed] = np.nan
    dollar_volume = np.where(listed, 1e7, np.nan)
    tradable = listed.copy()
    series = {}
    friday = _weekday(days) == 4
    if gap is not None:
        friday = friday & ~((np.arange(T) >= gap[0]) & (np.arange(T) < gap[1]))
    for e in ETFS:
        code = CFTC_CODE[e]
        oi = rng.uniform(1e5, 2e5, T)
        nl = rng.uniform(1e4, 6e4, T)
        ns = rng.uniform(1e4, 6e4, T)
        out = {}
        for key, x in (("nc_long", nl), ("nc_short", ns), ("open_interest", oi)):
            v = np.where(friday, x, np.nan)
            if ffill:
                idx = np.where(friday, np.arange(T), 0)
                idx = np.maximum.accumulate(idx)
                v = np.where(np.maximum.accumulate(friday), x[idx], np.nan)
            out[key] = v
        for key, v in out.items():
            series[DATASET + "." + code + "." + key] = v
    if drop_key is not None:
        del series[drop_key]
    return DataView(
        dates=dates,
        instruments=instruments,
        asset_class=tuple("us_etf" for _ in instruments),
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=tradable,
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=dollar_volume,
        universe=None,
        extras={},
        series=series,
        cash_ret=np.zeros(T),
    )


# ---------------------------------------------------------------- pure helpers

def test_report_rows_ffill_and_sparse_agree():
    nl = np.array([np.nan, 10, 10, 10, 12, 12, 12, 9.0])
    ns = np.array([np.nan, 5, 5, 5, 5, 5, 5, 5.0])
    oi = np.array([np.nan, 100, 100, 100, 100, 100, 100, 100.0])
    rows, s = report_rows(nl, ns, oi)
    assert rows.tolist() == [1, 4, 7]
    assert np.allclose(s, [0.05, 0.07, 0.04])
    sp = np.full(8, np.nan)
    nl2, ns2, oi2 = sp.copy(), sp.copy(), sp.copy()
    for r, a in zip([1, 4, 7], [10, 12, 9]):
        nl2[r], ns2[r], oi2[r] = a, 5, 100
    rows2, s2 = report_rows(nl2, ns2, oi2)
    assert rows2.tolist() == [1, 4, 7]
    assert np.allclose(s2, s)


def test_report_rows_skips_nonpositive_open_interest():
    rows, s = report_rows(np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([0.0, 10.0]))
    assert rows.tolist() == [1]


def test_market_z_by_hand():
    day = np.arange(40) * 1
    rows = np.array([0, 7, 14, 21])
    s = np.array([1.0, 2.0, 3.0, 4.0])
    z = market_z(rows, s, day, 21, 2, 4)
    expected = (3.5 - 2.5) / np.std([1, 2, 3, 4], ddof=1)
    assert z == pytest.approx(expected)
    # only reports released on or before t: at t = 20 the last report is row 14
    z20 = market_z(rows, s, day, 20, 2, 3)
    assert z20 == pytest.approx((2.5 - 2.0) / 1.0)
    # coverage: window 8 needs ceil(6) reports, only 4 exist
    assert np.isnan(market_z(rows, s, day, 21, 2, 8))
    # window 5 needs ceil(3.75) = 4 reports: valid with 4
    assert np.isfinite(market_z(rows, s, day, 21, 2, 5))
    # stale: latest report more than 21 calendar days old
    assert np.isnan(market_z(rows, s, np.arange(60), 43, 2, 4))
    assert np.isfinite(market_z(rows, s, np.arange(60), 42, 2, 4))
    # not enough reports for the average
    assert np.isnan(market_z(rows[:1], s[:1], day, 21, 2, 1))


def test_select_nine_eligible_odd_split():
    # 9 eligible + 1 ineligible column; low-return half gets 5 (the middle one), high half 4; k = 1 each
    ret = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, -1.0])
    z = np.array([0.5, -1.0, 2.0, 0.0, 1.0, -2.0, 3.0, 0.1, 0.2, 9.0])
    vol = np.array([0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.04, 0.01, 0.01, 0.01])
    elig = np.ones(10, dtype=bool)
    elig[9] = False
    w = select_weights(z, ret, vol, elig)
    # low half cols 0..4: long col 2 (z 2.0), short col 1 (z -1.0); high half cols 5..8: long 6, short 5
    nz = set(np.flatnonzero(w).tolist())
    assert nz == {1, 2, 5, 6}
    # longs 2 (vol .01) and 6 (vol .04): inverse vol 100 and 25
    assert w[2] == pytest.approx(0.5 * 100 / 125)
    assert w[6] == pytest.approx(0.5 * 25 / 125)
    # shorts 1 (vol .02) and 5 (vol .01): inverse vol 50 and 100
    assert w[1] == pytest.approx(-0.5 * 50 / 150)
    assert w[5] == pytest.approx(-0.5 * 100 / 150)
    assert w.sum() == pytest.approx(0.0)
    assert np.abs(w).sum() == pytest.approx(1.0)


def test_select_eighteen_k_two():
    ret = np.arange(18) / 100.0
    z = np.array([5, 4, 3, 2, 1, 0, -1, -2, -3, 9, 8, 7, 6, -6, -7, -8, -9, 0.5])
    vol = np.full(18, 0.01)
    w = select_weights(z, ret, vol, np.ones(18, dtype=bool))
    # low half 0..8: long 0, 1; short 7, 8. high half 9..17: long 9, 10; short 15, 16
    assert set(np.flatnonzero(w > 0).tolist()) == {0, 1, 9, 10}
    assert set(np.flatnonzero(w < 0).tolist()) == {7, 8, 15, 16}
    assert np.allclose(w[w > 0], 0.125)
    assert np.allclose(w[w < 0], -0.125)


def test_select_fifteen_mixed_k():
    ret = np.arange(15) / 100.0
    z = np.arange(15)[::-1].astype(float)
    w = select_weights(z, ret, np.full(15, 0.01), np.ones(15, dtype=bool))
    # low half 0..7 (8 ETFs, k = 2): long 0, 1, short 6, 7; high half 8..14 (7, k = 1): long 8, short 14
    assert set(np.flatnonzero(w > 0).tolist()) == {0, 1, 8}
    assert set(np.flatnonzero(w < 0).tolist()) == {6, 7, 14}
    assert (w != 0).sum() == 6


def test_select_fewer_than_eight_is_flat():
    elig = np.zeros(18, dtype=bool)
    elig[:7] = True
    w = select_weights(np.arange(18.0), np.arange(18.0), np.full(18, 0.01), elig)
    assert np.all(w == 0)


# ---------------------------------------------------------------- full strategy

def _first_report_session_per_month(view):
    day = view.dates.astype(np.int64)
    month = view.dates.astype("datetime64[M]").astype(np.int64)
    fri = np.flatnonzero(_weekday(day) == 4)
    out = []
    seen = set()
    for t in fri:
        if month[t] not in seen:
            seen.add(month[t])
            out.append(t)
    return out


def test_decision_rows_and_book_shape():
    view = make_view()
    w = target_weights(view, SMALL)
    T, N = w.shape
    assert (T, N) == (len(view.dates), len(view.instruments))
    decided = np.flatnonzero(~np.all(np.isnan(w), axis=1))
    assert decided.tolist() == _first_report_session_per_month(view)
    assert not np.any(np.isnan(w[decided]))
    active = [t for t in decided if np.abs(w[t]).sum() > 0]
    assert len(active) >= 15
    mkt = view.col("MKT")
    for t in active:
        assert np.abs(w[t]).sum() == pytest.approx(1.0)
        assert w[t].sum() == pytest.approx(0.0)
        assert w[t][w[t] > 0].sum() == pytest.approx(0.5)
        assert (w[t] != 0).sum() == 8          # 18 eligible: halves of 9, k = 2
        assert w[t, mkt] == 0


def test_warmup_rows_are_flat():
    view = make_view()
    w = target_weights(view, SMALL)
    decided = np.flatnonzero(~np.all(np.isnan(w), axis=1))
    # the first decision (January, < 63 sessions) has no positions
    assert np.all(w[decided[0]] == 0)


def test_late_listing_not_eligible_until_63_sessions():
    start = 300
    view = make_view(listing={"CPER": start})
    w = target_weights(view, SMALL)
    j = view.col("CPER")
    rows = np.flatnonzero(~np.all(np.isnan(w), axis=1))
    before = [t for t in rows if t < start + 62]
    after = [t for t in rows if t >= start + 62 and np.abs(w[t]).sum() > 0]
    assert all(w[t, j] == 0 for t in before)
    # 17 eligible before: halves 9 and 8, k = 2 each
    assert all((w[t] != 0).sum() == 8 for t in before if np.abs(w[t]).sum() > 0)
    assert len(after) > 0


def test_too_few_eligible_means_flat():
    late = {e: 400 for e in ETFS[:11]}            # only 7 ETFs listed before row 400
    view = make_view(listing=late)
    w = target_weights(view, SMALL)
    rows = np.flatnonzero(~np.all(np.isnan(w), axis=1))
    assert all(np.all(w[t] == 0) for t in rows if t < 400)


def test_shutdown_month_keeps_positions():
    view0 = make_view()
    months = view0.dates.astype("datetime64[M]").astype(np.int64)
    m = months[350]
    rows_m = np.flatnonzero(months == m)
    view = make_view(gap=(rows_m[0], rows_m[-1] + 1))
    w = target_weights(view, SMALL)
    assert np.all(np.isnan(w[rows_m]))
    # sparse representation gives the same book
    w_sparse = target_weights(make_view(gap=(rows_m[0], rows_m[-1] + 1), ffill=False), SMALL)
    assert np.array_equal(np.isnan(w), np.isnan(w_sparse))
    assert np.allclose(np.nan_to_num(w), np.nan_to_num(w_sparse))


def test_stale_report_is_not_eligible():
    view = make_view()
    code = CFTC_CODE["GLD"]
    T = len(view.dates)
    for key in ("nc_long", "nc_short", "open_interest"):
        v = view.series[DATASET + "." + code + "." + key]
        v[250:] = v[250]                       # GLD reports stop: forward-filled value only
    w = target_weights(view, SMALL)
    j = view.col("GLD")
    day = view.dates.astype(np.int64)
    rows = np.flatnonzero(~np.all(np.isnan(w), axis=1))
    late = [t for t in rows if day[t] - day[250] > 28]
    assert late and all(w[t, j] == 0 for t in late)
    assert all((w[t] != 0).sum() == 8 for t in late if np.abs(w[t]).sum() > 0)


def test_params_change_output():
    view = make_view(n_cal_days=900)
    base = target_weights(view, SMALL)
    a = target_weights(view, {"avg_reports": 4, "z_window_weeks": 8})
    b = target_weights(view, {"avg_reports": 2, "z_window_weeks": 20})
    assert not np.allclose(np.nan_to_num(base), np.nan_to_num(a))
    assert not np.allclose(np.nan_to_num(base), np.nan_to_num(b))
    # longer window needs ceil(0.75 * 20) = 15 reports: first active decision is later
    first = lambda w: min(t for t in range(len(w)) if np.nansum(np.abs(w[t])) > 0)
    assert first(b) > first(base)


def test_primary_params_run():
    view = make_view(n_cal_days=1400)
    w = target_weights(view, PARAMS)
    assert w.shape == (len(view.dates), len(view.instruments))
    gross = np.nansum(np.abs(w), axis=1)
    assert np.all(gross <= 1 + 1e-9)
    # 156-report window needs 117 reports (~2.25 years) before any position
    day = view.dates.astype(np.int64)
    active = [t for t in range(len(w)) if gross[t] > 0]
    assert active and day[active[0]] - day[0] >= 116 * 7


def test_point_in_time_truncation():
    view = make_view(seed=3)
    full = target_weights(view, SMALL)
    for cut in (200, 333, 410):
        sub = make_view(seed=3)
        T = cut + 1
        trunc = DataView(
            dates=sub.dates[:T], instruments=sub.instruments, asset_class=sub.asset_class,
            ret_co=sub.ret_co[:T], ret_oc=sub.ret_oc[:T], tradable=sub.tradable[:T], listed=sub.listed[:T],
            delisting=sub.delisting[:T], close=sub.close[:T], dollar_volume=sub.dollar_volume[:T],
            universe=None, extras={}, series={k: v[:T] for k, v in sub.series.items()}, cash_ret=sub.cash_ret[:T],
        )
        part = target_weights(trunc, SMALL)
        assert np.array_equal(np.isnan(part), np.isnan(full[:T]))
        assert np.allclose(np.nan_to_num(part), np.nan_to_num(full[:T]))


def test_deterministic():
    view = make_view(seed=5)
    a = target_weights(view, SMALL)
    b = target_weights(view, SMALL)
    assert np.array_equal(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9))


def test_missing_series_raises():
    view = make_view(drop_key=DATASET + "." + CFTC_CODE["FXF"] + ".open_interest")
    with pytest.raises(KeyError):
        target_weights(view, SMALL)


def test_diagnostic_matches_positions():
    view = make_view(seed=7)
    w = target_weights(view, SMALL)
    mask, side = events(view, SMALL)
    w0 = np.nan_to_num(w)
    assert np.array_equal(mask, w0 != 0)
    assert np.array_equal(side[mask], np.sign(w0[mask]))
