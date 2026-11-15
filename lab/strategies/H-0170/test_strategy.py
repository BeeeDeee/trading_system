import dataclasses

import numpy as np
import pytest

import strategy as S
from strategy import target_weights, PARAMS
from strategy import (
    _decision_days, _ffill, _lag_diff, _trailing_sd, _shock_z, _market_move,
    _book_to_market, _revenue_growth, _winsorize_rows,
)
from lab.framework.data import synthetic


def _view(seed=1, late_member=True):
    v = synthetic()
    T, N = len(v.dates), len(v.instruments)
    rng = np.random.default_rng(seed)
    uni = np.ones((T, N), dtype=bool)
    if late_member:
        uni[: T // 2, 0] = False          # joins the universe late
        uni[T // 3:, 1] = False           # leaves the universe
    ex = dict(v.extras) if v.extras else {}
    ex["sf1_art_divyield"] = rng.uniform(0, 0.05, (T, N)).astype(np.float32)
    ex["sf1_art_pb"] = rng.uniform(0.5, 6, (T, N)).astype(np.float32)
    ex["sf1_art_roe"] = rng.normal(0.1, 0.1, (T, N)).astype(np.float32)
    ex["sf1_art_revenue"] = rng.uniform(1e8, 2e8, (T, N)).astype(np.float32)
    ex["sf1_art_roe"][:, 2] = np.nan     # never scored
    se = dict(v.series) if v.series else {}
    se["fred_macro.DFII10"] = np.cumsum(rng.normal(0, 0.05, T))
    return dataclasses.replace(v, universe=uni, extras=ex, series=se)


def _core(v):
    m = np.asarray(v.universe, bool) & np.isfinite(np.asarray(v.close, float))
    return np.where(m, 1.0 / np.maximum(m.sum(1), 1)[:, None], 0.0)


def test_no_shock_holds_equal_weight_core_on_fridays():
    v = _view()
    w = target_weights(v, {"n_estimators": 20, "shock_z": 1e9})
    dec = _decision_days(v.dates)
    assert np.all(np.isnan(w[~dec]))
    assert np.allclose(w[dec], _core(v)[dec])
    assert np.all(w[dec][~np.asarray(v.universe, bool)[dec]] == 0)


def test_tilt_rows_hold_top_fifth_long_only(monkeypatch):
    monkeypatch.setattr(S, "MIN_SCORED", 5)
    v = _view()
    p = {"n_estimators": 20, "shock_z": 1.0}
    w = target_weights(v, p)
    dec = _decision_days(v.dates)
    uni = np.asarray(v.universe, bool)
    wd = w[dec]
    assert np.all(wd >= 0)
    assert np.all(wd.sum(1) <= 1 + 1e-9)
    assert np.all(wd[~uni[dec]] == 0)
    score, shock, _ = S._scores(v, p)
    core = _core(v)
    tilt_rows = np.flatnonzero(dec & shock & (np.isfinite(score).sum(1) >= 5))
    assert tilt_rows.size > 0
    # first tilt row needs >= 200 valid 5-session changes first
    assert tilt_rows[0] >= 204
    for t in tilt_rows[:5]:
        held = w[t] > 0
        n_sc = int(np.isfinite(score[t]).sum())
        assert held.sum() <= max(1, int(np.ceil(0.2 * n_sc))) and held.sum() >= 1
        assert np.isclose(w[t].sum(), 1.0)
        assert not held[2]                       # stock with a missing feature is never held in a tilt
        assert np.min(score[t][held]) >= np.max(np.where(held, -np.inf, np.nan_to_num(score[t], nan=-np.inf)))
    other = dec.copy()
    other[tilt_rows] = False
    assert np.allclose(w[other], core[other])
    # shock_z changes the book
    w2 = target_weights(v, {"n_estimators": 20, "shock_z": 1.5})
    assert not np.allclose(np.nan_to_num(w, nan=-1), np.nan_to_num(w2, nan=-1))
    # scores exist only on shock-week decision rows
    assert np.all(np.isnan(score[~(dec & shock)]))


def _bdays(start_days, n):
    # business days counted from an integer day number (no date strings)
    d0 = np.arange(start_days, start_days + 1).astype("datetime64[D]")[0]
    return np.busday_offset(d0, np.arange(n), roll="forward")


def test_params_match_card():
    assert PARAMS == {"n_estimators": 100, "shock_z": 1.0}


def test_decision_days_are_fridays():
    d = _bdays(10, 15)  # 1970-01-12 is a Monday (day 11); day 10 is a Sunday -> rolls to Monday
    dec = _decision_days(d)
    wd = (d.astype(np.int64) + 3) % 7
    assert np.array_equal(dec, wd == 4)
    assert dec.sum() == 3
    # a week with a missing Friday has no decision
    d2 = np.delete(d, np.flatnonzero(wd == 4)[0])
    assert _decision_days(d2).sum() == 2


def test_ffill_and_lag_diff():
    x = np.array([np.nan, 1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
    f = _ffill(x)
    assert np.isnan(f[0])
    assert np.allclose(f[1:], [1, 1, 3, 3, 3, 6])
    d = _lag_diff(np.arange(8.0), 5)
    assert np.all(np.isnan(d[:5])) and np.allclose(d[5:], 5.0)


def test_trailing_sd_min_count_and_window():
    x = np.arange(10.0)
    sd = _trailing_sd(x, 4, 3)
    assert np.all(np.isnan(sd[:2]))
    assert np.isclose(sd[2], np.std([0, 1, 2], ddof=1))
    assert np.isclose(sd[9], np.std([6, 7, 8, 9], ddof=1))
    x[8] = np.nan
    sd = _trailing_sd(x, 4, 4)
    assert np.isnan(sd[9]) and np.isnan(sd[10 - 1])


def test_shock_z_needs_200_values_and_is_point_in_time():
    rng = np.random.default_rng(0)
    r = np.cumsum(rng.normal(0, 0.05, 400))
    z = _shock_z(r)
    # first 5-session change at row 5; 200 valid changes first at row 204
    assert np.all(np.isnan(z[:204])) and np.isfinite(z[204])
    s = r[300] - r[295]
    ch = r[5:301] - r[:296]
    win = ch[-252:]
    assert np.isclose(z[300], s / np.std(win, ddof=1))
    # truncation does not change earlier values
    assert np.allclose(_shock_z(r[:301])[:301], z[:301], equal_nan=True)
    # a missing day carries the last value
    r2 = r.copy()
    r2[300] = np.nan
    assert np.isclose(_shock_z(r2)[300], (r[299] - r[295]) / np.nanstd(
        np.where(np.arange(296) == 295, r[299] - r[295], ch)[-252:], ddof=1))


def test_market_move():
    T, N = 260, 3
    idx = np.ones((T, N))
    g = np.linspace(0.0, 0.02, T)
    idx[:, 0] = np.cumprod(1 + g)
    idx[:, 1] = np.cumprod(1 + 2 * g)
    idx[:, 2] = np.cumprod(1 + 10 * g)
    members = np.ones((T, N), bool)
    members[:, 2] = False          # non-member excluded from the mean
    listed = np.ones((T, N), bool)
    m = _market_move(idx, members, listed)
    t = 259
    mean5 = np.mean([idx[t, 0] / idx[t - 5, 0] - 1, idx[t, 1] / idx[t - 5, 1] - 1])
    seq = []
    for u in range(t - 251, t + 1):
        if u >= 5:
            seq.append(np.mean([idx[u, j] / idx[u - 5, j] - 1 for j in (0, 1)]))
    assert np.isclose(m[t], mean5 / np.std(seq, ddof=1))
    assert np.all(np.isnan(m[:204]))


def test_book_to_market_and_growth():
    pb = np.array([[2.0, 0.0, np.nan, -4.0]])
    b = _book_to_market(pb)
    assert np.isclose(b[0, 0], 0.5) and np.isnan(b[0, 1]) and np.isnan(b[0, 2]) and np.isclose(b[0, 3], -0.25)
    rev = np.full((300, 4), 100.0)
    rev[252:, 0] = 150.0
    rev[10, 1] = 0.0           # earlier value <= 0 -> missing at row 262
    rev[20, 2] = np.nan        # earlier missing -> missing at row 272
    g = _revenue_growth(rev)
    assert np.all(np.isnan(g[:252]))
    assert np.isclose(g[260, 0], 0.5)
    assert np.isnan(g[262, 1]) and np.isclose(g[263, 1], 0.0)
    assert np.isnan(g[272, 2])


def test_winsorize_rows_only_on_given_rows_and_members():
    x = np.tile(np.arange(101.0), (3, 1))
    members = np.ones_like(x, bool)
    members[1, 100] = False
    w = _winsorize_rows(x, members, [1])
    assert np.all(np.isnan(w[0])) and np.all(np.isnan(w[2]))
    vals = np.arange(100.0)
    lo, hi = np.percentile(vals, [1, 99])
    assert np.isclose(w[1, 0], lo) and np.isclose(w[1, 99], hi)
    assert np.isnan(w[1, 100])
    assert np.isclose(w[1, 50], 50.0)
