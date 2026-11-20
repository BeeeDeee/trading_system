import math

import numpy as np
import pytest

from lab.framework.api import total_return_index
from lab.framework.data import DataView
from strategy import PARAMS, RISK, DEFENSIVE, _features, _book, _events_from_score, model_score, target_weights

NAMES = tuple(RISK + DEFENSIVE)


def make_view(T=1000, seed=0, listed=None, tradable=None):
    rng = np.random.default_rng(seed)
    N = len(NAMES)
    dates = np.arange(10957, 10957 + T).astype("datetime64[D]")
    vol = np.exp(rng.normal(np.log(0.008), 0.4, N))
    ret_co = rng.normal(0, 0.3, (T, N)) * vol
    ret_oc = rng.normal(0, 1.0, (T, N)) * vol * (1.0 + 0.5 * np.sin(np.arange(T) / 90.0))[:, None]
    if listed is None:
        listed = np.ones((T, N), dtype=bool)
    ret_co = np.where(listed, ret_co, 0.0)
    ret_oc = np.where(listed, ret_oc, 0.0)
    close = np.where(listed, 100 * np.cumprod(1 + ret_co + ret_oc, axis=0), np.nan)
    dv = np.where(listed, np.exp(rng.normal(16, 0.5, (T, N)) + rng.normal(0, 1, N)), np.nan)
    if tradable is None:
        tradable = listed.copy()
    return DataView(dates=dates, instruments=NAMES, asset_class=tuple("us_etf" for _ in NAMES),
                    ret_co=ret_co, ret_oc=ret_oc, tradable=tradable, listed=listed,
                    delisting=np.zeros((T, N), dtype=bool), close=close, dollar_volume=dv,
                    universe=None, extras={}, series={}, cash_ret=np.zeros(T))


# ----------------------------------------------------------------------------- features
def test_features_match_hand_formulas():
    d = make_view(T=900)
    over, trend, recent, surge, volreg = _features(d)
    L = np.log(np.asarray(total_return_index(d), dtype=float))
    dv = np.asarray(d.dollar_volume, dtype=float)
    for t in (800, 899):
        for j in (0, 20, 40):
            r = np.diff(L[:, j])                 # r[i-1] = return of row i
            r252 = r[t - 252:t]                  # returns of rows t-251..t
            r756 = r[t - 756:t]
            r63 = r[t - 63:t]
            s252 = np.std(r252, ddof=1)
            s756 = np.std(r756, ddof=1)
            s63 = np.std(r63, ddof=1)
            assert trend[t, j] == pytest.approx((L[t - 21, j] - L[t - 252, j]) / (s252 * math.sqrt(231)), rel=1e-6)
            assert over[t, j] == pytest.approx((L[t, j] - L[t - 755:t + 1, j].mean()) / (s756 * math.sqrt(252)), rel=1e-6)
            assert recent[t, j] == pytest.approx((L[t, j] - L[t - 21, j]) / (s252 * math.sqrt(21)), rel=1e-6)
            assert surge[t, j] == pytest.approx(math.log(dv[t - 62:t + 1, j].mean() / dv[t - 251:t + 1, j].mean()), rel=1e-6)
            assert volreg[t, j] == pytest.approx(math.log(s63 / s756), rel=1e-6)


def test_features_warmup_rows():
    d = make_view(T=800)
    feats = _features(d)
    # 756-day windows need 681 non-missing days (90 %); returns start at row 1 -> first row 681
    for f in feats:
        assert not np.isfinite(f[:681]).any()
        assert np.isfinite(f[681:]).all()


def test_late_listing_and_gaps():
    T = 2400
    listed = np.ones((T, len(NAMES)), dtype=bool)
    late = NAMES.index("EWZ")
    listed[:1500, late] = False                 # lists at row 1500
    small = NAMES.index("TLT")
    listed[1000:1004, small] = False            # 4-day hole: every window still >= 90 %
    big = NAMES.index("GLD")
    listed[1000:1081, big] = False              # 81-day hole: 756-day window below 90 %
    d = make_view(T=T, listed=listed)
    over = _features(d)[0]
    # late lister: 681 returns from row 1501 on -> first scored row 2181
    assert not np.isfinite(over[:2181, late]).any()
    assert np.isfinite(over[2181:, late]).all()
    # small hole: scored again from row 1004, except where the t-21 / t-252 endpoints of trend and
    # recent fall inside the hole (an undefined endpoint makes the feature, and so the score, missing)
    hole_end = np.zeros(T, dtype=bool)
    for lag in (0, 21, 252):
        hole_end[1000 + lag:1004 + lag] = True
    rows = np.arange(1000, T)
    assert np.isfinite(over[rows[~hole_end[rows]], small]).all()
    assert not np.isfinite(over[hole_end, small]).any()
    assert not np.isfinite(over[1000:1300, big]).any()
    assert np.isfinite(over[1081 + 757:, big]).all()
    # everything else unaffected
    assert np.isfinite(over[700:, NAMES.index("SPY")]).all()


def test_features_point_in_time():
    d = make_view(T=900, seed=3)
    full = _features(d)
    cut = 820
    dt = DataView(dates=d.dates[:cut], instruments=d.instruments, asset_class=d.asset_class,
                  ret_co=d.ret_co[:cut], ret_oc=d.ret_oc[:cut], tradable=d.tradable[:cut],
                  listed=d.listed[:cut], delisting=d.delisting[:cut], close=d.close[:cut],
                  dollar_volume=d.dollar_volume[:cut], universe=None, extras={}, series={},
                  cash_ret=d.cash_ret[:cut])
    part = _features(dt)
    for a, b in zip(full, part):
        np.testing.assert_allclose(a[:cut], b, rtol=1e-9, equal_nan=True)


# ----------------------------------------------------------------------------- book
def ramp_score(T, N):
    """Score increasing with column index, constant over time."""
    return np.tile(np.arange(N, dtype=float), (T, 1))


def cols(names):
    return np.array([NAMES.index(n) for n in names])


def test_book_decision_rows_and_picks():
    T = 200
    d = make_view(T=T)
    N = len(NAMES)
    w = _book(ramp_score(T, N), d, PARAMS)
    dec = np.arange(0, T, 20)
    assert np.isfinite(w[dec]).all()
    other = np.setdiff1d(np.arange(T), dec)
    assert np.isnan(w[other]).all()
    row = w[40]
    # RISK: round(0.2 * 32) = 6 highest columns of the sleeve at 0.6 / 6; DEFENSIVE: round(0.2 * 15) = 3 at 0.4 / 3
    risk = cols(RISK)
    dfn = cols(DEFENSIVE)
    assert np.allclose(np.sort(row[risk])[::-1][:6], 0.1)
    assert np.allclose(row[risk[-6:]], 0.1) and np.allclose(row[risk[:-6]], 0.0)
    assert np.allclose(row[dfn[-3:]], 0.4 / 3) and np.allclose(row[dfn[:-3]], 0.0)
    assert row.sum() == pytest.approx(1.0)
    assert (w[dec] >= 0).all() and (w[dec].sum(axis=1) <= 1 + 1e-12).all()


@pytest.mark.parametrize("top,k_risk,k_def", [(0.15, 5, 2), (0.2, 6, 3), (0.3, 10, 5), (0.01, 2, 2)])
def test_book_pick_counts(top, k_risk, k_def):
    T = 60
    d = make_view(T=T)
    w = _book(ramp_score(T, len(NAMES)), d, {"n_estimators": 100, "top": top})
    row = w[40]
    assert (row[cols(RISK)] > 0).sum() == k_risk
    assert (row[cols(DEFENSIVE)] > 0).sum() == k_def
    assert row[cols(RISK)].sum() == pytest.approx(0.6)
    assert row[cols(DEFENSIVE)].sum() == pytest.approx(0.4)


def test_book_tranche_average():
    T = 120
    d = make_view(T=T)
    N = len(NAMES)
    s = np.full((T, N), np.nan)
    a = ramp_score(T, N)
    s[0] = a[0]                                  # tranche 0: high columns
    s[20] = -a[0]                                # tranche 1: low columns
    s[40] = a[0]
    s[60] = a[0]
    w = _book(s, d, PARAMS)
    t0 = np.zeros(N)
    t0[cols(RISK)[-6:]] = 0.1
    t0[cols(DEFENSIVE)[-3:]] = 0.4 / 3
    t1 = np.zeros(N)
    t1[cols(RISK)[:6]] = 0.1
    t1[cols(DEFENSIVE)[:3]] = 0.4 / 3
    np.testing.assert_allclose(w[0], t0)
    np.testing.assert_allclose(w[20], (t0 + t1) / 2)
    np.testing.assert_allclose(w[40], (t0 + t1 + t0) / 3)
    np.testing.assert_allclose(w[60], (t1 + t0 + t0) / 3)
    # row 80 has no score: its tranche is the fallback (EW over tradable ETFs with >= 63 sessions)
    ew = np.zeros(N)
    ew[cols(RISK)] = 0.6 / 32
    ew[cols(DEFENSIVE)] = 0.4 / 15
    np.testing.assert_allclose(w[80], (t0 + t0 + ew) / 3)
    assert w[80].sum() == pytest.approx(1.0)


def test_book_fallback_and_cash():
    T = 160
    listed = np.ones((T, len(NAMES)), dtype=bool)
    listed[:100, cols(DEFENSIVE)] = False        # no defensive ETF exists before row 100
    d = make_view(T=T, listed=listed)
    w = _book(np.full((T, len(NAMES)), np.nan), d, PARAMS)
    # rows 0..60: fewer than 63 priced sessions -> cash
    assert np.allclose(w[[0, 20, 40, 60]], 0.0)
    # row 80: risk sleeve EW over 32 (81 sessions of history), defensive sleeve not listed -> cash
    ew = np.zeros(len(NAMES))
    ew[cols(RISK)] = 0.6 / 32
    np.testing.assert_allclose(w[80], ew / 3)
    np.testing.assert_allclose(w[120], (ew + ew + ew) / 3)   # defensive still < 63 sessions at 120
    # 4 scored defensive members (< 5) -> fallback for that sleeve, model picks for RISK
    s = np.full((T, len(NAMES)), np.nan)
    s[:, cols(RISK)] = np.arange(32, dtype=float)
    s[:, cols(DEFENSIVE)[:4]] = 1.0
    d2 = make_view(T=T)
    w2 = _book(s, d2, PARAMS)
    np.testing.assert_allclose(w2[60, cols(DEFENSIVE)], 0.0)            # all three tranches still < 63 sessions
    np.testing.assert_allclose(w2[100, cols(DEFENSIVE)], 0.4 / 15 * 2 / 3)  # tranches 80, 100 in fallback
    row = w2[140]
    np.testing.assert_allclose(row[cols(DEFENSIVE)], 0.4 / 15)
    assert (row[cols(RISK)] > 0).sum() == 6


def test_book_untradable_pick_redistributed():
    T = 60
    N = len(NAMES)
    tradable = np.ones((T, N), dtype=bool)
    best = cols(RISK)[-1]
    tradable[40, best] = False
    d = make_view(T=T, tradable=tradable)
    w = _book(ramp_score(T, N), d, PARAMS)
    full = np.zeros(N)
    full[cols(RISK)[-6:]] = 0.1
    full[cols(DEFENSIVE)[-3:]] = 0.4 / 3
    part = np.zeros(N)
    part[cols(RISK)[-6:-1]] = 0.12
    part[cols(DEFENSIVE)[-3:]] = 0.4 / 3
    np.testing.assert_allclose(w[40], (full + full + part) / 3)
    assert w[40].sum() == pytest.approx(1.0)


# ----------------------------------------------------------------------------- diagnostic
def test_events_new_entrants_only():
    T = 80
    N = len(NAMES)
    d = make_view(T=T)
    s = ramp_score(T, N)
    s[40:] = -s[40:]                             # ranking flips at decision row 40
    mask, side = _events_from_score(s, d, PARAMS)
    risk, dfn = cols(RISK), cols(DEFENSIVE)
    # row 0: top 6 / 3 enter (+1), bottom 6 / 3 enter (-1)
    assert mask[0].sum() == 18
    assert (side[0, risk[-6:]] == 1).all() and (side[0, risk[:6]] == -1).all()
    assert (side[0, dfn[-3:]] == 1).all() and (side[0, dfn[:3]] == -1).all()
    assert mask[20].sum() == 0                   # unchanged buckets: no events
    assert mask[40].sum() == 18                  # flip: all buckets are new
    assert (side[40, risk[:6]] == 1).all() and (side[40, risk[-6:]] == -1).all()
    assert mask[60].sum() == 0
    other = np.setdiff1d(np.arange(T), np.arange(0, T, 20))
    assert not mask[other].any()


def test_events_require_tradable_and_scores():
    T = 40
    N = len(NAMES)
    tradable = np.ones((T, N), dtype=bool)
    tradable[0, cols(RISK)[-1]] = False
    d = make_view(T=T, tradable=tradable)
    s = ramp_score(T, N)
    s[:, cols(DEFENSIVE)[:11]] = np.nan          # only 4 scored defensive members: no buckets there
    mask, side = _events_from_score(s, d, PARAMS)
    assert mask[0].sum() == 11
    assert not mask[0, cols(DEFENSIVE)].any()
    assert not mask[0, cols(RISK)[-1]]


# ----------------------------------------------------------------------------- end to end (model)
def test_end_to_end_model_path():
    T = 1500
    d = make_view(T=T, seed=1)
    s50 = model_score(d, {"n_estimators": 50, "top": 0.2})
    s100 = model_score(d, PARAMS)
    first = np.where(np.isfinite(s100).any(axis=1))[0]
    assert first.size > 0
    assert first[0] >= 681                       # no score before the features exist
    t = T - 1
    assert np.nanstd(s100[t]) > 0                # the model separates ETFs
    assert not np.allclose(s50[first[0]:], s100[first[0]:], equal_nan=True)
    w = target_weights(d, PARAMS)
    dec = np.arange(0, T, 20)
    assert w.shape == (T, len(NAMES))
    assert (w[dec] >= -1e-15).all() and (w[dec].sum(axis=1) <= 1 + 1e-9).all()
    assert np.isnan(np.delete(w, dec, axis=0)).all()
    w2 = target_weights(d, PARAMS)
    np.testing.assert_array_equal(w, w2)
