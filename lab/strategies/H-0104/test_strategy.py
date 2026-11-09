import dataclasses
import math

import numpy as np
import pytest

from strategy import PARAMS, target_weights
from strategy import book, decision_rows, event_features, prior_scale, universe_relative, daily_returns
from lab.framework.data import synthetic


# ---------------------------------------------------------------- helpers on hand-made arrays

def test_daily_returns_and_universe_relative():
    ret_co = np.array([[0.01, 0.0, 0.02], [0.0, 0.0, 0.0]])
    ret_oc = np.array([[0.01, 0.10, 0.0], [0.02, -0.02, 0.05]])
    listed = np.array([[True, True, True], [True, True, False]])
    r = daily_returns(ret_co, ret_oc, listed)
    assert r[0, 0] == pytest.approx(1.01 * 1.01 - 1)
    assert np.isnan(r[1, 2])
    uni = np.array([[True, True, False], [True, True, True]])
    u = universe_relative(r, uni)
    m0 = (r[0, 0] + r[0, 1]) / 2           # non-member col 2 not in the mean
    assert u[0, 2] == pytest.approx(r[0, 2] - m0)
    m1 = (0.02 - 0.02) / 2                 # member without a return not in the mean
    assert u[1, 0] == pytest.approx(0.02 - m1)
    assert np.isnan(u[1, 2])


def test_prior_scale_window_and_min_valid():
    rng = np.random.default_rng(1)
    u = rng.normal(0, 0.02, size=(120, 2))
    u[:, 1] = np.nan
    u[70:, 1] = rng.normal(0, 0.02, size=50)
    S = prior_scale(u)
    # row 100: std of rows 37..99 (63 values), ddof 1
    assert S[100, 0] == pytest.approx(np.std(u[37:100, 0], ddof=1), rel=1e-9)
    # fewer than 40 valid values before row 40
    assert np.isnan(S[39, 0]) and np.isfinite(S[40, 0])
    assert S[40, 0] == pytest.approx(np.std(u[0:40, 0], ddof=1), rel=1e-9)
    # column 1 has values from row 70: 40 valid values first at row 110
    assert np.isnan(S[109, 1]) and np.isfinite(S[110, 1])
    # views shorter than the window do not crash
    assert prior_scale(u[:5]).shape == (5, 2)


def test_event_features_by_hand():
    rng = np.random.default_rng(2)
    T = 400
    u = rng.normal(0, 0.01, size=(T, 1))
    ev = np.zeros((T, 1), dtype=bool)
    ev[100, 0] = True       # R0
    ev[200, 0] = True       # R1 (100 sessions after R0)
    rz, rzp, since, post, pre = event_features(u, ev)
    S = prior_scale(u)
    s1 = S[200, 0]
    assert s1 == pytest.approx(np.std(u[137:200, 0], ddof=1), rel=1e-9)
    t = 210
    assert since[t, 0] == 10
    assert rz[t, 0] == pytest.approx(u[200, 0] / s1)
    assert post[t, 0] == pytest.approx(u[201:211, 0].sum() / s1)
    assert pre[t, 0] == pytest.approx(u[179:200, 0].sum() / (s1 * math.sqrt(21)))
    assert rzp[t, 0] == pytest.approx(u[100, 0] / S[100, 0])
    # on the reaction session itself
    assert since[200, 0] == 0 and post[200, 0] == 0.0
    # before R0: nothing; between R0 and R1: R0 is the latest, no previous one
    assert np.isnan(rz[99, 0]) and np.isnan(since[99, 0])
    assert since[150, 0] == 50 and np.isnan(rzp[150, 0]) and np.isfinite(rz[150, 0])
    # R1 within the last 126 sessions: t - R1 <= 125
    assert since[325, 0] == 125 and np.isnan(rz[326, 0]) and np.isnan(since[326, 0])


def test_event_features_previous_window():
    rng = np.random.default_rng(3)
    u = rng.normal(0, 0.01, size=(400, 2))
    ev = np.zeros((400, 2), dtype=bool)
    ev[100, 0] = ev[226, 0] = True          # 126 apart: previous counts
    ev[100, 1] = ev[227, 1] = True          # 127 apart: previous missing
    _, rzp, _, _, _ = event_features(u, ev)
    assert np.isfinite(rzp[230, 0])
    assert np.isnan(rzp[230, 1])


def test_event_features_no_lookahead():
    rng = np.random.default_rng(4)
    u = rng.normal(0, 0.01, size=(300, 3))
    ev = rng.random((300, 3)) < 0.03
    full = event_features(u, ev)
    cut = event_features(u[:180], ev[:180])
    for a, b in zip(full, cut):
        np.testing.assert_array_equal(a[:180], b)


def test_decision_rows_fridays():
    d = np.arange(0, 14).astype("datetime64[D]")     # day 0 = Thursday 1970-01-01
    dec = decision_rows(d)
    assert list(np.flatnonzero(dec)) == [1, 8]


# ---------------------------------------------------------------- book

def test_book_top_fraction_fallback_and_universe_exit():
    T, N = 6, 10
    dec = np.array([True, False, False, True, False, True])
    uni = np.ones((T, N), dtype=bool)
    uni[4, 9] = False                                   # top stock leaves between decisions
    score = np.tile(np.arange(N, dtype=float), (T, 1))
    score[5, :] = np.nan                                 # too few scored -> equal weight members
    score[5, :3] = 1.0
    w = book(score, uni, dec, top=0.2, min_scored=5)
    np.testing.assert_allclose(w[0], [0] * 8 + [0.5, 0.5])
    assert np.all(np.isnan(w[1])) and np.all(np.isnan(w[2]))
    np.testing.assert_allclose(w[3], [0] * 8 + [0.5, 0.5])
    assert w[4, 9] == 0.0 and np.isnan(w[4, :9]).all()   # sold on the exit row, others kept
    np.testing.assert_allclose(w[5], np.full(N, 0.1))
    w3 = book(score, uni, dec, top=0.5, min_scored=5)
    assert (w3[0] > 0).sum() == 5
    for row in (w, w3):
        assert np.nanmax(np.nansum(np.abs(row), axis=1)) <= 1.0 + 1e-12


def test_book_exit_emitted_once():
    T, N = 6, 6
    dec = np.array([True, False, False, False, False, True])
    uni = np.ones((T, N), dtype=bool)
    uni[2:4, 5] = False
    score = np.tile(np.arange(N, dtype=float), (T, 1))
    w = book(score, uni, dec, top=0.2, min_scored=5)
    assert w[2, 5] == 0.0 and np.isnan(w[3, 5]) and np.isnan(w[4, 5])


# ---------------------------------------------------------------- end to end on a synthetic market

def _market(T=1300, N=70, seed=7):
    base = synthetic()
    rng = np.random.default_rng(seed)
    dates = base.dates[:T]
    ret_co = rng.normal(0, 0.005, size=(T, N))
    ret_oc = rng.normal(0, 0.015, size=(T, N))
    listed = np.ones((T, N), dtype=bool)
    listed[:300, 0] = False                             # late listing
    ret_co[~listed] = 0.0
    ret_oc[~listed] = 0.0
    uni = listed.copy()
    uni[600:700, 1] = False
    flag = (rng.random((T, N)) < 1 / 63).astype(float)
    flag[:200, 2] = np.nan                              # no 8-K data yet
    close = np.where(listed, 50.0, np.nan)
    return dataclasses.replace(
        base, dates=dates, instruments=tuple("X%02d" % j for j in range(N)),
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=ret_co, ret_oc=ret_oc, tradable=listed.copy(), listed=listed,
        delisting=np.zeros((T, N), dtype=bool), close=close, dollar_volume=np.where(listed, 1e8, np.nan),
        universe=uni, extras={"ea_reaction_day": flag}, series={})


def test_end_to_end_book():
    data = _market()
    w = target_weights(data, PARAMS)
    T, N = w.shape
    assert w.shape == (len(data.dates), len(data.instruments))
    rows = ~np.all(np.isnan(w), axis=1)
    gross = np.nansum(np.abs(w), axis=1)
    assert gross.max() <= 1.0 + 1e-9 and np.nanmin(w) >= 0.0
    # never weight on non-members
    assert not np.any((np.nan_to_num(w) > 0) & ~data.universe)
    dec = decision_rows(data.dates)
    assert np.all(rows[dec])
    # warm-up: first decision holds all members equal weight
    first = np.flatnonzero(dec)[0]
    np.testing.assert_allclose(w[first][data.universe[first]], 1.0 / data.universe[first].sum())
    # later decisions: the model book holds the top fraction
    last = np.flatnonzero(dec)[-1]
    held = (w[last] > 0).sum()
    assert held < data.universe[last].sum()
    w_top = target_weights(data, {**PARAMS, "top": 0.1})
    assert (w_top[last] > 0).sum() < held
    w_nl = target_weights(data, {**PARAMS, "num_leaves": 31})
    assert w_nl.shape == w.shape
    assert not np.array_equal(w_nl, w, equal_nan=True)


def test_diagnostic_gbm_only_entries():
    # strategy.py carries a verbatim copy of diagnostic.py's event helpers (tests may import only strategy)
    import strategy as diagnostic
    T, N = 5, 10
    dec = np.array([True, False, True, False, True])
    uni = np.ones((T, N), dtype=bool)
    tradable = np.ones((T, N), dtype=bool)
    gbm = np.full((T, N), np.nan)
    ridge = np.full((T, N), np.nan)
    # row 0: no model yet; row 2: first ranking (no previous ranking -> no events); row 4: entries
    gbm[2] = np.arange(N, dtype=float)                   # top 2: cols 8, 9
    ridge[2] = np.arange(N, dtype=float)
    gbm[4] = np.arange(N, dtype=float)[::-1]             # top 2: cols 0, 1 (new entries)
    ridge[4] = np.arange(N, dtype=float)
    ridge[4, 0] = 100.0                                  # col 0 also in the ridge top -> no event
    tradable[4, 5] = False
    # the real minimum of 50 scored members: no events on 10 names
    assert diagnostic.gbm_only_entries(gbm, ridge, uni, tradable, dec, 0.2).sum() == 0
    mask = diagnostic.gbm_only_entries(gbm, ridge, uni, tradable, dec, 0.2, min_scored=5)
    assert np.argwhere(mask).tolist() == [[4, 1]]          # first ranking date (row 2) has no events
    tradable[4, 1] = False
    assert diagnostic.gbm_only_entries(gbm, ridge, uni, tradable, dec, 0.2, min_scored=5).sum() == 0
    uni[4, 1] = False
    g_top, ok = diagnostic.top_sets(gbm, np.ones((T, N), dtype=bool), 0.2, min_scored=5)
    assert ok.tolist() == [False, False, True, False, True]
    assert np.flatnonzero(g_top[4]).tolist() == [0, 1]


def test_missing_flag_raises():
    data = _market(T=300)
    data = dataclasses.replace(data, extras={})
    with pytest.raises(KeyError):
        target_weights(data, PARAMS)
