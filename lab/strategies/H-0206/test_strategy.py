import dataclasses

import numpy as np
import pytest

from strategy import (PARAMS, book_from_scores, candidate_sets, day_returns, model_scores, target_weights,
                      trailing_window_stat, zscore_feature)


def _rg(rs, gs):
    r = np.array([rs], dtype=float)
    g = np.array([gs], dtype=float)
    valid = np.isfinite(r) & np.isfinite(g)
    return r, g, valid


def test_candidate_sets_thresholds():
    # cols: -10% exact close, -15% exact, -15.1%, gap -10% w/ close -5%, -9%, -8.5%, -8.4%, -9% w/ gap -10%, -12% w/ NaN
    rs = [-0.10, -0.15, -0.151, -0.05, -0.09, -0.085, -0.084, -0.09, np.nan]
    gs = [-0.02, -0.03, -0.05, -0.10, -0.01, -0.02, -0.02, -0.10, -0.12]
    r, g, valid = _rg(rs, gs)
    uni = np.ones_like(valid)
    R, N = candidate_sets(r, g, valid, uni, 0.015)
    assert R[0].tolist() == [True, True, False, True, False, False, False, True, False]
    assert N[0].tolist() == [False, False, False, False, True, True, False, False, False]
    assert not np.any(R & N)
    # wider band picks up -8.4 %
    R2, N2 = candidate_sets(r, g, valid, uni, 0.02)
    assert N2[0, 6] and not N[0, 6]
    # non-members are never candidates
    uni2 = uni.copy()
    uni2[0, 0] = False
    uni2[0, 4] = False
    R3, N3 = candidate_sets(r, g, valid, uni2, 0.015)
    assert not R3[0, 0] and not N3[0, 4]


def test_trailing_window_stat_uses_only_past_full_windows():
    T, N, n = 8, 2, 3
    x = np.arange(T * N, dtype=float).reshape(T, N)
    ok = np.ones((T, N), dtype=bool)
    ok[4, 1] = False
    mask = np.ones((T, N), dtype=bool)
    med = trailing_window_stat(x, ok, mask, n, "median")
    assert np.all(np.isnan(med[:n]))
    # row 3 col 0: rows 0..2 -> 0, 2, 4 -> median 2
    assert med[3, 0] == 2.0
    assert med[7, 0] == x[5, 0]
    # col 1 row 5..7 windows contain row 4 (not ok) -> NaN; row 8 would be fine
    assert np.isnan(med[5, 1]) and np.isnan(med[7, 1])
    assert med[4, 1] == x[2, 1]
    sd = trailing_window_stat(x, ok, mask, n, "std")
    assert sd[3, 0] == pytest.approx(np.std([0.0, 2.0, 4.0], ddof=1))
    # masked cells are not computed
    mask2 = mask.copy()
    mask2[6, 0] = False
    assert np.isnan(trailing_window_stat(x, ok, mask2, n, "median")[6, 0])
    # a view shorter than the window
    assert np.all(np.isnan(trailing_window_stat(x[:2], ok[:2], mask[:2], n, "median")))


def test_book_cohorts_overlap_and_close():
    T, N = 6, 6
    score = np.full((T, N), np.nan)
    cand = np.zeros((T, N), dtype=bool)
    uni = np.ones((T, N), dtype=bool)
    # row 1: 5 scored candidates (odd) -> long 2 highest, short 2 lowest, median gets nothing
    cand[1, :5] = True
    score[1, :5] = [0.5, -1.0, 2.0, 0.1, -0.3]
    # row 2: 2 candidates, one unscored -> k = 1 -> no cohort
    cand[2, 0:2] = True
    score[2, 0] = 1.0
    # row 3: a scored non-candidate is ignored; 2 candidates
    score[3, 5] = 9.0
    cand[3, 1:3] = True
    score[3, 1:3] = [0.2, -0.2]
    w = book_from_scores(score, cand, uni)
    assert np.all(w[0] == 0)
    exp1 = np.array([0.15, -0.15, 0.15, 0.0, -0.15, 0.0])
    np.testing.assert_allclose(w[1], exp1)
    np.testing.assert_allclose(w[2], exp1)            # cohort of row 1 still held (2 sessions)
    exp3 = np.array([0.0, 0.15, -0.15, 0, 0, 0])
    np.testing.assert_allclose(w[3], exp3)            # row-1 cohort closed at the open after row 2
    np.testing.assert_allclose(w[4], exp3)
    assert np.all(w[5] == 0)
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)


def test_book_overlap_adds_scales_and_drops_nonmembers():
    T, N = 3, 12
    score = np.full((T, N), np.nan)
    cand = np.zeros((T, N), dtype=bool)
    uni = np.ones((T, N), dtype=bool)
    cand[0, :4] = True
    score[0, :4] = [1.0, 2.0, -1.0, -2.0]   # long 0,1 short 2,3
    cand[1, :4] = True
    score[1, :4] = [3.0, -3.0, 0.0, 1.0]    # long 0,3 short 1,2
    w = book_from_scores(score, cand, uni)
    np.testing.assert_allclose(w[1, :4], [0.30, 0.0, -0.30, 0.0])
    # many candidates -> gross scaled to 1
    cand2 = np.ones((T, N), dtype=bool)
    score2 = np.tile(np.arange(N, dtype=float), (T, 1))
    w2 = book_from_scores(score2, cand2, uni)
    g = np.abs(w2).sum(axis=1)
    assert g[0] == pytest.approx(1.0)       # 12 x 0.15 = 1.8 -> scaled
    assert g[1] == pytest.approx(1.0)
    assert np.allclose(w2.sum(axis=1), 0.0)
    # a stock leaving the universe is closed
    uni3 = uni.copy()
    uni3[1, 0] = False
    w3 = book_from_scores(score, cand, uni3)
    assert w3[1, 0] == 0.0 and w3[0, 0] == pytest.approx(0.15)


def test_zscore_constant_feature_gets_zero():
    uni = np.ones((2, 4), dtype=bool)
    f = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, np.nan]])
    z = zscore_feature(f, uni)
    assert np.all(z[0] == 0.0)
    assert np.isnan(z[1, 3])
    assert z[1, 1] > 0 > z[1, 0]


def _shocked_synthetic():
    """Synthetic market with a universe, a sparse 0/1 earnings flag and injected R / N drops (the plain synthetic
    returns never fall 10 % in a day, so the book would be empty)."""
    from lab.framework.data import synthetic
    data = synthetic()
    T, N = data.ret_co.shape
    rng = np.random.default_rng(7)
    co = np.array(data.ret_co, dtype=float)
    oc = np.array(data.ret_oc, dtype=float)
    rows = np.repeat(rng.integers(100, T, size=200), 3)              # clustered: several candidates per date
    cols = rng.integers(0, N, size=600)
    kind = rng.integers(0, 4, size=600)
    for t, j, k in zip(rows, cols, kind):
        if k == 0:      # restricted by the close: -12 %
            co[t, j], oc[t, j] = -0.03, 0.88 / 0.97 - 1.0
        elif k == 1:    # restricted by the open gap: -11 % open, -6 % close
            co[t, j], oc[t, j] = -0.11, 0.94 / 0.89 - 1.0
        elif k == 2:    # near miss: -9.2 %
            co[t, j], oc[t, j] = -0.02, 0.908 / 0.98 - 1.0
        else:           # near miss only with near_band 0.02: -8.3 %
            co[t, j], oc[t, j] = -0.02, 0.917 / 0.98 - 1.0
    ex = dict(data.extras)
    ex["ea_reaction_day"] = (rng.random((T, N)) < 0.02).astype(np.float32)
    uni = np.asarray(data.listed, dtype=bool).copy()
    uni[:, : N // 4] &= np.arange(T)[:, None] >= T // 3          # some late members
    return dataclasses.replace(data, ret_co=co, ret_oc=oc, extras=ex, universe=uni)


def _truncate(data, c):
    T = len(data.dates)
    upd = {}
    for name, v in dataclasses.asdict(data).items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T:
            upd[name] = v[:c]
        elif isinstance(v, dict):
            upd[name] ={k: (a[:c] if isinstance(a, np.ndarray) and a.shape[0] == T else a) for k, a in v.items()}
    return dataclasses.replace(data, **upd)


def test_end_to_end_synthetic():
    data = _shocked_synthetic()
    w = target_weights(data, PARAMS)
    assert w.shape == (len(data.dates), len(data.instruments))
    assert np.all(np.isfinite(w))
    gross = np.abs(w).sum(axis=1)
    assert np.all(gross <= 1.0 + 1e-9)
    assert np.count_nonzero(gross) > 50
    np.testing.assert_allclose(w.sum(axis=1), 0.0, atol=1e-9)    # cohorts are dollar-neutral
    assert not np.any(w[~np.asarray(data.universe)])              # members only
    # positions only on candidates of row t or t-1
    r, g, valid = day_returns(data)
    R, Nn = candidate_sets(r, g, valid, np.asarray(data.universe), PARAMS["near_band"])
    c = R | Nn
    c2 = c.copy()
    c2[1:] |= c[:-1]
    assert not np.any(w[~c2])
    # determinism
    np.testing.assert_array_equal(w, target_weights(data, PARAMS))
    # point-in-time: truncation leaves earlier rows unchanged
    cut = 2 * len(data.dates) // 3
    wt = target_weights(_truncate(data, cut), PARAMS)
    np.testing.assert_allclose(wt, w[:cut], atol=1e-12)
    # each parameter matters
    w_nb = target_weights(data, dict(PARAMS, near_band=0.02))
    assert not np.array_equal(w_nb, w)
    uni = np.asarray(data.universe)
    s1 = model_scores(data, dict(PARAMS, alpha=1.0), uni)
    s2 = model_scores(data, dict(PARAMS, alpha=100.0), uni)
    assert not np.allclose(np.nan_to_num(s1), np.nan_to_num(s2))
    # warm-up: no weights before the first scores
    s = model_scores(data, PARAMS, uni)
    first = np.argmax(np.isfinite(s).any(axis=1))
    assert first > 0 and np.all(w[:first] == 0)
    assert R.sum() > 100 and Nn.sum() > 50
