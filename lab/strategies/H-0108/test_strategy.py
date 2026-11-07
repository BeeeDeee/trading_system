import numpy as np
import pytest

from strategy import (
    PARAMS, _beta, _bottom_fraction, _current_mcap, _rsum, book, cohort_legs, compute_inputs,
    model_scores, target_weights,
)


class FakeData:
    """Minimal stand-in exposing the attributes compute_inputs reads."""

    def __init__(self, T, N, seed=0):
        rng = np.random.default_rng(seed)
        self.instruments = tuple("E%d" % i for i in range(N))
        self.m = rng.normal(0.0, 0.01, T)
        self.beta_true = np.linspace(0.5, 1.5, N)
        idio = rng.normal(0.0, 0.01, (T, N))
        r = self.beta_true[None, :] * self.m[:, None] + idio
        self.ret_co = r * rng.uniform(0.0, 0.6, (T, N))
        self.ret_oc = (1.0 + r) / (1.0 + self.ret_co) - 1.0
        self.listed = np.ones((T, N), dtype=bool)
        self.close = np.full((T, N), 50.0)
        self.dollar_volume = 1e7 * np.exp(rng.normal(0.0, 0.3, (T, N)))
        self.universe = np.ones((T, N), dtype=bool)
        self.extras = {
            "sf1_art_netinc": np.tile(np.linspace(-5.0, 20.0, N, dtype=np.float32), (T, 1)),
            "sf1_art_equity": np.full((T, N), 100.0, dtype=np.float32),
            "sf1_art_marketcap": np.full((T, N), 1000.0, dtype=np.float32),
            "ea_reaction_day": np.zeros((T, N), dtype=np.float32),
        }

    def shock(self, t, j, size):
        """A total-return drop of `size` at row t on stock j, all intraday."""
        r = (1 + self.ret_co[t, j]) * (1 + self.ret_oc[t, j]) - 1 + size
        self.ret_oc[t, j] = (1 + r) / (1 + self.ret_co[t, j]) - 1


# ------------------------------------------------------------------ helpers
def test_rsum_trailing_and_partial():
    x = np.arange(1.0, 7.0)[:, None]
    out = _rsum(x, 3)[:, 0]
    assert np.allclose(out, [1, 3, 6, 9, 12, 15])
    # window longer than the view: plain cumulative sums
    assert np.allclose(_rsum(x, 10)[:, 0], np.cumsum(x[:, 0]))


def test_beta_exact_and_min_obs():
    rng = np.random.default_rng(1)
    T = 300
    m = rng.normal(0, 0.01, T)
    r = np.stack([2.0 * m, 0.5 * m], axis=1)
    valid = np.ones((T, 2), dtype=bool)
    valid[:, 1] = np.arange(T) >= 150            # stock 1 has only 150 valid sessions at the end
    r = np.where(valid, r, 0.0)
    b = _beta(r, m, valid)
    assert np.isnan(b[198, 0]) and np.isfinite(b[199, 0])     # 200 sessions needed
    assert np.allclose(b[199:, 0], 2.0)
    assert np.all(np.isnan(b[:, 1][:349]))                     # < 200 valid until row 349
    assert np.allclose(b[349:, 1], 0.5)


def test_current_mcap_compounds_from_first_usable_session():
    mc = np.array([np.nan, 100, 100, 100, 200, 200], dtype=np.float64)[:, None]
    r = np.array([0.3, 0.1, 0.1, 0.0, 0.5, 0.0])[:, None]
    cap = _current_mcap(mc, r)[:, 0]
    assert np.isnan(cap[0])
    assert np.allclose(cap[1:], [110.0, 121.0, 121.0, 300.0, 300.0])


def test_bottom_fraction_counts_valid_only():
    x = np.arange(20.0)[None, :]
    valid = np.ones((1, 20), dtype=bool)
    valid[0, 0] = False                       # the lowest value is not valid
    sel = _bottom_fraction(x, valid, 0.1)     # floor(0.1 * 19) = 1 name
    assert sel[0].sum() == 1 and sel[0, 1]
    sel = _bottom_fraction(np.arange(20.0)[None, :], np.ones((1, 20), bool), 0.1)
    assert list(np.flatnonzero(sel[0])) == [0, 1]


def test_cohort_legs_thirds_and_minimum():
    score = np.full((3, 10), np.nan)
    score[0, :7] = np.arange(7.0)             # 7 scored -> 2 per side
    score[1, :5] = np.arange(5.0)             # 5 scored -> 1 per side -> no cohort
    score[2, :9] = np.arange(9.0)[::-1]       # 9 scored -> 3 per side
    lo, sh = cohort_legs(score)
    assert list(np.flatnonzero(lo[0])) == [5, 6] and list(np.flatnonzero(sh[0])) == [0, 1]
    assert not lo[1].any() and not sh[1].any()
    assert list(np.flatnonzero(lo[2])) == [0, 1, 2] and list(np.flatnonzero(sh[2])) == [6, 7, 8]


def test_book_hold_five_sessions_and_universe_exit():
    T, N = 12, 4
    lo = np.zeros((T, N), bool)
    sh = np.zeros((T, N), bool)
    lo[2, 0] = lo[2, 1] = True
    sh[2, 2] = sh[2, 3] = True
    uni = np.ones((T, N), bool)
    uni[5, 1] = False                          # stock 1 leaves the universe on row 5
    w = book(lo, sh, uni)
    assert np.allclose(w[:2], 0.0)
    for t in range(2, 7):                      # rows 2..6: entered at open 3, exit at open 8
        assert np.isclose(w[t, 0], 0.05) and np.isclose(w[t, 2], -0.05) and np.isclose(w[t, 3], -0.05)
    assert np.isclose(w[4, 1], 0.05) and w[5, 1] == 0.0
    assert np.allclose(w[7:], 0.0)


def test_book_gross_at_most_one_with_five_cohorts():
    T, N = 20, 30
    rng = np.random.default_rng(3)
    lo = np.zeros((T, N), bool)
    sh = np.zeros((T, N), bool)
    for t in range(T):
        p = rng.permutation(N)
        lo[t, p[:7]] = True
        sh[t, p[7:14]] = True
    w = book(lo, sh, np.ones((T, N), bool))
    gross = np.abs(w).sum(axis=1)
    assert np.all(gross <= 1.0 + 1e-9)
    assert np.isclose(gross[10], 1.0) or gross[10] < 1.0      # netting only lowers gross
    assert np.allclose(w.sum(axis=1), 0.0)


# ------------------------------------------------------------------ candidates
def _fake_with_drops():
    d = FakeData(320, 40, seed=5)
    t = 300
    for j in (3, 7, 11, 15):
        d.shock(t - 1, j, -0.08)
    return d, t


def test_candidates_are_the_residual_tail():
    d, t = _fake_with_drops()
    feats, cand = compute_inputs(d, {"tail_q": 0.1, "n_estimators": 100})
    assert set(np.flatnonzero(cand[t])) == {3, 7, 11, 15}       # floor(0.1 * 40) = 4
    assert not cand[:199].any()                                   # beta needs 200 sessions
    res5 = feats[0]
    assert np.all(res5[t, [3, 7, 11, 15]] < 0.0)
    # the residual removes the market: a stock moving exactly beta * m has ~ zero residual
    assert np.nanmax(np.abs(res5[250:, :])) < 0.2
    assert len(feats) == 7


def test_tail_q_changes_candidates():
    d, t = _fake_with_drops()
    _, c1 = compute_inputs(d, {"tail_q": 0.05, "n_estimators": 100})
    _, c2 = compute_inputs(d, {"tail_q": 0.1, "n_estimators": 100})
    assert c1[t].sum() == 2 and c2[t].sum() == 4


def test_exclusions_earnings_sf1_roe_universe():
    d, t = _fake_with_drops()
    d.extras["ea_reaction_day"][t - 4, 3] = 1.0           # earnings session inside t-4..t
    d.extras["ea_reaction_day"][t - 5, 7] = 1.0           # outside the window: still a candidate
    d.extras["sf1_art_equity"][:, 11] = -5.0              # negative equity -> ROE NaN -> excluded
    d.extras["sf1_art_netinc"][t, 15] = np.nan            # no current sf1 row
    d.extras["ea_reaction_day"][:, 20] = np.nan           # NaN flag is not an earnings session
    _, cand = compute_inputs(d, {"tail_q": 0.1, "n_estimators": 100})
    assert set(np.flatnonzero(cand[t])) == {7}
    d.universe[t, 7] = False
    _, cand = compute_inputs(d, {"tail_q": 0.1, "n_estimators": 100})
    assert not cand[t, 7]


def test_features_values():
    d, t = _fake_with_drops()
    d.dollar_volume[:, 3] = 1e7
    d.dollar_volume[t - 4:t + 1, 3] = 3e7                 # 3x volume in t-4..t
    d.dollar_volume[:, 5] = 2e7
    d.dollar_volume[t - 70:t - 4, 5] = np.nan             # no base-window volume -> NaN
    feats, _ = compute_inputs(d, {"tail_q": 0.1, "n_estimators": 100})
    res5, share, vshock, ey, bm, roe, trend = feats
    assert np.isclose(vshock[t, 3], 3.0) and np.isnan(vshock[t, 5])
    # row t+1: recent t-3..t+1 = four 3e7 and one 1e7; base t-66..t-4 holds one 3e7 (row t-4)
    assert np.isclose(vshock[t + 1, 3], ((4 * 3e7 + 1e7) / 5) / ((62 * 1e7 + 3e7) / 63))
    assert np.all(np.isnan(vshock[:67]))
    ni3 = float(d.extras["sf1_art_netinc"][t, 3])
    assert np.isclose(roe[t, 3], ni3 / 100.0)
    assert np.all((share[np.isfinite(share)] >= -1) & (share[np.isfinite(share)] <= 2))
    assert np.all(np.isnan(trend[:251])) and np.isfinite(trend[t, 3])
    # E/P and B/M use the market cap compounded since the (single) sf1 row
    r = (1 + d.ret_co[:, 3]) * (1 + d.ret_oc[:, 3]) - 1
    cap = 1000.0 * np.prod(1 + r[: t + 1])
    assert np.isclose(ey[t, 3], ni3 / cap, rtol=1e-5) and np.isclose(bm[t, 3], 100.0 / cap, rtol=1e-5)


def test_point_in_time_inputs():
    d, t = _fake_with_drops()
    f_full, c_full = compute_inputs(d, PARAMS)
    cut = 290
    d2 = FakeData(320, 40, seed=5)
    for j in (3, 7, 11, 15):
        d2.shock(t - 1, j, -0.08)
    rng = np.random.default_rng(9)
    for a, lvl in ((d2.ret_co, 0.0), (d2.ret_oc, 0.0), (d2.close, 50.0), (d2.dollar_volume, 1e7)):
        a[cut + 1:] = lvl + rng.normal(0, 0.05, a[cut + 1:].shape) * max(lvl, 1.0)
    f2, c2 = compute_inputs(d2, PARAMS)
    assert np.array_equal(c_full[: cut + 1], c2[: cut + 1])
    for a, b in zip(f_full, f2):
        assert np.array_equal(np.isnan(a[: cut + 1]), np.isnan(b[: cut + 1]))
        assert np.allclose(a[: cut + 1], b[: cut + 1], equal_nan=True)


def test_missing_input_raises():
    d, _ = _fake_with_drops()
    del d.extras["sf1_art_equity"]
    with pytest.raises(KeyError):
        compute_inputs(d, PARAMS)
    d, _ = _fake_with_drops()
    d.universe = None
    with pytest.raises(ValueError):
        compute_inputs(d, PARAMS)


# ------------------------------------------------------------------ full pipeline (walk_forward)
def _model_data(T=1500, N=60, seed=2):
    d = FakeData(T, N, seed=seed)
    d.tradable = np.ones((T, N), bool)
    d.delisting = np.zeros((T, N), bool)
    d.dates = np.arange(T).astype("datetime64[D]")
    d.universe[:700, 50:] = False             # late entrants to the universe
    return d


def _head(d, n):
    e = FakeData(n, d.ret_co.shape[1], seed=0)
    e.ret_co, e.ret_oc = d.ret_co[:n].copy(), d.ret_oc[:n].copy()
    e.listed, e.close, e.dollar_volume = d.listed[:n].copy(), d.close[:n].copy(), d.dollar_volume[:n].copy()
    e.universe = d.universe[:n].copy()
    e.extras = {k: v[:n].copy() for k, v in d.extras.items()}
    e.tradable, e.delisting, e.dates = d.tradable[:n].copy(), d.delisting[:n].copy(), d.dates[:n].copy()
    return e


P_WIDE = {"tail_q": 0.5, "n_estimators": 20}      # wide tail so the small test market has cohorts


def test_pipeline_scores_only_candidates_and_book():
    d = _model_data()
    feats, cand = compute_inputs(d, P_WIDE)
    score = model_scores(d, P_WIDE, feats, cand)
    fin = np.isfinite(score)
    assert fin.any() and not (fin & ~cand).any()
    assert not fin[:252].any()                          # no scores before history + training
    lo, sh = cohort_legs(score)
    assert np.array_equal(lo.sum(1), sh.sum(1)) and not (lo & sh).any()
    assert not ((lo | sh) & ~d.universe).any()
    w = target_weights(d, P_WIDE)
    assert w.shape == d.ret_co.shape and np.all(np.isfinite(w))
    gross = np.abs(w).sum(1)
    assert np.all(gross <= 1.0 + 1e-9) and gross.max() > 0.5
    assert np.allclose(w.sum(1), 0.0, atol=1e-12)
    assert not (w[~d.universe] != 0).any()


def test_pipeline_point_in_time_truncation():
    d = _model_data()
    w_full = target_weights(d, P_WIDE)
    for cut in (800, 1100):
        w_cut = target_weights(_head(d, cut + 1), P_WIDE)
        assert np.allclose(w_full[: cut + 1], w_cut, atol=1e-12)


def test_n_estimators_changes_scores():
    d = _model_data()
    feats, cand = compute_inputs(d, P_WIDE)
    s1 = model_scores(d, P_WIDE, feats, cand)
    s2 = model_scores(d, {"tail_q": 0.5, "n_estimators": 50}, feats, cand)
    ok = np.isfinite(s1) & np.isfinite(s2)
    assert ok.any() and not np.allclose(s1[ok], s2[ok])


def test_params_complete():
    assert set(PARAMS) == {"n_estimators", "tail_q"}
    assert PARAMS["n_estimators"] == 100 and PARAMS["tail_q"] == 0.05
