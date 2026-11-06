import numpy as np
import pytest

from strategy import target_weights, PARAMS
import strategy as S


# ------------------------------------------------------------------------------------------------
# rolling / cross-sectional helpers
# ------------------------------------------------------------------------------------------------
def test_lagged_mean_std_window_ends_t_minus_1():
    x = np.arange(10, dtype=float).reshape(10, 1)
    m, s = S._lagged_mean_std(x, 3)
    assert np.all(np.isnan(m[:3]))
    # row 3 uses rows 0,1,2
    assert m[3, 0] == pytest.approx(1.0)
    assert s[3, 0] == pytest.approx(1.0)          # std ddof=1 of 0,1,2
    assert m[9, 0] == pytest.approx(7.0)          # rows 6,7,8
    # a NaN in the window blanks every row whose window contains it
    x2 = x.copy()
    x2[5, 0] = np.nan
    m2, _ = S._lagged_mean_std(x2, 3)
    assert np.isnan(m2[6:9, 0]).all()
    assert m2[9, 0] == pytest.approx(7.0)
    assert m2[5, 0] == pytest.approx(3.0)         # rows 2,3,4 do not include row 5


def test_lagged_mean_std_short_view():
    m, s = S._lagged_mean_std(np.ones((2, 3)), 63)
    assert m.shape == (2, 3) and np.isnan(m).all() and np.isnan(s).all()


def test_cs_median_and_mean_over_members():
    x = np.array([[1.0, 5.0, 3.0, 100.0], [np.nan, 2.0, 4.0, 6.0], [1.0, 2.0, 3.0, 4.0]])
    mask = np.array([[1, 1, 1, 0], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=bool)
    med = S._cs_median(x, mask)
    assert med[0] == pytest.approx(3.0)
    assert med[1] == pytest.approx(4.0)
    assert np.isnan(med[2])
    mean = S._cs_mean(x, mask)
    assert mean[0] == pytest.approx(3.0)
    assert mean[1] == pytest.approx(4.0)
    assert np.isnan(mean[2])
    # even count -> average of the middle two
    assert S._cs_median(np.array([[1.0, 2.0, 3.0, 10.0]]), np.ones((1, 4), bool))[0] == pytest.approx(2.5)


def test_lagged_median_on_members_only():
    x = np.arange(12, dtype=float).reshape(6, 2)
    mask = np.zeros((6, 2), bool)
    mask[4, 0] = True
    out = S._lagged_median_on(x, mask, 3)
    assert out[4, 0] == pytest.approx(4.0)        # rows 1,2,3 of col 0: 2,4,6
    assert np.isnan(out[4, 1]) and np.isnan(out[5]).all()


# ------------------------------------------------------------------------------------------------
# entry rule
# ------------------------------------------------------------------------------------------------
def test_entry_rule_quantile_and_minimum():
    T, N = 10, 30
    score = np.full((T, N), np.nan)
    score[2, :20] = np.arange(20)                  # 20 scored rows
    score[5, :30] = np.arange(30)                  # 50 scored rows in window
    e = S.entry_mask(score, 0.2, window=126, min_scored=40)
    assert not e[2].any()                          # only 20 scored so far -> no entries
    pool = np.concatenate([np.arange(20), np.arange(30)]).astype(float)
    thr = np.quantile(pool, 0.8)
    assert np.array_equal(np.nonzero(e[5])[0], np.nonzero(np.arange(30) >= thr)[0])
    # bigger top -> more entries
    e3 = S.entry_mask(score, 0.3, window=126, min_scored=40)
    assert e3[5].sum() > e[5].sum()
    # window: the row-2 scores drop out of a 3-row window at t=5
    e_w = S.entry_mask(score, 0.2, window=3, min_scored=30)
    thr_w = np.quantile(np.arange(30.0), 0.8)
    assert np.array_equal(np.nonzero(e_w[5])[0], np.nonzero(np.arange(30) >= thr_w)[0])
    # entries only where a score exists
    assert not e[~np.isfinite(score)].any()


def test_entry_rule_point_in_time():
    rng = np.random.default_rng(1)
    score = rng.normal(size=(300, 20))
    score[rng.random(score.shape) < 0.7] = np.nan
    full = S.entry_mask(score, 0.2)
    for cut in (120, 180, 250):
        part = S.entry_mask(score[:cut], 0.2)
        assert np.array_equal(full[:cut], part)


# ------------------------------------------------------------------------------------------------
# slot book
# ------------------------------------------------------------------------------------------------
def test_slot_book_hold_five_rows_and_sell():
    T, N = 12, 3
    entry = np.zeros((T, N), bool)
    entry[2, 0] = True
    score = np.ones((T, N))
    uni = np.ones((T, N), bool)
    listed = np.ones((T, N), bool)
    w = S.slot_book(entry, score, uni, listed, slots=20, hold=5)
    assert np.allclose(w[2:7, 0], 0.05)            # decided t=2..6 -> held open t+1 .. open t+6
    assert w[7, 0] == 0.0 and w[:2, 0].sum() == 0.0
    assert w[:, 1:].sum() == 0.0


def test_slot_book_no_stacking_and_universe_exit():
    T, N = 12, 2
    entry = np.zeros((T, N), bool)
    entry[1, 0] = True
    entry[3, 0] = True                              # second release while held: same slot, no extension
    entry[1, 1] = True
    score = np.ones((T, N))
    uni = np.ones((T, N), bool)
    uni[3, 1] = False                               # leaves the universe on row 3 -> sold
    listed = np.ones((T, N), bool)
    w = S.slot_book(entry, score, uni, listed, slots=20, hold=5)
    assert np.allclose(w[1:6, 0], 0.05) and w[6, 0] == 0.0
    assert w[:, 0].max() == pytest.approx(0.05)
    assert np.allclose(w[1:3, 1], 0.05) and (w[3:, 1] == 0.0).all()


def test_slot_book_cap_by_score():
    T, N = 8, 30
    entry = np.zeros((T, N), bool)
    entry[0, :18] = True
    entry[1, 18:30] = True                          # 12 new while 18 open -> only 2 admitted
    score = np.tile(np.arange(N, dtype=float), (T, 1))
    uni = np.ones((T, N), bool)
    listed = np.ones((T, N), bool)
    w = S.slot_book(entry, score, uni, listed, slots=20, hold=5)
    assert set(np.nonzero(w[1, 18:])[0] + 18) == {28, 29}   # highest scores first
    assert np.abs(w).sum(axis=1).max() <= 1.0 + 1e-12
    assert (w >= 0).all()
    # after the first batch exits (row 5), nothing new is opened without entries
    assert w[5, :18].sum() == 0.0


# ------------------------------------------------------------------------------------------------
# features on a hand-made view
# ------------------------------------------------------------------------------------------------
class _View:
    """Duck-typed DataView with the fields the strategy reads."""

    def __init__(self, T, N, seed=0):
        rng = np.random.default_rng(seed)
        self.dates = np.arange(T).astype("datetime64[D]")
        self.instruments = tuple("S%02d" % j for j in range(N))
        self.ret_co = rng.normal(0, 0.005, (T, N))
        self.ret_oc = rng.normal(0, 0.01, (T, N))
        self.listed = np.ones((T, N), bool)
        self.tradable = np.ones((T, N), bool)
        self.delisting = np.zeros((T, N), bool)
        self.dollar_volume = np.exp(rng.normal(17, 0.5, (T, N)))
        self.close = np.full((T, N), 50.0)
        self.universe = np.ones((T, N), bool)
        ea = np.zeros((T, N))
        self.extras = {"ea_reaction_day": ea, "f13_io": rng.uniform(0.3, 0.9, (T, N))}
        self.series = {}

    def col(self, name):
        return self.instruments.index(name)


def test_raw_features_hand_values():
    T, N = 100, 5
    v = _View(T, N)
    uni = v.universe
    react, io, liq, relvol, runup, ivol = S.raw_features(v, uni)
    t, j = 80, 2
    r = (1 + v.ret_co) * (1 + v.ret_oc) - 1
    sd = np.std(r[t - 63:t, j], ddof=1)
    assert react[t, j] == pytest.approx((r[t, j] - np.median(r[t])) / sd)
    assert liq[t, j] == pytest.approx(np.log(np.mean(v.dollar_volume[t - 63:t, j])))
    assert relvol[t, j] == pytest.approx(np.log(v.dollar_volume[t, j] / np.median(v.dollar_volume[t - 63:t, j])))
    resid = r - r.mean(axis=1, keepdims=True)
    assert ivol[t, j] == pytest.approx(np.std(resid[t - 63:t, j], ddof=1))
    assert io[t, j] == pytest.approx(v.extras["f13_io"][t, j])
    assert np.isnan(react[:63]).all() and np.isnan(ivol[:63]).all()


def test_features_late_listing_and_nan_io():
    T, N = 120, 4
    v = _View(T, N, seed=3)
    v.listed[:90, 3] = False
    v.universe[:90, 3] = False
    v.dollar_volume[:90, 3] = np.nan
    v.extras["f13_io"][:, 1] = np.nan
    react, io, liq, relvol, runup, ivol = S.raw_features(v, v.universe)
    assert np.isnan(react[:90 + 63, 3]).all()       # needs 63 listed sessions before t
    assert np.isnan(liq[:90 + 63, 3]).all()
    assert np.isnan(io[:, 1]).all()                 # unknown IO stays NaN (row not scored)
    assert np.isfinite(react[100, :3]).all()


def test_missing_inputs_raise():
    v = _View(80, 3)
    del v.extras["f13_io"]
    with pytest.raises(KeyError):
        S.raw_features(v, v.universe)
    v2 = _View(80, 3)
    v2.universe = None
    with pytest.raises(ValueError):
        S.release_rows(v2)


# ------------------------------------------------------------------------------------------------
# whole pipeline with a stand-in model (the real walk_forward is framework code)
# ------------------------------------------------------------------------------------------------
def test_pipeline_with_stub_model(monkeypatch):
    T, N = 400, 40
    v = _View(T, N, seed=5)
    rng = np.random.default_rng(7)
    v.extras["ea_reaction_day"] = (rng.random((T, N)) < 0.15).astype(float)
    seen = {}

    def fake_wf(X, y, horizon, model, params, retrain_every, universe, **kw):
        seen.update(horizon=horizon, model=model, params=dict(params), retrain_every=retrain_every)
        assert X.ndim == 3 and X.shape[1:] == (N, 6)
        complete = np.isfinite(X).all(axis=2)
        s = np.where(complete, np.where(np.isfinite(X), X, 0.0).sum(axis=2), np.nan)
        s[:150] = np.nan                            # nothing scored before the first fit
        return s

    monkeypatch.setattr(S, "walk_forward", fake_wf)
    w = S.target_weights(v, PARAMS)
    assert seen["model"] == "gbm" and seen["horizon"] == 5 and seen["retrain_every"] == 63
    assert seen["params"]["n_estimators"] == 100 and seen["params"]["num_leaves"] == 8
    assert w.shape == (T, N)
    assert (w >= 0).all()
    assert np.abs(w).sum(axis=1).max() <= 1.0 + 1e-12
    assert w[:150].sum() == 0.0
    assert w[150:].sum() > 0
    # every held weight is 1/20
    assert np.allclose(w[w > 0], 0.05)
    # positions only in release-row members' slots
    rel = (v.extras["ea_reaction_day"] == 1) & v.universe
    first = (w > 0) & ~np.vstack([np.zeros((1, N), bool), w[:-1] > 0])
    assert rel[first].all()
    # top changes the book; n_estimators is passed through
    w3 = S.target_weights(v, {"n_estimators": 100, "top": 0.3})
    assert not np.array_equal(w, w3)
    S.target_weights(v, {"n_estimators": 50, "top": 0.2})
    assert seen["params"]["n_estimators"] == 50
    # point in time: truncation leaves earlier rows unchanged
    for cut in (200, 300):
        vc = _View(T, N, seed=5)
        vc.ret_co = v.ret_co[:cut]
        vc.ret_oc = v.ret_oc[:cut]
        vc.listed = v.listed[:cut]
        vc.tradable = v.tradable[:cut]
        vc.delisting = v.delisting[:cut]
        vc.dollar_volume = v.dollar_volume[:cut]
        vc.close = v.close[:cut]
        vc.universe = v.universe[:cut]
        vc.dates = v.dates[:cut]
        vc.extras = {k: a[:cut] for k, a in v.extras.items()}
        wc = S.target_weights(vc, PARAMS)
        assert np.array_equal(w[:cut], wc)
