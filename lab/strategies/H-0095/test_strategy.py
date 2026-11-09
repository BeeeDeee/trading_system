import dataclasses

import numpy as np
import pytest

from lab.framework.data import DataView, synthetic
from strategy import PARAMS, target_weights
from strategy import (build_book, core_weights, entry_flags, event_mask, first_fit_row, gbm_params,
                      lagged_median, lagged_std, member_mean, raw_features, trailing_mean)
import strategy as S


def test_params_names():
    assert set(PARAMS) == {"entry_quantile", "n_estimators"}
    assert callable(target_weights)


def test_gbm_params_reads_n_estimators():
    p = gbm_params({"n_estimators": 50, "entry_quantile": 0.7})
    assert p["n_estimators"] == 50
    assert p["num_leaves"] == 8 and p["min_child_samples"] == 200
    assert gbm_params({"n_estimators": 200})["n_estimators"] == 200


def test_member_mean_only_members():
    x = np.array([[1.0, 2.0, 100.0], [np.nan, 4.0, 6.0]])
    uni = np.array([[True, True, False], [True, True, True]])
    m = member_mean(x, uni)
    assert m[0] == pytest.approx(1.5)
    assert m[1] == pytest.approx(5.0)


def test_lagged_std_window_excludes_today():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 2))
    sd = lagged_std(x, 63)
    assert np.all(np.isnan(sd[:63]))
    for t in (63, 70, 79):
        assert sd[t, 0] == pytest.approx(np.std(x[t - 63:t, 0], ddof=1), rel=1e-9)
    x2 = x.copy()
    x2[65, 1] = np.nan
    sd2 = lagged_std(x2, 63)
    assert np.isnan(sd2[66, 1]) and np.isnan(sd2[79, 1])   # window with a gap -> NaN
    assert np.isfinite(sd2[65, 1])                         # today's value is not in the window


def test_lagged_median_and_trailing_mean():
    x = np.arange(1.0, 101.0)[:, None] * np.ones((1, 2))
    mask = np.ones((100, 2), dtype=bool)
    med = lagged_median(x, mask, 63)
    assert np.isnan(med[0, 0])                            # no earlier value
    assert med[62, 0] == pytest.approx(31.5)              # finite values of the window only (rows 0..61)
    # rows t-63..t-1 hold values t-62..t ; median of 63 consecutive ints = middle one
    assert med[63, 0] == pytest.approx(32.0)
    assert med[99, 1] == pytest.approx(np.median(x[36:99, 1]))
    mask[:, 1] = False
    assert np.all(np.isnan(lagged_median(x, mask, 63)[:, 1]))
    tm = trailing_mean(x, 63)
    assert tm[62, 0] == pytest.approx(np.mean(x[0:63, 0]))
    assert tm[99, 0] == pytest.approx(np.mean(x[37:100, 0]))


def _small_market(T=260, N=4, seed=3):
    rng = np.random.default_rng(seed)
    ret_co = rng.normal(0, 0.005, size=(T, N))
    ret_oc = rng.normal(0, 0.01, size=(T, N))
    listed = np.ones((T, N), dtype=bool)
    dv = rng.uniform(1e6, 2e6, size=(T, N))
    uni = np.ones((T, N), dtype=bool)
    ea = np.zeros((T, N))
    return ret_co, ret_oc, listed, dv, uni, ea


def test_features_by_hand():
    ret_co, ret_oc, listed, dv, uni, ea = _small_market()
    f1, f2, f3, f4, f5, mdv = raw_features(ret_co, ret_oc, listed, dv, uni, ea)
    cc = (1 + ret_co) * (1 + ret_oc) - 1
    abn = cc - cc.mean(axis=1, keepdims=True)
    t, j = 100, 2
    sd = np.std(abn[t - 63:t, j], ddof=1)
    assert f1[t, j] == pytest.approx((ret_co[t, j] - ret_co[t].mean()) / sd, rel=1e-9)
    assert f2[t, j] == pytest.approx((ret_oc[t, j] - ret_oc[t].mean()) / sd, rel=1e-9)
    assert f3[t, j] == pytest.approx(np.log(dv[t, j] / np.median(dv[t - 63:t, j])), rel=1e-9)
    runup = np.prod(1 + abn[t - 21:t, j]) - 1
    assert f4[t, j] == pytest.approx(runup / (sd * np.sqrt(21)), rel=1e-9)
    assert mdv[t, j] == pytest.approx(dv[t - 62:t + 1, j].mean(), rel=1e-9)
    assert np.isnan(f1[62, j]) and np.isfinite(f1[63, j])
    assert np.all(f5 == 0.0)                              # no release ever -> 0


def test_prev_reaction_window():
    ret_co, ret_oc, listed, dv, uni, ea = _small_market()
    ea[80, 1] = 1.0
    f1, f2, f3, f4, f5, mdv = raw_features(ret_co, ret_oc, listed, dv, uni, ea)
    cc = (1 + ret_co) * (1 + ret_oc) - 1
    abn = cc - cc.mean(axis=1, keepdims=True)
    rz = abn[80, 1] / np.std(abn[17:80, 1], ddof=1)
    assert f5[80, 1] == 0.0                               # strictly earlier events only
    assert f5[81, 1] == pytest.approx(rz, rel=1e-9)
    assert f5[220, 1] == pytest.approx(rz, rel=1e-9)      # 140 sessions after
    assert f5[221, 1] == 0.0                              # 141 sessions after -> none
    assert np.all(f5[:, 0] == 0.0)
    # a later event replaces the earlier one
    ea[150, 1] = 1.0
    f5b = raw_features(ret_co, ret_oc, listed, dv, uni, ea)[4]
    rz2 = abn[150, 1] / np.std(abn[87:150, 1], ddof=1)
    assert f5b[151, 1] == pytest.approx(rz2, rel=1e-9)


def test_features_late_listing_and_short_view():
    ret_co, ret_oc, listed, dv, uni, ea = _small_market()
    listed[:100, 3] = False
    uni[:100, 3] = False
    ret_co[:100, 3] = 0.0
    dv[:100, 3] = np.nan
    f1 = raw_features(ret_co, ret_oc, listed, dv, uni, ea)[0]
    assert np.all(np.isnan(f1[:163, 3]))
    assert np.isfinite(f1[163, 3])
    # views shorter than the windows do not crash
    out = raw_features(ret_co[:10], ret_oc[:10], listed[:10], dv[:10], uni[:10], ea[:10])
    assert np.all(np.isnan(out[0]))


def test_event_mask_and_first_fit_row():
    ea = np.array([[np.nan, np.nan], [np.nan, 0.0], [1.0, 1.0], [0.0, 1.0]])
    uni = np.array([[True, True], [True, True], [True, False], [True, True]])
    m = event_mask(ea, uni)
    assert m.tolist() == [[False, False], [False, False], [True, False], [False, True]]
    assert first_fit_row(ea) == 1 + 504
    assert first_fit_row(np.full((3, 2), np.nan)) == 3


def test_entry_flags_percentile_rules():
    T, N = 600, 3
    score = np.full((T, N), np.nan)
    score[50:, 0] = np.arange(T - 50, dtype=float)        # increasing: always above the whole pool
    e = entry_flags(score, 0.7)
    assert not e[50:176, 0].any()                         # fewer than 126 sessions of earlier scores
    assert e[176:, 0].all()
    assert not e[:, 1:].any()
    score2 = np.full((T, N), np.nan)
    score2[50:, 0] = -np.arange(T - 50, dtype=float)      # decreasing: always below the pool
    assert not entry_flags(score2, 0.7).any()


def test_entry_flags_pool_is_previous_252_sessions():
    T, N = 700, 2
    score = np.full((T, N), np.nan)
    score[0:300, 0] = 0.0                                 # old scores
    score[300:600, 0] = 10.0                              # recent pool
    score[600, 1] = 5.0                                   # below all of rows 348..599 -> pct 0
    score[650, 1] = 5.0
    e = entry_flags(score, 0.7)
    assert not e[600, 1]
    score[600, 1] = 10.5
    e = entry_flags(score, 0.7)
    assert e[600, 1]
    # ties do not count as below: pct = share strictly below
    score3 = np.full((T, N), np.nan)
    score3[0:400, 0] = 1.0
    score3[400, 1] = 1.0
    assert not entry_flags(score3, 0.7)[400, 1]
    assert entry_flags(score3, 0.0)[400, 1]


def test_entry_quantile_changes_entries():
    rng = np.random.default_rng(7)
    score = rng.normal(size=(500, 20))
    score[rng.uniform(size=score.shape) < 0.8] = np.nan
    n6 = entry_flags(score, 0.6).sum()
    n7 = entry_flags(score, 0.7).sum()
    n8 = entry_flags(score, 0.8).sum()
    assert n6 > n7 > n8 > 0


def test_core_weights_proxy():
    mem = np.array([True, True, True, True, False])
    mcap = np.array([100.0, 300.0, np.nan, np.nan, 500.0])
    mdv = np.array([10.0, 10.0, 20.0, np.nan, 10.0])
    w = core_weights(mcap, mdv, mem)
    # ratios 10 and 30 -> median 20 ; stock 2 proxy 400 ; stock 3 no data ; stock 4 not a member
    assert w == pytest.approx(np.array([100.0, 300.0, 400.0, 0.0, 0.0]) / 800.0)
    w2 = core_weights(np.full(5, np.nan), mdv, mem)       # no marketcap at all -> dollar-volume core
    assert w2 == pytest.approx(np.array([10.0, 10.0, 20.0, 0.0, 0.0]) / 40.0)


def test_book_slots_hold_and_gross():
    T, N = 20, 4
    uni = np.ones((T, N), dtype=bool)
    mcap = np.tile(np.array([1.0, 1.0, 1.0, 1.0]), (T, 1))
    mdv = np.ones((T, N))
    reb = np.zeros(T, dtype=bool)
    reb[0] = True
    entry = np.zeros((T, N), dtype=bool)
    entry[2, 1] = True
    entry[4, 1] = True                                    # inside the hold: no new slot, no extension
    W = build_book(entry, uni, mcap, mdv, reb)
    assert np.allclose(W.sum(axis=1), 1.0)
    assert (W >= 0).all()
    assert W[1] == pytest.approx(np.full(4, 0.25))
    for t in range(2, 7):
        assert W[t, 1] == pytest.approx(0.98 * 0.25 + 0.02)
        assert W[t, 0] == pytest.approx(0.98 * 0.25)
    assert W[7] == pytest.approx(np.full(4, 0.25))        # exit at the open of t+6


def test_book_universe_exit_and_core_fixed_between_rebalances():
    T, N = 12, 3
    uni = np.ones((T, N), dtype=bool)
    uni[5:, 2] = False
    mcap = np.tile(np.array([1.0, 1.0, 2.0]), (T, 1))
    mcap[3:, 0] = 5.0                                     # new filing after the rebalance: ignored
    mdv = np.ones((T, N))
    reb = np.zeros(T, dtype=bool)
    reb[0] = True
    reb[10] = True
    entry = np.zeros((T, N), dtype=bool)
    entry[3, 2] = True
    W = build_book(entry, uni, mcap, mdv, reb)
    assert W[3, 2] == pytest.approx(0.98 * 0.5 + 0.02)
    assert W[4, 2] == pytest.approx(0.98 * 0.5 + 0.02)
    assert W[5, 2] == 0.0                                 # left the universe: closed, core renormalized
    assert W[5] == pytest.approx(np.array([0.5, 0.5, 0.0]))
    assert W[9] == pytest.approx(np.array([0.5, 0.5, 0.0]))
    assert W[10] == pytest.approx(np.array([5.0 / 6.0, 1.0 / 6.0, 0.0]))


def test_book_tilt_cap_scales_new_slots_only():
    T, N = 6, 40
    uni = np.ones((T, N), dtype=bool)
    mcap = np.ones((T, N))
    mdv = np.ones((T, N))
    reb = np.zeros(T, dtype=bool)
    reb[0] = True
    entry = np.zeros((T, N), dtype=bool)
    entry[0, :10] = True                                  # 0.20 of tilt
    entry[1, 10:20] = True                                # would be 0.40 -> new ones get 0.01 each
    W = build_book(entry, uni, mcap, mdv, reb)
    core = 0.7 / 40
    assert W[1, 0] == pytest.approx(core + 0.02)
    assert W[1, 15] == pytest.approx(core + 0.01)
    assert W[1, 30] == pytest.approx(core)
    assert np.allclose(W.sum(axis=1), 1.0)
    entry2 = entry.copy()
    entry2[2, 30] = True                                  # cap full: no slot
    W2 = build_book(entry2, uni, mcap, mdv, reb)
    assert W2[2, 30] == pytest.approx(core)
    assert np.allclose(W2.sum(axis=1), 1.0)


# ----------------------------------------------------------------------------- point-in-time regression
# G0 failed on real data because the scored columns were "ever a member" over the whole view: a stock
# that joins the universe after the cut changed the column set, and with it the model's fits.

def _pit_view():
    d = synthetic()
    f = dataclasses.asdict(d)
    rng = np.random.default_rng(5)
    T0 = 2200
    out = {}
    for k, v in f.items():
        out[k] = v[:T0].copy() if isinstance(v, np.ndarray) and v.shape[:1] == d.dates.shape else v
    T, N = out["ret_co"].shape
    uni = np.zeros((T, N), dtype=bool)
    for j in range(N):
        a = int(rng.integers(0, T // 2))
        b = int(rng.integers(T // 2, T + 400))
        uni[a:b, j] = True                                # late entries (some after row 900) and exits
    uni[:1500, N - 1] = False
    uni[1500:, N - 1] = True
    ea = np.where(rng.uniform(size=(T, N)) < 0.3, 1.0, 0.0)
    ea[:150] = np.nan                                     # event data starts late
    out["dollar_volume"] = out["dollar_volume"] * rng.lognormal(0, 0.5, size=(T, N))
    out["universe"] = uni
    out["extras"] = {"ea_reaction_day": ea.astype(np.float32),
                     "sf1_arq_marketcap": rng.uniform(1e9, 1e10, size=(T, N)).astype(np.float32)}
    out["series"] = {}
    return out


def _cut(f, c):
    g = {}
    for k, v in f.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == f["dates"].shape:
            g[k] = v[:c + 1].copy()
        elif isinstance(v, dict):
            g[k] = {kk: vv[:c + 1].copy() for kk, vv in v.items()}
        else:
            g[k] = v
    return g


def _perturb(f, c, seed):
    rng = np.random.default_rng(seed)
    g = dict(f)
    for k in ("ret_co", "ret_oc"):
        w = f[k].copy()
        w[c + 1:] = rng.normal(0, 0.05, size=w[c + 1:].shape)
        g[k] = w
    w = f["dollar_volume"].copy()
    w[c + 1:] = rng.uniform(1, 1e9, size=w[c + 1:].shape)
    g["dollar_volume"] = w
    w = f["universe"].copy()
    w[c + 1:] = rng.uniform(size=w[c + 1:].shape) < 0.5
    g["universe"] = w
    g["extras"] = {}
    for kk, vv in f["extras"].items():
        w = vv.copy()
        w[c + 1:] = (rng.uniform(size=w[c + 1:].shape) < 0.3).astype(w.dtype)
        g["extras"][kk] = w
    return g


def _equal(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(np.nan_to_num(a) - np.nan_to_num(b)) <= 1e-12)))


def test_point_in_time_with_events_and_late_members():
    f = _pit_view()
    p = {"entry_quantile": 0.7, "n_estimators": 10}
    full = DataView(**f)
    sc_full, _ = S.event_scores(full, p)
    w_full = target_weights(full, p)
    assert np.isfinite(sc_full).any() and S.entry_flags(sc_full, 0.7).any()   # the model path is exercised
    for c in (900, 1500):
        for g in (_cut(f, c), _perturb(f, c, c)):
            v = DataView(**g)
            sc, _ = S.event_scores(v, p)
            assert _equal(sc[:c + 1], sc_full[:c + 1])
            assert _equal(target_weights(v, p)[:c + 1], w_full[:c + 1])


def test_constants_match_card():
    assert S.HOLD == 5 and S.SLOT_W == 0.02 and S.TILT_CAP == 0.30
    assert S.PCT_WIN == 252 and S.PCT_MIN == 126 and S.MIN_FIT_ROWS == 504 and S.RETRAIN == 63
