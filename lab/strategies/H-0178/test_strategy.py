import dataclasses
import numpy as np
import pytest
from lab.framework.data import synthetic
from lab.framework.api import period_starts
from strategy import (PARAMS, GBM_FIXED, cap_stale, trend_z, stock_betas, winsorize_rows, sleeve_book, features,
                      model_scores, target_weights, decision_rows)


# ---------------------------------------------------------------- helpers on hand-made inputs

def test_cap_stale_five_rows():
    v = np.array([np.nan, 1, 1, 1, 1, 1, 1, 1, 2, 2, np.nan, 3.0])
    out = cap_stale(v, 5)
    # value 1 first seen at row 1; carried rows 2..6 (age 1..5) kept, row 7 (age 6) -> NaN
    assert np.isnan(out[0])
    assert np.all(out[1:7] == 1)
    assert np.isnan(out[7])
    assert out[8] == 2 and out[9] == 2
    assert np.isnan(out[10]) and out[11] == 3


def test_trend_z_formula_and_warmup():
    rng = np.random.default_rng(1)
    F = np.cumsum(rng.normal(0, 0.1, 600))
    rows = np.array([100, 251, 252, 400, 599])
    z = trend_z(F, rows, 63)
    assert np.isnan(z[100]) and np.isnan(z[251])          # fewer than 252 daily changes
    for t in (252, 400, 599):
        dF = np.diff(F)[t - 252:t]                          # changes ending at row t (252 of them)
        expect = (F[t] - F[t - 63]) / (np.std(dF, ddof=1) * np.sqrt(63))
        assert z[t] == pytest.approx(expect, rel=1e-12)
    assert np.isnan(z[300])                                 # not a requested row
    z2 = trend_z(F, rows, 126)
    assert z2[400] != z[400]


def _beta_market(n_weeks_c=70):
    """Three member stocks whose 5-row returns are exact linear functions; their EW mean is the market m."""
    rng = np.random.default_rng(7)
    T = 5 * 120 + 3
    t = T - 1
    Fy = np.cumsum(rng.normal(0, 0.05, T))
    Fo = np.cumsum(rng.normal(0, 0.02, T))
    Fd = np.cumsum(rng.normal(0, 0.01, T))
    m = rng.normal(0, 0.02, T)
    beta_a = np.array([0.001, 1.0, 0.5, 0.2, -0.4])
    beta_b = np.array([-0.001, 1.0, -0.5, -0.2, 0.4])
    beta_c = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    idx = np.ones((T, 3))
    listed = np.ones((T, 3), dtype=bool)
    grid = np.arange(t % 5, T, 5)
    for k in range(1, len(grid)):
        s, p = grid[k], grid[k - 1]
        x = np.array([1.0, m[s], Fy[s] - Fy[p], Fo[s] - Fo[p], Fd[s] - Fd[p]])
        for j, b in enumerate((beta_a, beta_b, beta_c)):
            idx[s, j] = idx[p, j] * (1 + x @ b)
            idx[p + 1:s, j] = idx[p, j]
    first_c = t - 5 * n_weeks_c                             # C listed only for the last n_weeks_c weeks
    listed[:first_c, 2] = False
    idx[:first_c, 2] = np.nan
    uni = np.ones((T, 3), dtype=bool)
    return idx, listed, uni, (Fy, Fo, Fd), t, beta_a, beta_b


def test_stock_betas_recover_partial_betas_and_min_weeks():
    idx, listed, uni, facs, t, ba, bb = _beta_market(70)
    by, bo, bd = stock_betas(idx, listed, uni, facs, np.array([t]))
    assert by[t, 0] == pytest.approx(ba[2], abs=1e-4)
    assert bo[t, 0] == pytest.approx(ba[3], abs=1e-4)
    assert bd[t, 0] == pytest.approx(ba[4], abs=1e-4)
    assert by[t, 1] == pytest.approx(bb[2], abs=1e-4)
    assert bd[t, 1] == pytest.approx(bb[4], abs=1e-4)
    assert np.isnan(by[t, 2])                               # 70 < 78 valid weeks: not scored
    assert np.all(np.isnan(by[:t]))                         # only requested rows
    idx, listed, uni, facs, t, _, _ = _beta_market(90)
    by, _, _ = stock_betas(idx, listed, uni, facs, np.array([t]))
    assert by[t, 2] == pytest.approx(0.0, abs=1e-4)         # 90 >= 78 weeks: scored


def test_stock_betas_non_member_not_scored():
    idx, listed, uni, facs, t, _, _ = _beta_market(90)
    uni = uni.copy()
    uni[t, 1] = False
    by, _, _ = stock_betas(idx, listed, uni, facs, np.array([t]))
    assert np.isnan(by[t, 1]) and np.isfinite(by[t, 0])


def test_winsorize_rows():
    x = np.full((3, 201), np.nan, dtype=np.float32)
    x[1] = np.arange(201, dtype=np.float32)
    x[1, 0] = -1000.0
    x[1, 200] = 1000.0
    x[1, 50] = np.nan
    w = winsorize_rows(x, np.array([1]))
    vals = x[1][np.isfinite(x[1])].astype(np.float64)
    lo, hi = np.percentile(vals, [1, 99])
    assert w[1, 0] == pytest.approx(lo) and w[1, 200] == pytest.approx(hi)
    assert w[1, 100] == 100 and np.isnan(w[1, 50])
    assert np.all(np.isnan(w[0]))


# ---------------------------------------------------------------- the two-sleeve book

def _sleeve_inputs():
    T, N = 6, 70
    rows = np.array([0, 2, 4, 5])
    uni = np.zeros((T, N), dtype=bool)
    uni[:, :60] = True
    score = np.full((T, N), np.nan)
    score[2, :60] = np.arange(60)                           # top fifth = columns 48..59
    score[4, :60] = -np.arange(60.0)                        # top fifth = columns 0..11
    score[5, :40] = np.arange(40)                           # only 40 scored: EW sleeve
    keep = uni.copy()
    keep[4, 50] = False                                     # stops trading before row 4
    return score, uni, keep, rows


def test_sleeve_book_mechanics():
    score, uni, keep, rows = _sleeve_inputs()
    W, is_model, in_top = sleeve_book(score, uni, keep, rows)
    ew = np.where(uni[0], 1 / 60, 0.0)
    assert np.allclose(W[0], ew)                            # warm-up: EW universe, not cash
    assert not is_model[0]
    top2 = np.zeros(70)
    top2[48:60] = 1 / 12
    assert is_model[2] and np.flatnonzero(in_top[2]).tolist() == list(range(48, 60))
    assert np.allclose(W[2], 0.5 * top2 + 0.5 * ew)
    top4 = np.zeros(70)
    top4[:12] = 1 / 12
    prev = top2.copy()
    prev[50] = 0
    prev /= prev.sum()                                      # 11 names renormalized
    assert np.allclose(W[4], 0.5 * top4 + 0.5 * prev)
    assert not is_model[5]                                  # fewer than 50 scored -> EW members
    assert np.allclose(W[5], 0.5 * ew + 0.5 * top4)
    assert np.all(np.isnan(W[[1, 3]]))
    for t in rows:
        assert W[t].min() >= 0 and W[t].sum() == pytest.approx(1.0)
        assert np.all(W[t][~uni[t]] == 0)


# ---------------------------------------------------------------- full pipeline on the synthetic market

def _view(late=True):
    d = synthetic()
    T = len(d.dates)
    N = len(d.instruments)
    rng = np.random.default_rng(3)
    uni = np.zeros((T, N), dtype=bool)
    for j, name in enumerate(d.instruments):
        if name.startswith("S"):
            uni[:, j] = np.asarray(d.listed)[:, j]
    if late:
        uni[: T // 2, d.col("S05")] = False                 # a late entrant
    wti = 30 * np.exp(np.cumsum(rng.normal(0, 0.02, T)))
    wti[T // 3] = -5.0                                      # a non-positive print
    series = dict(d.series)
    series["fred_macro.DGS10"] = (4 + np.cumsum(rng.normal(0, 0.05, T))).astype(np.float64)
    series["fred_macro.DCOILWTICO"] = wti
    series["fred_macro.DEXUSEU"] = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.005, T)))
    return dataclasses.replace(d, universe=uni, series=series)


def test_features_shapes_and_nan_off_decision_rows():
    d = _view()
    feats, rows = features(d, PARAMS)
    T, N = np.asarray(d.universe).shape
    assert len(feats) == 6
    dec = np.zeros(T, dtype=bool)
    dec[rows] = True
    uni = np.asarray(d.universe)
    for f in feats:
        f = np.asarray(f)
        assert f.shape == (T, N)
        assert np.all(np.isnan(f[~dec]))
        assert np.all(np.isnan(f[~uni]))
    for f in feats[3:]:                                     # date-level, raw, same for every member
        f = np.asarray(f)
        for t in rows[-5:]:
            v = f[t][uni[t]]
            assert np.all(v == v[0]) and np.isfinite(v[0])
    t = rows[-1]
    assert np.isfinite(np.asarray(feats[0])[t][uni[t]]).all()
    assert np.all(np.isnan(np.asarray(feats[0])[: 5 * 78]))  # betas need >= 78 weeks


def test_trend_lookback_changes_only_trend_features():
    d = _view()
    f1, rows = features(d, PARAMS)
    f2, _ = features(d, dict(PARAMS, trend_lookback=126))
    t = rows[-1]
    assert np.array_equal(np.asarray(f1[0]), np.asarray(f2[0]), equal_nan=True)
    assert not np.allclose(np.asarray(f1[3])[t], np.asarray(f2[3])[t], equal_nan=True)


def test_missing_fred_series_raises():
    d = _view()
    series = dict(d.series)
    del series["fred_macro.DEXUSEU"]
    with pytest.raises(KeyError):
        features(dataclasses.replace(d, series=series), PARAMS)


def test_features_point_in_time_truncation():
    d = _view()
    f_full, rows = features(d, PARAMS)
    cut = int(rows[-20]) + 3
    sl = slice(0, cut + 1)
    dt = dataclasses.replace(
        d, dates=d.dates[sl], ret_co=d.ret_co[sl], ret_oc=d.ret_oc[sl], tradable=d.tradable[sl],
        listed=d.listed[sl], delisting=d.delisting[sl], close=d.close[sl], dollar_volume=d.dollar_volume[sl],
        universe=np.asarray(d.universe)[sl], cash_ret=None if d.cash_ret is None else d.cash_ret[sl],
        series={k: np.asarray(v)[sl] for k, v in d.series.items()},
        extras={k: np.asarray(v)[sl] for k, v in d.extras.items()})
    f_cut, _ = features(dt, PARAMS)
    for a, b in zip(f_full, f_cut):
        assert np.array_equal(np.asarray(a)[: cut + 1], np.asarray(b), equal_nan=True)


def test_full_pipeline_gbm_params_and_weights():
    d = _view()
    score, rows = model_scores(d, PARAMS)                   # also checks the gbm hyperparameter names
    assert score.shape == np.asarray(d.universe).shape
    assert set(GBM_FIXED) == {"num_leaves", "learning_rate", "min_child_samples", "subsample", "colsample"}
    W = np.asarray(target_weights(d, PARAMS))
    dec = np.zeros(len(d.dates), dtype=bool)
    dec[rows] = True
    assert np.all(np.isnan(W[~dec]))
    uni = np.asarray(d.universe)
    for t in rows:
        assert np.nanmin(W[t]) >= 0
        assert np.nansum(W[t]) <= 1 + 1e-9
        assert np.nansum(W[t]) == pytest.approx(1.0)         # never cash: < 50 members here -> EW book
        assert np.all(W[t][~uni[t]] == 0)
    ps, r2 = decision_rows(d.dates)
    assert np.array_equal(r2, rows)
    assert np.array_equal(np.flatnonzero(np.asarray(period_starts(d.dates, "M"))), rows)
