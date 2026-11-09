import numpy as np
import pytest
from lab.framework.data import DataView

from strategy import (PARAMS, build_book, compute_features, core_weights, model_scores, quarter_calendar,
                      target_weights, window_end_rows, zscore_scored)

SH = "sf1_arq_sharesbas"


def _weekdays(start_offset, n):
    # weekday dates from a day offset relative to 1970-01-01 (no literals, no holidays)
    base = np.arange(start_offset, start_offset + 3 * n).astype("datetime64[D]")
    return base[np.is_busday(base)][:n]


def _view(T, N, ret_oc=None, close=None, S=None, universe=None, dv=None, start=10000):
    dates = _weekdays(start, T)
    z = np.zeros((T, N))
    roc = z.copy() if ret_oc is None else ret_oc
    if close is None:
        close = 10.0 * np.cumprod(1.0 + roc, axis=0)
    return DataView(dates=dates, instruments=tuple("E%d" % (j + 1) for j in range(N)),
                    asset_class=tuple("us_equity" for _ in range(N)), ret_co=z.copy(), ret_oc=roc,
                    tradable=np.ones((T, N), bool), listed=np.ones((T, N), bool), delisting=np.zeros((T, N), bool),
                    close=close, dollar_volume=(z + 1e6) if dv is None else dv,
                    universe=np.ones((T, N), bool) if universe is None else universe,
                    extras={SH: (z + 100.0) if S is None else S}, series={}, cash_ret=np.zeros(T))


def test_params():
    assert PARAMS == {"n_estimators": 100, "top": 0.1}


# ------------------------------------------------------------------------------------------------ calendar
def test_calendar_decisions_are_five_weekdays_before_third_friday():
    dates = _weekdays(10000, 600)
    is_dec, tf = quarter_calendar(dates)
    rows = np.flatnonzero(is_dec)
    assert rows.size >= 8
    ends = window_end_rows(dates, is_dec, tf)
    for d in rows:
        f = tf[d]
        assert f.astype("datetime64[M]").astype(np.int64) % 12 + 1 in (3, 6, 9, 12)
        assert np.is_busday(f, weekmask="Fri")
        day = (f - f.astype("datetime64[M]").astype("datetime64[D]")).astype(int) + 1
        assert 15 <= day <= 21                          # third Friday
        assert dates[d].astype("datetime64[M]") == f.astype("datetime64[M]")
        if ends[int(d)] >= 0:                           # no holidays here: R is exactly 5 rows after D
            assert ends[int(d)] - d == 5
            assert dates[ends[int(d)]] == f
    # consecutive decisions are a quarter apart
    assert np.all(np.diff(rows) > 55) and np.all(np.diff(rows) < 70)


def test_calendar_missing_day_and_point_in_time():
    dates = _weekdays(10000, 600)
    is_dec, tf = quarter_calendar(dates)
    d = int(np.flatnonzero(is_dec)[2])
    # remove the decision day itself (a holiday): the next session becomes the decision row
    keep = np.ones(dates.size, bool)
    keep[d] = False
    is2, _ = quarter_calendar(dates[keep])
    assert is2[d] and not is2[d - 1]
    # truncation does not change earlier rows
    for k in (50, 211, 333, 599):
        sub, _ = quarter_calendar(dates[:k])
        assert np.array_equal(sub, is_dec[:k])


# ------------------------------------------------------------------------------------------------ features
def _feature_case():
    T, N = 260, 5
    dates = _weekdays(10000, T)
    is_dec, _ = quarter_calendar(dates)
    rows = np.flatnonzero(is_dec)
    dp2, dp, d = int(rows[1]), int(rows[2]), int(rows[3])
    roc = np.zeros((T, N))
    roc[:, 0] = np.where(np.arange(T) % 2 == 0, 0.01, -0.01)
    roc[:, 1:] = 0.001
    close = 10.0 * np.cumprod(1.0 + roc, axis=0)
    close[d - 20:, 2] *= 0.5                         # split-like jump without a return
    close[: d - 30, 4] = np.nan                      # late listing: < 63 sessions of returns
    S = np.full((T, N), 100.0)
    c1, c2 = dp2 + 5, d - 7
    S[c1:c2, 0] = 110.0
    S[c2:, 0] = 121.0
    S[:, 1] = 50.0
    S[: dp2 + 1, 3] = np.nan                         # S missing at the decision date before last
    dv = np.full((T, N), 1e6)
    dv[:, 0] = 2e6
    return _view(T, N, ret_oc=roc, close=close, S=S, dv=dv), is_dec, (dp2, dp, d, c2)


def test_features_on_known_rows():
    v, is_dec, (dp2, dp, d, c2) = _feature_case()
    rows, feats, scored = compute_features(v, is_dec)
    assert rows.tolist() == np.flatnonzero(is_dec).tolist()
    k = int(np.flatnonzero(rows == d)[0])
    dq, dq_prev, demand, age, vol, ret5 = (feats[i, k] for i in range(6))
    assert scored[k].tolist() == [True, True, False, False, False]
    assert dq[0] == pytest.approx(np.log(121 / 110), rel=1e-5)
    assert dq_prev[0] == pytest.approx(np.log(110 / 100), rel=1e-5)
    assert dq[1] == 0.0 and dq_prev[1] == 0.0
    assert demand[0] == pytest.approx(np.log(121 / 110) * 121 * v.close[d, 0] / 2e6, rel=1e-4)
    assert age[0] == d - c2 and age[1] == 252
    lr = np.log1p(v.ret_oc[d - 62:d + 1, 0])
    assert vol[0] == pytest.approx(np.std(lr, ddof=1), rel=1e-4)
    assert vol[1] == pytest.approx(0.0, abs=1e-6)
    assert ret5[0] == pytest.approx(np.prod(1 + v.ret_oc[d - 4:d + 1, 0]) - 1, rel=1e-4)
    # the first two decisions have no Dprev2 / Dprev: nothing scored there
    assert not scored[:2].any()
    for i in range(6):
        assert np.array_equal(np.isfinite(feats[i]), scored)


def test_features_point_in_time():
    v, is_dec, (dp2, dp, d, c2) = _feature_case()
    rows, full, sc_full = compute_features(v, is_dec)
    k = d + 1
    vt = DataView(dates=v.dates[:k], instruments=v.instruments, asset_class=v.asset_class, ret_co=v.ret_co[:k],
                  ret_oc=v.ret_oc[:k], tradable=v.tradable[:k], listed=v.listed[:k], delisting=v.delisting[:k],
                  close=v.close[:k], dollar_volume=v.dollar_volume[:k], universe=v.universe[:k],
                  extras={SH: v.extras[SH][:k]}, series={}, cash_ret=v.cash_ret[:k])
    rows_c, cut, sc_cut = compute_features(vt, is_dec[:k])
    n = rows_c.size
    assert np.array_equal(rows_c, rows[:n]) and np.array_equal(sc_cut, sc_full[:n])
    assert np.allclose(cut, full[:, :n], equal_nan=True)


def test_zscore_flat_feature_is_zero():
    f = np.full((3, 4), np.nan, dtype=np.float32)
    scored = np.zeros((3, 4), bool)
    f[1] = [5, 5, 5, np.nan]
    scored[1] = [True, True, True, False]
    f[2] = [1, 2, 3, 4]
    scored[2] = True
    z = zscore_scored(f, scored)
    assert z[1, :3].tolist() == [0.0, 0.0, 0.0] and np.isnan(z[1, 3])
    assert np.all(np.isfinite(z[2])) and abs(float(z[2].mean())) < 1e-5 and z[2, 3] > z[2, 0]
    assert np.isnan(z[0]).all()


# ------------------------------------------------------------------------------------------------ book
def _book_case(N=60, T=300):
    close = np.tile(np.arange(1, N + 1, dtype=float), (T, 1))
    v = _view(T, N, close=close, S=np.ones((T, N)))
    is_dec, tf = quarter_calendar(v.dates)
    return v, is_dec, tf


def _held(W):
    held = np.zeros(W.shape[1])
    out = []
    for t in range(W.shape[0]):
        held = np.where(np.isfinite(W[t]), W[t], held)
        out.append(held.copy())
    return np.array(out)


def test_core_weights_are_cap_weights():
    v, _, _ = _book_case()
    w = core_weights(v, 10)
    cap = np.arange(1, 61, dtype=float)
    assert np.allclose(w, cap / cap.sum())
    # members with a missing S are left out
    v.extras[SH][10, 0] = np.nan
    w = core_weights(v, 10)
    assert w[0] == 0 and w.sum() == pytest.approx(1.0)


def test_window_switch_hold_and_switch_back():
    v, is_dec, tf = _book_case()
    rows = np.flatnonzero(is_dec)
    d = int(rows[1])
    score = np.full((rows.size, 60), np.nan)
    score[1] = np.arange(60)
    v.universe[d + 2, 59] = False                     # a window member leaves the index inside the window
    W = build_book(v, rows, score, 0.1, is_dec, tf)
    top = np.flatnonzero(W[d] > 0)
    assert top.tolist() == list(range(54, 60))
    assert np.allclose(W[d, top], 1 / 6) and W[d].sum() == pytest.approx(1.0)
    assert np.isnan(W[d + 1]).all() and np.isnan(W[d + 3]).all() and np.isnan(W[d + 4]).all()
    assert W[d + 2, 59] == 0 and np.isnan(np.delete(W[d + 2], 59)).all()
    end = window_end_rows(v.dates, is_dec, tf)[d]
    assert end == d + 5 and v.dates[end] == tf[d]
    assert np.allclose(W[end], core_weights(v, end))
    # before the window the book holds the core, with weekly resets
    H = _held(W)
    assert np.allclose(H[d - 1], core_weights(v, d - 1)) or np.allclose(H[d - 1].sum(), 1.0)
    g = np.abs(H).sum(axis=1)
    assert g.max() <= 1.0 + 1e-9
    # top parameter changes the window
    W2 = build_book(v, rows, score, 0.2, is_dec, tf)
    assert np.flatnonzero(W2[d] > 0).tolist() == list(range(48, 60))


def test_fewer_than_50_scored_keeps_core():
    v, is_dec, tf = _book_case()
    rows = np.flatnonzero(is_dec)
    d = int(rows[1])
    score = np.full((rows.size, 60), np.nan)
    score[1, :49] = np.arange(49)
    W = build_book(v, rows, score, 0.1, is_dec, tf)
    assert not (W[d] > 0.1).any()                      # no 1/n window weights
    H = _held(W)
    assert np.allclose(H[d], H[d - 1]) or np.allclose(W[d], core_weights(v, d))


def test_no_score_means_core_only():
    v, is_dec, tf = _book_case()
    rows = np.flatnonzero(is_dec)
    W = build_book(v, rows, np.full((rows.size, 60), np.nan), 0.1, is_dec, tf)
    H = _held(W)
    assert np.allclose(H.sum(axis=1), 1.0)
    dec = np.flatnonzero(np.isfinite(W).any(axis=1))
    wk = np.diff(v.dates[dec].astype("datetime64[W]").astype(np.int64))
    assert np.all(wk >= 1)                              # resets only once per week


# ------------------------------------------------------------------------------------------------ model
def test_scores_only_on_scored_decision_rows():
    rng = np.random.default_rng(1)
    T, N = 900, 60
    roc = rng.normal(0, 0.01, (T, N))
    S = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, (T, N)) * (rng.random((T, N)) < 0.02), axis=0))
    v = _view(T, N, ret_oc=roc, S=S, dv=np.exp(rng.normal(14, 0.5, (T, N))))
    is_dec, tf = quarter_calendar(v.dates)
    rows, _, scored = compute_features(v, is_dec)
    rows2, sc = model_scores(v, PARAMS, is_dec)
    assert np.array_equal(rows, rows2) and sc.shape == scored.shape
    assert not (np.isfinite(sc) & ~scored).any()
    W = target_weights(v, PARAMS)
    assert W.shape == (T, N)
    H = _held(W)
    assert np.abs(H).sum(axis=1).max() <= 1.0 + 1e-9 and (H >= 0).all()
