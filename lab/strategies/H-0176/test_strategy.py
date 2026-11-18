import numpy as np
import pytest

from strategy import PARAMS, target_weights
from strategy import build_book, compute_features, decision_rows, scored_mask, select_sleeve

EPOCH_2020 = 18262          # days since 1970-01-01 of 2020-01-01 (a Wednesday)


def weekdays(start, n_days):
    d = np.arange(start, start + n_days).astype("datetime64[D]")
    return d[np.is_busday(d)]


# ------------------------------------------------------------------ calendar
def test_decisions_on_last_weekday_of_month():
    d = weekdays(EPOCH_2020, 100)                 # Jan .. early April 2020
    dec = decision_rows(d)
    got = d[dec]
    # Jan 31 (Fri), Feb 28 (Fri, the 29th is a Saturday), Mar 31 (Tue)
    exp = np.array([EPOCH_2020 + 30, EPOCH_2020 + 58, EPOCH_2020 + 90]).astype("datetime64[D]")
    assert np.array_equal(got, exp)


def test_holiday_on_last_weekday_moves_decision_to_next_month_start():
    d = weekdays(EPOCH_2020, 100)
    jan31 = np.array([EPOCH_2020 + 30]).astype("datetime64[D]")[0]
    d2 = d[d != jan31]                            # Jan 31 is a holiday
    dec = decision_rows(d2)
    got = d2[dec]
    feb3 = np.array([EPOCH_2020 + 33]).astype("datetime64[D]")[0]   # first row of February (Monday)
    assert got[0] == feb3
    assert len(got) == 3


def test_decisions_point_in_time_under_truncation():
    d = weekdays(EPOCH_2020, 400)
    full = decision_rows(d)
    for t in (5, 21, 22, 23, 60, 150, len(d) - 1):
        assert np.array_equal(decision_rows(d[: t + 1]), full[: t + 1])


# ------------------------------------------------------------------ sleeve selection
def test_fallback_when_fewer_than_ten_scored():
    N = 12
    score = np.arange(N, dtype=float)
    mask = np.zeros(N, dtype=bool)
    mask[:9] = True
    vol = np.ones(N)
    w, model = select_sleeve(score, mask, vol, 0.2, 10, 11)
    assert not model
    assert w[10] == pytest.approx(0.6) and w[11] == pytest.approx(0.4)
    assert w.sum() == pytest.approx(1.0)
    # no model yet: all scores NaN
    w, model = select_sleeve(np.full(N, np.nan), np.ones(N, bool), vol, 0.2, 10, 11)
    assert not model and w[10] == pytest.approx(0.6)


def test_top_fraction_rounding_and_inverse_vol():
    N = 20
    score = np.arange(N, dtype=float)             # best = column 19
    mask = np.ones(N, dtype=bool)
    vol = np.ones(N)
    vol[19] = 2.0
    w, model = select_sleeve(score, mask, vol, 0.2, 0, 1)
    assert model
    assert set(np.flatnonzero(w)) == {16, 17, 18, 19}        # round(0.2 * 20) = 4
    assert w[19] == pytest.approx(0.5 * w[18])                 # 1 / vol
    assert w.sum() == pytest.approx(1.0)
    w, _ = select_sleeve(score, mask, vol, 0.3, 0, 1)
    assert np.count_nonzero(w) == 6
    w, _ = select_sleeve(score, mask, vol, 0.15, 0, 1)
    assert np.count_nonzero(w) == 3                            # round(3.0)
    mask[:10] = False                                          # 10 scored: 0.15 * 10 = 1.5 -> 2 -> at least 3
    w, _ = select_sleeve(score, mask, vol, 0.15, 0, 1)
    assert set(np.flatnonzero(w)) == {17, 18, 19}
    w, _ = select_sleeve(score, mask, vol, 0.3, 0, 1)
    assert np.count_nonzero(w) == 3


# ------------------------------------------------------------------ two-sleeve book
def test_book_two_sleeves_with_drift():
    T, N = 10, 4
    score = np.full((T, N), np.nan)
    mask = np.zeros((T, N), dtype=bool)
    vol = np.ones((T, N))
    ret_co = np.zeros((T, N))
    tr = np.ones((T, N))
    dec = np.zeros(T, dtype=bool)
    dec[[2, 5, 8]] = True
    # instrument 2 doubles between the open of row 3 and the close of row 5
    tr[4:, 2] = 2.0
    score[2] = [0, 0, 5, 4]; mask[2] = True       # top 0.2 of 4 -> at least 3 names: 2, 3, 0 (stable tie)
    score[5] = [9, 0, 0, 0]; mask[5, 0] = True    # one scored -> 3 names capped at 1
    score[8] = [0, 9, 0, 0]; mask[8, 1] = True
    out = build_book(score, mask, vol, tr, ret_co, dec, 0.2, 0, 1, min_scored=1)
    assert np.all(np.isnan(out[[0, 1, 3, 4, 6, 7, 9]]))
    # first decision: both halves from the same sleeve
    assert np.allclose(out[2], [1 / 3, 0, 1 / 3, 1 / 3])
    # second: old sleeve drifted (instrument 2 doubled -> 2/4, others 1/4), rescaled to half; new sleeve = col 0
    assert np.allclose(out[5], [0.5 * 0.25 + 0.5, 0, 0.5 * 0.5, 0.5 * 0.25])
    # third: sleeve of row 2 closed; sleeve of row 5 (col 0 only) held; new sleeve col 1
    assert np.allclose(out[8], [0.5, 0.5, 0, 0])
    assert np.all(np.nansum(np.abs(out), axis=1) <= 1 + 1e-12)


# ------------------------------------------------------------------ features
def test_features_hand_values_and_warmup():
    T, N = 700, 2
    rng = np.random.default_rng(0)
    ret_co = rng.normal(0, 0.003, (T, N))
    ret_oc = rng.normal(0, 0.01, (T, N))
    close = np.ones((T, N))
    close[:150, 1] = np.nan                         # instrument 1 lists at row 150
    ret_co[:151, 1] = 0.0
    ret_oc[:150, 1] = 0.0
    tr_raw, f, bar, vol63 = compute_features(ret_co, ret_oc, close)
    tr = np.cumprod((1 + ret_co) * (1 + ret_oc), axis=0)
    t = 600
    assert f["r21"][t, 0] == pytest.approx(tr[t, 0] / tr[t - 21, 0] - 1)
    assert f["r12_1"][t, 0] == pytest.approx(tr[t - 21, 0] / tr[t - 252, 0] - 1)
    assert f["dd"][t, 0] == pytest.approx(tr[t, 0] / tr[t - 251: t + 1, 0].max() - 1)
    lr = np.log((1 + ret_co) * (1 + ret_oc))
    assert vol63[t, 0] == pytest.approx(np.std(lr[t - 62: t + 1, 0], ddof=1))
    exp_vr = np.log(np.std(lr[t - 20: t + 1, 0], ddof=1) / np.std(lr[t - 125: t + 1, 0], ddof=1))
    assert f["vr"][t, 0] == pytest.approx(exp_vr)
    exp_vlev = np.log(np.std(lr[t - 62: t + 1, 0], ddof=1) / np.std(lr[t - 503: t + 1, 0], ddof=1))
    assert f["vlev"][t, 0] == pytest.approx(exp_vlev)
    # vlev needs the 504 window (row 503 is the first) for the full-history ETF
    assert np.isnan(f["vlev"][502, 0]) and np.isfinite(f["vlev"][503, 0])
    # late lister: first valid return at row 151; 378 valid returns are reached at row 528
    assert np.isnan(f["vlev"][527, 1]) and np.isfinite(f["vlev"][528, 1])
    m = scored_mask(f, bar)
    assert not m[:528, 1].any() and m[528, 1]
    # a missing bar on t -> not scored on t
    close2 = close.copy()
    close2[650, 0] = np.nan
    _, f2, bar2, _ = compute_features(ret_co, ret_oc, close2)
    m2 = scored_mask(f2, bar2)
    assert not m2[650, 0]


def test_features_point_in_time():
    T, N = 640, 3
    rng = np.random.default_rng(1)
    ret_co = rng.normal(0, 0.003, (T, N))
    ret_oc = rng.normal(0, 0.01, (T, N))
    close = np.ones((T, N))
    _, f_full, _, v_full = compute_features(ret_co, ret_oc, close)
    for t in (300, 520, 600):
        rc2, ro2 = ret_co.copy(), ret_oc.copy()
        rc2[t + 1:] = rng.normal(0, 0.05, (T - t - 1, N))
        ro2[t + 1:] = rng.normal(0, 0.05, (T - t - 1, N))
        _, f_p, _, v_p = compute_features(rc2, ro2, close)
        for k in f_full:
            assert np.allclose(f_full[k][: t + 1], f_p[k][: t + 1], equal_nan=True)
        assert np.allclose(v_full[: t + 1], v_p[: t + 1], equal_nan=True)


def test_params_present():
    assert PARAMS == {"n_estimators": 100, "top": 0.2}
    assert callable(target_weights)
