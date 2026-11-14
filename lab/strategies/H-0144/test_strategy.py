import math

import numpy as np
import pytest

from strategy import (MIN_SCORED, N_SLEEVES, PARAMS, raw_features, sleeve_book, target_weights, top_set,
                      _window_max, _window_sum)


def test_params_match_card():
    assert PARAMS == {"n_estimators": 100, "top": 0.2}


# ------------------------------------------------------------------ rolling helpers
def test_window_sum_and_max_incomplete_windows_are_nan():
    x = np.arange(10, dtype=float).reshape(10, 1)
    s = _window_sum(x, 3)
    assert np.all(np.isnan(s[:2]))
    assert s[2, 0] == 0 + 1 + 2 and s[9, 0] == 7 + 8 + 9
    m = _window_max(x[::-1].copy(), 4)
    assert np.all(np.isnan(m[:3]))
    assert m[3, 0] == 9 and m[9, 0] == 3
    # views shorter than the window do not crash
    assert np.all(np.isnan(_window_sum(x[:2], 5)))
    assert np.all(np.isnan(_window_max(x[:2], 5)))


# ------------------------------------------------------------------ features
def _flat_market(T=400, N=2, r=0.01):
    ret_co = np.zeros((T, N))
    ret_oc = np.full((T, N), r)
    close = np.cumprod(1 + ret_oc, axis=0)
    qv = np.ones((T, N))
    listed = np.ones((T, N), dtype=bool)
    return ret_co, ret_oc, close, qv, listed


def test_features_by_hand():
    ret_co, ret_oc, close, qv, listed = _flat_market()
    qv[:, 1] = np.where(np.arange(400) >= 370, 4.0, 1.0)      # volume up 4x in the last 30 rows
    close[300:, 1] = close[300:, 1] * 0.1                      # price level drop (stage), returns unchanged
    feats, nvalid = raw_features(ret_co, ret_oc, close, qv, listed)
    trend, recent, stage, vtrend, vol90 = feats
    t = 399
    lr = math.log(1.01)
    assert trend[t, 0] == pytest.approx(150 * lr)
    assert recent[t, 0] == pytest.approx(30 * lr)
    assert stage[t, 0] == pytest.approx(0.0)
    # peak of t-179..t is the close of row 299 (before the drop); close_399 = 0.1 x close_299 x 1.01^100
    assert stage[t, 1] == pytest.approx(math.log(0.1) + 100 * lr)
    assert vtrend[t, 0] == pytest.approx(0.0)
    assert vtrend[t, 1] == pytest.approx(math.log(4.0 / ((150 + 30 * 4) / 180)))
    assert vol90[t, 0] == pytest.approx(0.0, abs=1e-6)
    assert nvalid[t, 0] == 180
    # nothing before the 180-row formation window is complete
    assert np.all(np.isnan(trend[:180])) and np.all(np.isnan(stage[:179]))
    assert np.isfinite(trend[180, 0])


def test_late_listing_and_valid_bar_count():
    ret_co, ret_oc, close, qv, listed = _flat_market()
    listed[:260, 1] = False
    close[:260, 1] = np.nan
    qv[:260, 1] = np.nan
    ret_oc[:260, 1] = 0.0
    feats, nvalid = raw_features(ret_co, ret_oc, close, qv, listed)
    # at t = 399 the coin has 140 valid bars (< 150): unscored by the strategy rule
    assert nvalid[399, 1] == 140
    assert nvalid[399, 0] == 180
    # 150 valid bars at t = 409
    ret_co2, ret_oc2, close2, qv2, listed2 = _flat_market(T=410)
    listed2[:260, 1] = False
    close2[:260, 1] = np.nan
    _, nv2 = raw_features(ret_co2, ret_oc2, close2, qv2, listed2)
    assert nv2[409, 1] == 150


def test_features_point_in_time():
    rng = np.random.default_rng(0)
    T, N = 420, 3
    ret_co = rng.normal(0, 0.01, (T, N))
    ret_oc = rng.normal(0, 0.03, (T, N))
    close = np.cumprod((1 + ret_co) * (1 + ret_oc), axis=0)
    qv = rng.uniform(1, 2, (T, N))
    listed = np.ones((T, N), dtype=bool)
    full, nv = raw_features(ret_co, ret_oc, close, qv, listed)
    cut = 300
    part, nvp = raw_features(ret_co[:cut], ret_oc[:cut], close[:cut], qv[:cut], listed[:cut])
    for a, b in zip(full, part):
        np.testing.assert_allclose(a[:cut], b, equal_nan=True, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(nv[:cut], nvp)
    # vol90 equals the sample std of the last 90 daily log total returns
    lr = np.log1p(ret_co) + np.log1p(ret_oc)
    assert full[4][399, 2] == pytest.approx(np.std(lr[310:400, 2], ddof=1), rel=1e-9)
    # trend: t-180 -> t-30
    assert full[0][399, 1] == pytest.approx(lr[220:370, 1].sum(), rel=1e-9)


# ------------------------------------------------------------------ selection
def test_top_set_rounding_and_minimum():
    s = np.full(40, np.nan)
    assert top_set(s, 0.2) is None
    s[:14] = np.arange(14)
    assert top_set(s, 0.2) is None                     # 14 < 15 scored
    s[:15] = np.arange(15)
    assert list(top_set(s, 0.2)) == [14, 13, 12]       # floor(3.0) = 3
    assert list(top_set(s, 0.1)) == [14, 13, 12]       # floor(1.5) -> at least 3
    s[:29] = np.arange(29)
    assert len(top_set(s, 0.2)) == 5                   # floor(5.8)
    assert len(top_set(s, 0.3)) == 8                   # floor(8.7)
    assert MIN_SCORED == 15


# ------------------------------------------------------------------ sleeve book
def _scores(T, N, n_scored, best):
    s = np.full((T, N), np.nan)
    s[:, :n_scored] = 0.0
    s[:, best] = 1.0 + np.arange(len(best))[::-1]
    return s


def test_sleeve_book_by_hand():
    T, N, btc = 30, 21, 20
    member = np.ones((T, N), dtype=bool)
    listed = np.ones((T, N), dtype=bool)
    dec = np.arange(0, 30, 3)                           # 10 decision rows
    score = np.full((T, N), np.nan)
    score[6:] = _scores(T, N, 20, [0, 1, 2, 3])[6:]     # 20 scored from row 6, top 0.2 -> 4 picks: 0..3
    w = sleeve_book(score, member, listed, dec, btc, 0.2)
    assert np.all(np.isnan(w[1]))                       # NaN off decision rows
    assert w[0, btc] == pytest.approx(1.0)              # no fit yet: all BTC
    assert w[3, btc] == pytest.approx(1.0)
    # row 6: one sleeve of 4 coins
    assert w[6, 0] == pytest.approx(1 / 8 / 4) and w[6, btc] == pytest.approx(7 / 8)
    # row 9: two sleeves
    assert w[9, 0] == pytest.approx(2 / 8 / 4) and w[9, btc] == pytest.approx(6 / 8)
    # after 8 sleeves of picks (decision index 2..9 -> row 27), BTC is gone
    assert w[27, btc] == pytest.approx(0.0) and w[27, 0] == pytest.approx(0.25)
    for t in dec:
        assert np.nansum(np.abs(w[t])) <= 1.0 + 1e-12
        assert np.nansum(w[t]) == pytest.approx(1.0)
        assert np.nanmin(w[t]) >= 0


def test_sleeve_rolls_off_after_eight_weeks():
    T, N, btc = 40, 21, 20
    member = np.ones((T, N), dtype=bool)
    listed = np.ones((T, N), dtype=bool)
    dec = np.arange(0, 40, 2)
    score = np.full((T, N), np.nan)
    score[0] = _scores(1, N, 20, [0, 1, 2, 3])[0]       # only the first decision has scores
    w = sleeve_book(score, member, listed, dec, btc, 0.2)
    for k in range(N_SLEEVES):
        assert w[dec[k], 0] == pytest.approx(1 / 32)
    assert w[dec[8], 0] == pytest.approx(0.0)           # sleeve 0 replaced at decision 8
    assert w[dec[8], btc] == pytest.approx(1.0)


def test_leaver_moves_to_btc_and_stays():
    T, N, btc = 30, 21, 20
    member = np.ones((T, N), dtype=bool)
    listed = np.ones((T, N), dtype=bool)
    dec = np.arange(0, 30, 3)
    score = np.full((T, N), np.nan)
    score[0] = _scores(1, N, 20, [0, 1, 2, 3])[0]
    member[4:7, 0] = False                              # coin 0 leaves at row 4, returns at row 7
    w = sleeve_book(score, member, listed, dec, btc, 0.2)
    assert w[0, 0] == pytest.approx(1 / 32)
    assert w[3, 0] == pytest.approx(1 / 32)
    assert w[6, 0] == pytest.approx(0.0)                # moved to BTC at the next decision row
    assert w[6, btc] == pytest.approx(7 / 8 + 1 / 32)
    assert w[9, 0] == pytest.approx(0.0)                # stays in BTC though coin 0 is a member again
    assert w[9, 1] == pytest.approx(1 / 32)
    # delisted (not listed) also moves
    listed2 = listed.copy()
    listed2[5:, 1] = False
    w2 = sleeve_book(score, member, listed2, dec, btc, 0.2)
    assert w2[6, 1] == pytest.approx(0.0)


def test_btc_filler_in_cash_while_btc_not_member():
    T, N, btc = 12, 21, 20
    member = np.ones((T, N), dtype=bool)
    member[:4, btc] = False
    listed = np.ones((T, N), dtype=bool)
    dec = np.array([0, 3, 6])
    score = np.full((T, N), np.nan)
    w = sleeve_book(score, member, listed, dec, btc, 0.2)
    assert np.all(w[0] == 0.0) and np.all(w[3] == 0.0)
    assert w[6, btc] == pytest.approx(1.0)


def test_top_param_changes_book():
    T, N, btc = 12, 31, 30
    member = np.ones((T, N), dtype=bool)
    listed = np.ones((T, N), dtype=bool)
    dec = np.array([0, 6])
    score = np.full((T, N), np.nan)
    score[:, :30] = np.arange(30)[None, :]
    a = sleeve_book(score, member, listed, dec, btc, 0.1)
    b = sleeve_book(score, member, listed, dec, btc, 0.3)
    assert (a[0] > 0).sum() == 1 + 3 and (b[0] > 0).sum() == 1 + 9


# ------------------------------------------------------------------ end to end on the synthetic market
def test_target_weights_runs_on_synthetic():
    from lab.framework.data import synthetic
    data = synthetic()
    if data.universe is None or "BTCUSDT" not in data.instruments:
        pytest.skip("synthetic market has no crypto_top_n universe / BTCUSDT")
    w = target_weights(data, PARAMS)
    assert w.shape == (len(data.dates), len(data.instruments))
    rows = np.isfinite(w).any(axis=1)
    assert np.all(np.nansum(np.abs(w[rows]), axis=1) <= 1.0 + 1e-9)
    assert np.nanmin(w) >= 0
