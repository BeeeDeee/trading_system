import math

import numpy as np
import pytest

from lab.framework.data import DataView, synthetic
from lab.framework.ml import cs_zscore
import strategy as S
from strategy import PARAMS, target_weights

MONDAY = 4 + 7 * 2000          # day number (days since 1970-01-01) of a Monday


def weekdays(T):
    base = np.array([MONDAY], dtype=np.int64).astype("datetime64[D]")
    return np.busday_offset(base[0], np.arange(T), roll="forward")


# ----------------------------------------------------------------------------- small helpers
def test_window_sum_counts_finite_values():
    x = np.array([[1.0], [2.0], [np.nan], [4.0]])
    s, c = S.window_sum(x, 2)
    assert np.allclose(s[:, 0], [1.0, 3.0, 2.0, 4.0])
    assert list(c[:, 0]) == [1, 2, 1, 1]


def test_rolling_sigma_window_and_minimum():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 0.02, size=(120, 1))
    e[:10] = np.nan                      # first valid at row 10
    sig = S.rolling_sigma(e)
    assert np.isnan(sig[48, 0])          # 39 valid
    assert np.isfinite(sig[49, 0])       # 40 valid
    assert sig[49, 0] == pytest.approx(np.std(e[10:50, 0], ddof=1), rel=1e-9)
    assert sig[110, 0] == pytest.approx(np.std(e[48:111, 0], ddof=1), rel=1e-9)


def test_residual_returns_member_mean_and_unlisted():
    ret_co = np.array([[0.0, 0.0, 0.0]])
    ret_oc = np.array([[0.01, 0.03, 0.50]])
    listed = np.array([[True, True, False]])
    close = np.array([[10.0, 20.0, np.nan]])
    uni = np.array([[True, True, False]])
    e = S.residual_returns(ret_co, ret_oc, listed, close, uni)
    assert e[0, 0] == pytest.approx(-0.01)
    assert e[0, 1] == pytest.approx(0.01)
    assert np.isnan(e[0, 2])
    # a non-member with a price gets a residual against the members' mean
    listed2 = np.array([[True, True, True]])
    close2 = np.array([[10.0, 20.0, 5.0]])
    e2 = S.residual_returns(ret_co, ret_oc, listed2, close2, uni)
    assert e2[0, 2] == pytest.approx(0.48)


def test_release_cycle_f3_cap_calendar_and_quiet_week():
    T = 400
    rel = np.zeros((T, 1), dtype=bool)
    rel[2, 0] = True
    day_num = np.arange(T, dtype=np.int64) * 2          # 2 calendar days per session
    f3, quiet = S.release_cycle(rel, day_num)
    assert np.isnan(f3[1, 0]) and quiet[1, 0]           # no release yet
    assert f3[2, 0] == 0 and not quiet[2, 0]
    assert f3[6, 0] == 4 and not quiet[6, 0]            # release in t-4..t
    assert f3[7, 0] == 5 and quiet[7, 0]
    assert f3[2 + 150, 0] == 126                        # capped
    assert f3[2 + 200, 0] == 126                        # 400 calendar days: still eligible
    assert np.isnan(f3[2 + 201, 0])                     # 402 calendar days: not eligible


def test_informativeness_last_four_min_two_and_window():
    T = 700
    e = np.zeros((T, 2))
    sigma = np.ones((T, 2))
    rel = np.zeros((T, 2), dtype=bool)
    for k, r in enumerate([10, 20, 30, 40, 50]):
        rel[r, 0] = True
        e[r, 0] = k + 1.0
    f4 = S.informativeness(rel, e, sigma)
    assert np.isnan(f4[15, 0])                          # one release only
    assert f4[20, 0] == pytest.approx(1.5)
    assert f4[45, 0] == pytest.approx(2.5)              # 1..4
    assert f4[55, 0] == pytest.approx(3.5)              # last four: 2..5
    # 504-row window on the second column
    rel[10, 1] = True
    rel[20, 1] = True
    e[10, 1] = 1.0
    e[20, 1] = 3.0
    f4 = S.informativeness(rel, e, sigma)
    assert f4[513, 1] == pytest.approx(2.0)             # rows 10..513 contain both
    assert np.isnan(f4[514, 1])                         # row 10 dropped
    # sigma of the session before the release is used
    sigma2 = np.ones((T, 2))
    sigma2[19, 1] = 0.5
    f4b = S.informativeness(rel, e, sigma2)
    assert f4b[30, 1] == pytest.approx((1.0 + 6.0) / 2)


def test_rd_intensity_rules():
    rnd = np.array([[5.0, np.nan, 30.0, 1.0, 2.0]])
    rev = np.array([[50.0, 40.0, 20.0, -3.0, np.nan]])
    uni = np.ones((1, 5), dtype=bool)
    f5 = S.rd_intensity(rnd, rev, uni)
    assert f5[0, 0] == pytest.approx(0.1)
    assert f5[0, 1] == 0.0                              # rnd missing, revenue > 0
    assert f5[0, 2] == 1.0                              # capped
    med = np.median([0.1, 0.0, 1.0])
    assert f5[0, 3] == pytest.approx(med)               # revenue <= 0 -> date median
    assert f5[0, 4] == pytest.approx(med)               # revenue missing -> date median


def test_abnormal_dollar_volume_windows():
    T = 80
    dv = np.ones((T, 2))
    dv[75:80, :] = 2.0                                  # t-4..t at t = 79
    dv[12:36, 1] = np.nan                               # 24 NaN in t-67..t-5 -> 39 valid
    cand = np.zeros((T, 2), dtype=bool)
    cand[79, :] = True
    cand[43, 0] = True                                  # rows -24..38: 39 valid
    cand[44, 0] = True                                  # rows -23..39: 40 valid
    f2 = S.abnormal_dollar_volume(dv, cand)
    assert f2[79, 0] == pytest.approx(math.log(2.0))
    assert np.isnan(f2[79, 1])
    assert np.isnan(f2[43, 0])
    assert f2[44, 0] == pytest.approx(0.0)
    assert np.isnan(f2[78, 0])                          # not a candidate
    # median, not mean
    dv2 = np.ones((T, 1))
    dv2[12:20, 0] = 100.0
    cand2 = np.zeros((T, 1), dtype=bool)
    cand2[79, 0] = True
    assert S.abnormal_dollar_volume(dv2, cand2)[79, 0] == pytest.approx(0.0)


def test_decision_days_last_weekday_of_week():
    d = weekdays(15)                                    # Mon..Fri x 3
    keep = np.ones(15, dtype=bool)
    keep[9] = False                                     # Friday of week 2 is a holiday
    dec = S.decision_days(d[keep])
    assert list(np.nonzero(dec)[0]) == [4, 13]          # Fri wk1, Fri wk3; Thu before the holiday is missed


def test_leg_masks_and_book():
    T, N = 3, 40
    score = np.full((T, N), np.nan)
    score[0, :30] = np.arange(30.0)
    score[1, :19] = np.arange(19.0)                     # fewer than 20 rows: flat
    score[2, :30] = np.arange(30.0)
    dec = np.array([True, True, False])
    w = S.make_book(score, dec, 0.1)
    assert np.allclose(w[0, 27:30], 0.5 / 3) and np.allclose(w[0, :3], -0.5 / 3)
    assert np.allclose(w[0, 3:27], 0.0) and np.allclose(w[0, 30:], 0.0)
    assert np.abs(w[0]).sum() == pytest.approx(1.0) and w[0].sum() == pytest.approx(0.0)
    assert np.allclose(w[1], 0.0)
    assert np.all(np.isnan(w[2]))
    w2 = S.make_book(score, dec, 0.2)
    assert np.allclose(w2[0, 24:30], 0.5 / 6)
    s20 = np.full((1, N), np.nan)
    s20[0, :20] = np.arange(20.0)
    w3 = S.make_book(s20, np.array([True]), 0.05)       # at least one name per leg
    assert w3[0, 19] == pytest.approx(0.5) and w3[0, 0] == pytest.approx(-0.5)


def test_continuation_events_new_entries_and_sign():
    T, N = 6, 4
    long_m = np.zeros((T, N), dtype=bool)
    short_m = np.zeros((T, N), dtype=bool)
    dec = np.zeros(T, dtype=bool)
    dec[[0, 5]] = True
    long_m[0, 0] = True
    short_m[0, 1] = True
    long_m[5, [0, 2]] = True                            # 0 stays, 2 new
    short_m[5, 3] = True                                # 3 new
    move = np.zeros((T, N))
    move[0] = [0.1, -0.1, 0.0, 0.0]
    move[5] = [0.1, 0.0, -0.2, -0.3]                    # 2: long after a fall = reversal call
    member = np.ones((T, N), dtype=bool)
    mask, side = S.continuation_events(long_m, short_m, dec, move, member)
    assert mask[0, 0] and side[0, 0] == 1.0
    assert mask[0, 1] and side[0, 1] == -1.0
    assert not mask[5, 0] and not mask[5, 2]
    assert mask[5, 3] and side[5, 3] == -1.0
    assert mask.sum() == 3


def test_framework_cs_zscore_uses_the_mask():
    x = np.array([[1.0, 2.0, 3.0, 50.0]], dtype=np.float32)
    m = np.array([[True, True, True, False]])
    z = np.asarray(cs_zscore(x, m), dtype=np.float64)
    assert np.isnan(z[0, 3])
    assert z[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert z[0, 0] < 0 < z[0, 2]


# ----------------------------------------------------------------------------- hand-made market
def make_market(T=700, N=120, seed=3, late=5):
    rng = np.random.default_rng(seed)
    dates = weekdays(T)
    ret_co = rng.normal(0, 0.005, (T, N))
    ret_oc = rng.normal(0, 0.015, (T, N))
    listed = np.ones((T, N), dtype=bool)
    listed[:300, :late] = False                         # late listings
    ret_co[~listed] = 0.0
    ret_oc[~listed] = 0.0
    close = np.where(listed, 20.0, np.nan)
    dv = np.where(listed, np.exp(rng.normal(16, 0.5, (T, N))), np.nan)
    uni = listed.copy()
    uni[:, N - 10:] = False                             # never members
    ea = np.zeros((T, N))
    for j in range(N):                                  # quarterly releases, staggered
        ea[(j % 63)::63, j] = 1.0
    ea[:100, : N // 4] = np.nan                         # EDGAR history starts late for some
    rnd = np.where(rng.random((T, N)) < 0.3, np.nan, rng.random((T, N)) * 5.0)
    rev = rng.random((T, N)) * 50.0 - 2.0
    data = DataView(
        dates=dates, instruments=tuple(f"E{j}" for j in range(N)), asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=ret_co, ret_oc=ret_oc, tradable=listed.copy(), listed=listed, delisting=np.zeros((T, N), dtype=bool),
        close=close, dollar_volume=dv, universe=uni,
        extras={"ea_reaction_day": ea, "sf1_art_rnd": rnd, "sf1_art_revenue": rev, "liq_rank": np.ones((T, N))},
        series={}, cash_ret=np.zeros(T))
    return data


def features_of(data):
    ex = data.extras
    return S.build_features(data.ret_co, data.ret_oc, data.listed, data.close, data.dollar_volume,
                            data.universe, ex["ea_reaction_day"], ex["sf1_art_rnd"], ex["sf1_art_revenue"],
                            data.dates.astype(np.int64))


def test_features_eligibility_rules():
    data = make_market()
    feats, elig, move = features_of(data)
    uni = data.universe
    assert elig.sum() > 0
    assert not (elig & ~uni).any()                      # members only
    assert np.all(np.abs(feats[0][elig]) >= 1.0)
    for f in feats:
        assert np.all(np.isfinite(f[elig]))
    rel = data.extras["ea_reaction_day"] == 1.0
    T = rel.shape[0]
    for t in range(4, T):
        recent = rel[t - 4:t + 1].any(axis=0)
        assert not (elig[t] & recent).any()             # earnings weeks excluded
    assert not elig[:300, :5].any()                     # not yet listed: never eligible
    assert np.allclose(np.sign(move[elig]), np.sign(feats[0][elig]))


def test_features_point_in_time_truncation():
    data = make_market(T=650)
    feats, elig, move = features_of(data)
    for cut in (120, 333, 500):
        ex = {k: v[:cut] for k, v in data.extras.items()}
        f2, e2, m2 = S.build_features(data.ret_co[:cut], data.ret_oc[:cut], data.listed[:cut], data.close[:cut],
                                      data.dollar_volume[:cut], data.universe[:cut], ex["ea_reaction_day"],
                                      ex["sf1_art_rnd"], ex["sf1_art_revenue"], data.dates[:cut].astype(np.int64))
        assert np.array_equal(e2, elig[:cut])
        for a, b in zip(feats, f2):
            assert np.allclose(a[:cut], b, equal_nan=True)


def test_missing_inputs_fail_loudly():
    v = synthetic()
    with pytest.raises((KeyError, ValueError, TypeError)):
        target_weights(v, PARAMS)
    data = make_market(T=200, N=30)
    del data.extras["sf1_art_rnd"]
    with pytest.raises(KeyError):
        target_weights(data, PARAMS)


def test_target_weights_end_to_end():
    data = make_market()
    w = target_weights(data, PARAMS)
    T, N = w.shape
    assert (T, N) == data.ret_co.shape
    dec = S.decision_days(data.dates)
    assert np.all(np.isnan(w[~dec]))
    wd = w[dec]
    assert np.all(np.isfinite(wd))
    gross = np.abs(wd).sum(axis=1)
    assert np.all(gross <= 1.0 + 1e-9)
    active = gross > 0
    assert active.any()
    assert np.allclose(wd[active].sum(axis=1), 0.0)    # dollar-neutral
    assert np.allclose(np.where(wd > 0, wd, 0).sum(axis=1)[active], 0.5)
    held = np.abs(np.nan_to_num(w)) > 0
    assert not (held & ~data.universe).any()            # only members
    first = np.nonzero(held.any(axis=1))[0][0]
    assert first > 100                                  # nothing before features and the first fit exist
    # the continuation events (as in diagnostic.py) sit on the strategy's legs
    score, elig, move = S.model_scores(data, PARAMS)
    lm, sm = S.leg_masks(score, dec, PARAMS["top"])
    member = data.universe & data.tradable
    mask, side = S.continuation_events(lm, sm, dec, move, member)
    assert mask.sum() > 0
    assert mask.shape == (T, N)
    assert np.all((side[mask] == 1.0) | (side[mask] == -1.0))
    wn = np.nan_to_num(w)
    assert np.all(wn[mask & (side > 0)] > 0)
    assert np.all(wn[mask & (side < 0)] < 0)
    # top changes the book
    w2 = target_weights(data, dict(PARAMS, top=0.2))
    assert not np.allclose(np.nan_to_num(w), np.nan_to_num(w2))
