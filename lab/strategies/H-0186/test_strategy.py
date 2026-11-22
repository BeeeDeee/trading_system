import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

import strategy as S
from strategy import PARAMS, target_weights


@dataclass
class FakeView:
    """Duck-typed stand-in for DataView with only the fields the feature code reads."""
    close: Any
    listed: Any
    ret_co: Any
    ret_oc: Any
    dollar_volume: Any
    extras: dict = field(default_factory=dict)
    universe: Any = None
    tradable: Any = None


def make_view(T=80, N=4, seed=0):
    rng = np.random.default_rng(seed)
    ret_oc = rng.normal(0, 0.03, (T, N))
    ret_co = np.zeros((T, N))
    close = 100 * np.cumprod(1 + ret_oc, axis=0)
    dv = rng.uniform(1e6, 2e6, (T, N))
    fund = rng.normal(0, 1e-4, (T, N))
    listed = np.ones((T, N), bool)
    return FakeView(close=close, listed=listed, ret_co=ret_co, ret_oc=ret_oc, dollar_volume=dv,
                    extras={"funding": fund}, tradable=np.ones((T, N), bool))


# ------------------------------------------------------------------ helpers
def test_params_match_card():
    assert PARAMS == {"horizon": 5, "leg_frac": 0.3}


def test_rolling_sum_trailing_and_short_views():
    x = np.arange(1, 6, dtype=float)[:, None]
    out = S._rolling_sum(x, 3)
    assert out[:, 0].tolist() == [1, 3, 6, 9, 12]
    # window longer than the view (G0 cuts can be short)
    out = S._rolling_sum(x[:2], 60)
    assert out[:, 0].tolist() == [1, 3]


def test_lag():
    x = np.arange(4, dtype=float)[:, None]
    assert np.isnan(S._lag(x, 1)[0, 0]) and S._lag(x, 1)[1:, 0].tolist() == [0, 1, 2]
    assert S._lag(x, 10, fill=0.0)[:, 0].tolist() == [0, 0, 0, 0]


# ------------------------------------------------------------------ features
def test_feature_windows_by_hand():
    v = make_view(T=80, N=4)
    feats, elig = S.raw_features(v)
    f1, f2, f3, f4, f5, f6 = feats
    t, j = 70, 1
    r = v.ret_oc[:, j]
    assert math.isclose(f1[t, j], r[t], rel_tol=1e-9)
    assert math.isclose(f2[t, j], np.prod(1 + r[t - 6:t]) - 1, rel_tol=1e-9)        # rows t-6..t-1
    assert math.isclose(f3[t, j], np.prod(1 + r[t - 27:t - 6]) - 1, rel_tol=1e-9)   # rows t-27..t-7
    # F1 + F2 + F3 compound to the 28-day return close(t-28) -> close(t)
    tot = (1 + f1[t, j]) * (1 + f2[t, j]) * (1 + f3[t, j]) - 1
    assert math.isclose(tot, v.close[t, j] / v.close[t - 28, j] - 1, rel_tol=1e-9)
    assert math.isclose(f4[t, j], np.std(r[t - 27:t + 1], ddof=1), rel_tol=1e-7)
    dv = v.dollar_volume[:, j]
    assert math.isclose(f5[t, j], math.log(dv[t - 6:t + 1].mean() / dv[t - 59:t + 1].mean()), rel_tol=1e-7)
    assert math.isclose(f6[t, j], v.extras["funding"][t - 6:t + 1, j].sum(), rel_tol=1e-7, abs_tol=1e-15)
    assert elig[t].all()


def test_eligibility_warmup_rows():
    v = make_view(T=80, N=3)
    _, elig = S.raw_features(v)
    # first valid return is row 1; 45 valid volume days are reached at row 44
    assert not elig[:44].any()
    assert elig[44:].all()


def test_missing_days_skipped_and_thresholds():
    v = make_view(T=80, N=3)
    t = 75
    # coin 0: 8 missing prices in t-27..t -> at most 19 valid returns (< 21): not scored
    v.close[t - 20:t - 12, 0] = np.nan
    v.listed[t - 20:t - 12, 0] = False
    # coin 1: 3 missing funding days of the last 7 -> 4 < 5: not scored
    v.extras["funding"][t - 2:t + 1, 1] = np.nan
    # coin 2: 2 missing funding days -> still scored, sum over the 5 valid days
    v.extras["funding"][t - 1:t + 1, 2] = np.nan
    feats, elig = S.raw_features(v)
    assert not elig[t, 0]
    assert not elig[t, 1]
    assert elig[t, 2]
    assert math.isclose(feats[5][t, 2], v.extras["funding"][t - 6:t - 1, 2].sum(), rel_tol=1e-7, abs_tol=1e-15)
    assert np.isnan(feats[0][t, 1])          # unscored -> NaN, never zero


def test_too_few_volume_days():
    v = make_view(T=80, N=2)
    t = 79
    v.dollar_volume[t - 59:t - 43, 0] = np.nan          # 16 missing -> 44 valid < 45
    _, elig = S.raw_features(v)
    assert not elig[t, 0] and elig[t, 1]


def test_late_listing_not_scored_before_history():
    v = make_view(T=120, N=2)
    v.close[:50, 1] = np.nan
    v.listed[:50, 1] = False
    v.dollar_volume[:50, 1] = np.nan
    v.extras["funding"][:50, 1] = np.nan
    _, elig = S.raw_features(v)
    assert not elig[:50 + 44, 1].any()
    assert elig[50 + 45:, 1].all()


def test_fallback_universe_top_n():
    v = make_view(T=40, N=5)
    v.dollar_volume[:] = np.array([1.0, 5.0, 3.0, 4.0, 2.0])[None, :]
    u = S.fallback_universe(v, n=2, window=30)
    assert u[10].tolist() == [False, True, False, True, False]
    # a perp without a price today is not a member
    v.close[20, 1] = np.nan
    u = S.fallback_universe(v, n=2, window=30)
    assert u[20].tolist() == [False, False, True, True, False]


def test_fallback_universe_point_in_time():
    v = make_view(T=60, N=6)
    full = S.fallback_universe(v)
    v2 = FakeView(close=v.close[:30], listed=v.listed[:30], ret_co=v.ret_co[:30], ret_oc=v.ret_oc[:30],
                  dollar_volume=v.dollar_volume[:30], extras={"funding": v.extras["funding"][:30]})
    assert (S.fallback_universe(v2) == full[:30]).all()


# ------------------------------------------------------------------ book
def _score_row(n_scored, N):
    s = np.full(N, np.nan)
    s[:n_scored] = np.arange(n_scored, dtype=float)     # column i has score i
    return s


def test_leg_masks_counts_and_order():
    N = 25
    score = np.vstack([_score_row(20, N), _score_row(14, N), _score_row(15, N)])
    lo, sh, k = S.leg_masks(score, 0.3)
    # 20 scored: floor(6) = 6 per leg; long the highest scores 14..19, short 0..5
    assert lo[0].sum() == 6 and sh[0].sum() == 6
    assert set(np.where(lo[0])[0]) == set(range(14, 20))
    assert set(np.where(sh[0])[0]) == set(range(0, 6))
    # 14 scored: below 15 -> no cohort
    assert not lo[1].any() and not sh[1].any()
    # 15 scored at 0.2: floor(3) = 3; at 0.3 floor(4.5) = 4
    assert lo[2].sum() == 4
    lo2, sh2, _ = S.leg_masks(score, 0.2)
    assert lo2[2].sum() == 3 and sh2[2].sum() == 3
    assert not (lo & sh).any()


def test_leg_minimum_three_and_ties():
    N = 16
    score = np.zeros((1, N))                        # all tied, 16 scored
    lo, sh, k = S.leg_masks(score, 0.1)             # floor(1.6) = 1 -> at least 3
    assert lo.sum() == 3 and sh.sum() == 3 and not (lo & sh).any()


def test_cohort_weights_dollar_neutral_gross():
    N = 30
    score = np.vstack([_score_row(30, N)] * 3)
    for h in (3, 5, 7):
        c = S.cohort_weights(score, h, 0.3)
        assert np.allclose(c.sum(axis=1), 0.0)
        assert np.allclose(np.abs(c).sum(axis=1), 1.0 / h)
        assert np.allclose(c[0][c[0] > 0], 0.5 / h / 9)


def test_book_holds_cohort_for_horizon_and_exits_universe():
    T, N, h = 12, 20, 3
    score = np.full((T, N), np.nan)
    score[2] = np.arange(N, dtype=float)            # one cohort formed on row 2
    member = np.ones((T, N), bool)
    member[3, 19] = False                           # a long name leaves the universe on row 3
    w = S.book(S.cohort_weights(score, h, 0.3), member, h)
    assert np.abs(w[:2]).sum() == 0
    assert np.allclose(np.abs(w[2]).sum(), 1.0 / h)
    assert w[3, 19] == 0.0 and w[4, 19] > 0         # closed while out; slice not re-filled elsewhere
    assert np.allclose(np.abs(w[3]).sum(), 1.0 / h - 0.5 / h / 6)
    assert np.allclose(np.abs(w[4]).sum(), 1.0 / h)
    assert np.abs(w[5:]).sum() == 0                 # rows 2..4 live, closed at the open after row 4


def test_book_overlapping_cohorts_gross_le_one():
    T, N = 30, 40
    rng = np.random.default_rng(1)
    score = rng.normal(size=(T, N))
    member = np.ones((T, N), bool)
    for h in (3, 5, 7):
        for lf in (0.2, 0.3, 0.4):
            w = S.book(S.cohort_weights(score, h, lf), member, h)
            assert (np.abs(w).sum(axis=1) <= 1.0 + 1e-12).all()
            assert np.allclose(w.sum(axis=1), 0.0, atol=1e-12)      # every cohort dollar-neutral


def test_params_change_book():
    N = 30
    score = np.vstack([_score_row(30, N)] * 10)
    member = np.ones((10, N), bool)
    a = S.book(S.cohort_weights(score, 5, 0.3), member, 5)
    b = S.book(S.cohort_weights(score, 5, 0.4), member, 5)
    c = S.book(S.cohort_weights(score, 3, 0.3), member, 3)
    assert (np.abs(a - b).sum() > 0) and (np.abs(a - c).sum() > 0)
    assert (a > 0).sum(axis=1)[-1] == 9 and (b > 0).sum(axis=1)[-1] == 12


def test_end_to_end_synthetic():
    from lab.framework.data import synthetic
    data = synthetic()
    T, N = len(data.dates), len(data.instruments)
    fund = np.random.default_rng(5).normal(0, 1e-4, (T, N))
    fund[:40] = np.nan                               # funding starts late
    data.extras["funding"] = fund
    w = target_weights(data, PARAMS)
    assert w.shape == (len(data.dates), len(data.instruments))
    assert np.isfinite(w).all()
    assert (np.abs(w).sum(axis=1) <= 1.0 + 1e-9).all()
    score, member = S.scores(data, PARAMS)
    first = np.where(np.isfinite(score).any(axis=1))[0]
    if len(first):
        assert np.abs(w[: first[0]]).sum() == 0     # no weights before the first fit
    assert not (w[~member] != 0).any()             # only members are held
