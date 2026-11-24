import numpy as np
import pytest

from strategy import (
    PARAMS,
    target_weights,
    lag_rows,
    low_vol_exposure,
    momentum_exposure,
    quintile_masks,
    scale_by_trailing_sd,
    state_inputs,
    top_fraction_book,
    value_exposure,
    weekly_decision_mask,
)


def test_params_match_card():
    assert PARAMS == {"n_estimators": 100, "top": 0.2}


def test_weekly_mask_is_friday():
    # 1970-01-01 (day 0) was a Thursday; days 1, 8, 15 are Fridays
    dates = np.arange(0, 20).astype("datetime64[D]")
    m = weekly_decision_mask(dates)
    assert list(np.flatnonzero(m)) == [1, 8, 15]


def test_lag_rows():
    x = np.arange(6, dtype=float).reshape(3, 2)
    out = lag_rows(x, 1)
    assert np.isnan(out[0]).all()
    assert np.array_equal(out[1:], x[:-1])
    b = lag_rows(np.ones((2, 2), dtype=bool), 5)
    assert not b.any()


def test_low_vol_min_valid_and_value():
    T = 70
    r = np.zeros((T, 2))
    r[:, 0] = np.where(np.arange(T) % 2 == 0, 0.01, -0.01)
    valid = np.ones((T, 2), dtype=bool)
    valid[:, 1] = np.arange(T) % 2 == 0          # at most 32 valid in any 63-row window
    out = low_vol_exposure(r, valid, window=63, min_valid=50)
    assert np.isnan(out[48, 0])                   # 49 rows only
    assert np.isfinite(out[49, 0])
    w = r[T - 63:, 0]
    assert out[-1, 0] == pytest.approx(-np.std(w, ddof=1))
    assert np.isnan(out[:, 1]).all()


def test_momentum_skip_month():
    T = 260
    idx = np.cumprod(np.full((T, 1), 1.001), axis=0)
    listed = np.ones((T, 1), dtype=bool)
    out = momentum_exposure(idx, listed, long=252, skip=21)
    assert np.isnan(out[251, 0])
    assert out[255, 0] == pytest.approx(idx[255 - 21, 0] / idx[255 - 252, 0] - 1.0)
    listed[3, 0] = False
    out2 = momentum_exposure(idx, listed, long=252, skip=21)
    assert np.isnan(out2[255, 0]) and np.isfinite(out2[256, 0])


def test_value_frozen_at_filing_and_moved_with_price():
    T = 8
    idx = np.array([1.0, 1.0, 1.1, 1.2, 1.0, 0.8, 0.9, 1.0])[:, None]
    pb = np.full((T, 1), np.nan)
    pb[2:5, 0] = 2.0        # visible from row 2 -> filing day row 1
    pb[5:, 0] = -1.0        # negative book -> not scored
    out = value_exposure(pb, idx)
    assert np.isnan(out[:2, 0]).all()
    assert out[2, 0] == pytest.approx(0.5 * 1.0 / 1.1)
    assert out[4, 0] == pytest.approx(0.5 * 1.0 / 1.0)
    assert np.isnan(out[5:, 0]).all()
    pb[6:, 0] = 4.0         # new filing visible on row 6 -> anchor row 5
    out = value_exposure(pb, idx)
    assert out[7, 0] == pytest.approx(0.25 * 0.8 / 1.0)


def test_quintiles_and_spread_use_t_minus_5_groups():
    T, N = 12, 10
    comp = np.tile(np.arange(N, dtype=float), (T, 1))
    uni = np.ones((T, N), dtype=bool)
    uni[:, 9] = False                              # non-member never in a quintile
    top, bot = quintile_masks(comp, uni)
    assert list(np.flatnonzero(top[0])) == [8]    # 9 members -> 1 name per quintile
    assert list(np.flatnonzero(bot[0])) == [0]
    idx = np.ones((T, N))
    idx[7:, 8] = 1.1                                # top name gains 10 % between rows 6 and 7
    comp2 = comp.copy()
    comp2[6:, :] = comp[6:, ::-1]                   # later re-sorting must not affect the t-5 groups
    s_raw_groups = quintile_masks(comp2, uni)
    assert s_raw_groups[0][0, 8]


def test_scale_by_trailing_sd_previous_window():
    rng = np.random.default_rng(0)
    s = rng.normal(size=300)
    out = scale_by_trailing_sd(s, window=252, min_valid=200)
    assert np.isnan(out[199]) and np.isfinite(out[200])
    assert out[260] == pytest.approx(s[260] / np.std(s[8:260], ddof=1))
    s[100:200] = np.nan
    out = scale_by_trailing_sd(s, window=252, min_valid=200)
    assert np.isnan(out[260])


def test_state_inputs_spread_direction():
    T, N = 300, 10
    comp = np.tile(np.arange(N, dtype=float), (T, 1))
    uni = np.ones((T, N), dtype=bool)
    rng = np.random.default_rng(1)
    rets = rng.normal(0, 0.01, size=(T, N))
    rets[290, 9] = -0.30                            # top-quintile names crash on row 290
    rets[290, 8] = -0.30
    idx = np.cumprod(1 + rets, axis=0)
    unwind, market = state_inputs(comp, idx, uni)
    assert unwind.shape == (T,) and market.shape == (T,)
    assert unwind[292] < -3.0
    assert np.isnan(unwind[:205]).all()


def test_helpers_are_point_in_time_under_truncation():
    rng = np.random.default_rng(3)
    T, N, cut = 520, 12, 470
    rets = rng.normal(0, 0.02, size=(T, N))
    idx = np.cumprod(1 + rets, axis=0)
    listed = np.ones((T, N), dtype=bool)
    listed[:300, 5] = False                         # late listing
    uni = listed.copy()
    uni[:400, 6] = False                            # late universe entrant
    pb = np.repeat(rng.uniform(0.5, 3, size=(T // 65 + 1, N)), 65, axis=0)[:T]
    valid = listed & lag_rows(listed, 1)
    comp = rng.normal(size=(T, N))
    full = [value_exposure(pb, idx), momentum_exposure(idx, listed), low_vol_exposure(rets, valid)]
    part = [value_exposure(pb[:cut], idx[:cut]), momentum_exposure(idx[:cut], listed[:cut]),
            low_vol_exposure(rets[:cut], valid[:cut])]
    for a, b in zip(full, part):
        assert np.array_equal(a[:cut], b, equal_nan=True)
    uf, mf = state_inputs(comp, idx, uni)
    up, mp = state_inputs(comp[:cut], idx[:cut], uni[:cut])
    assert np.array_equal(uf[:cut], up, equal_nan=True)
    assert np.array_equal(mf[:cut], mp, equal_nan=True)


def test_book_top_fraction_and_small_sets():
    T, N = 3, 60
    score = np.tile(np.arange(N, dtype=float), (T, 1))
    uni = np.ones((T, N), dtype=bool)
    score[1, 10:] = np.nan                          # only 10 scored -> all equally
    score[2, :] = np.nan                            # nothing scored -> flat
    w = top_fraction_book(score, uni, 0.2)
    assert list(np.flatnonzero(w[0])) == list(range(48, 60))
    assert w[0].sum() == pytest.approx(1.0)
    assert np.allclose(w[1, :10], 0.1) and w[1, 10:].sum() == 0
    assert w[2].sum() == 0
    w1 = top_fraction_book(score, uni, 0.1)
    assert list(np.flatnonzero(w1[0])) == list(range(54, 60))
    uni[0, 59] = False
    w2 = top_fraction_book(score, uni, 0.2)
    assert w2[0, 59] == 0 and (w2 >= 0).all()
