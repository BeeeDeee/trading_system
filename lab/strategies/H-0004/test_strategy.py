import numpy as np
import pytest

import strategy as strat_mod
from lab.framework.data import DataView, synthetic
from strategy import target_weights, PARAMS

T = 520
NAMES = ("S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07")


def _dates():
    # daily 24/7 calendar starting on a Thursday -> Sundays are rows r % 7 == 3
    base = synthetic().dates[0].astype("datetime64[D]")
    thu = np.busday_offset(base, 0, roll="forward", weekmask="0001000")
    return thu + np.arange(T)


def make_view(universe=None, dv=None):
    N = len(NAMES)
    bar = np.zeros((T, N), dtype=bool)
    bar[:, 0:4] = True              # seasoned pairs, listed on row 0
    bar[400:, 4:7] = True           # new listings on row 400
    bar[380:430, 7] = True          # S07 listed row 380, gap -> delisted on row 430
    bar[440:, 7] = True             # ... and re-appears on row 440 (must not be re-entered)
    bar[423, 5] = False             # S05 has no bar on the Sunday row 423
    delisting = np.zeros((T, N), dtype=bool)
    delisting[430, 7] = True
    listed = bar.copy()
    close = np.where(bar, 10.0, np.nan)
    if dv is None:
        dv = np.ones((T, N)) * 1e6
    dollar_volume = np.where(bar, dv, np.nan)
    tradable = np.zeros((T, N), dtype=bool)
    tradable[1:] = bar[:-1]
    return DataView(
        dates=_dates(), instruments=NAMES, asset_class=tuple("crypto_spot" for _ in NAMES),
        ret_co=np.zeros((T, N)), ret_oc=np.zeros((T, N)), tradable=tradable, listed=listed,
        delisting=delisting, close=close, dollar_volume=dollar_volume, universe=universe,
        extras={}, series={}, cash_ret=np.zeros(T),
    )


def test_params_match_card():
    assert PARAMS == {"max_age_days": 90, "skip_days": 14}


def test_sunday_schedule_and_nan_elsewhere():
    d = make_view()
    assert (d.dates[3] - d.dates[0]).astype("int64") == 3
    w = target_weights(d, PARAMS)
    assert w.shape == (T, len(NAMES))
    decided = ~np.isnan(w).all(axis=1)
    rows = np.flatnonzero(decided)
    assert np.all(rows % 7 == 3)
    assert rows.size == len(range(3, T, 7))
    assert not np.isnan(w[decided]).any()


def test_cash_before_legs_are_full():
    w = target_weights(make_view(), PARAMS)
    for t in range(3, 409, 7):
        assert np.all(w[t] == 0.0)


def test_both_legs_known_row():
    w = target_weights(make_view(), PARAMS)
    # row 416: new pairs age 16, S07 age 36 -> short 4; seasoned S00-S03 age 416 -> long 4
    np.testing.assert_allclose(w[416], [0.125] * 4 + [-0.125] * 4)
    assert abs(w[416].sum()) < 1e-12
    assert abs(np.abs(w[416]).sum() - 1.0) < 1e-12


def test_skip_days_excludes_too_young():
    w = target_weights(make_view(), PARAMS)
    # row 409: S04-S06 age 9 < 14, only S07 (age 29) in short leg -> fewer than 3 -> cash
    assert np.all(w[409] == 0.0)
    w7 = target_weights(make_view(), {"max_age_days": 90, "skip_days": 7})
    np.testing.assert_allclose(w7[409], [0.125] * 4 + [-0.125] * 4)


def test_missing_bar_not_eligible():
    w = target_weights(make_view(), PARAMS)
    # row 423: S05 has no bar -> short S04, S06, S07
    np.testing.assert_allclose(w[423], [0.125] * 4 + [-0.5 / 3, 0.0, -0.5 / 3, -0.5 / 3])


def test_delisted_never_reentered():
    w = target_weights(make_view(), PARAMS)
    for t in (437, 444, 451):
        np.testing.assert_allclose(w[t], [0.125] * 4 + [-0.5 / 3] * 3 + [0.0])


def test_max_age_boundary_and_param():
    w = target_weights(make_view(), PARAMS)
    np.testing.assert_allclose(w[486][4:7], [-0.5 / 3] * 3)   # age 86 <= 90
    assert np.all(w[493] == 0.0)                               # age 93 > 90 -> short empty -> cash
    w60 = target_weights(make_view(), {"max_age_days": 60, "skip_days": 14})
    assert np.all(w60[465] == 0.0)                             # age 65 > 60
    np.testing.assert_allclose(w[465][4:7], [-0.5 / 3] * 3)
    w120 = target_weights(make_view(), {"max_age_days": 120, "skip_days": 14})
    np.testing.assert_allclose(w120[493][4:7], [-0.5 / 3] * 3)


def test_seasoned_threshold_365():
    # with an older short candidate set: at row 360 seasoned pairs are age 360 < 365 -> no long leg
    w = target_weights(make_view(), {"max_age_days": 120, "skip_days": 7})
    assert np.all(w[360] == 0.0)


def test_universe_membership_respected():
    uni = np.ones((T, len(NAMES)), dtype=bool)
    uni[416, 0] = False
    w = target_weights(make_view(universe=uni), PARAMS)
    np.testing.assert_allclose(w[416], [0.0] + [0.5 / 3] * 3 + [-0.125] * 4)


def test_fallback_ranking_by_median_volume(monkeypatch):
    monkeypatch.setattr(strat_mod, "TOP_N", 6)
    dv = np.ones((T, len(NAMES))) * 100.0
    dv[:, 0] = 1.0     # S00 lowest median volume
    dv[:, 4] = 2.0     # S04 second lowest
    w = target_weights(make_view(dv=dv), PARAMS)
    np.testing.assert_allclose(w[416], [0.0] + [0.5 / 3] * 3 + [0.0] + [-0.5 / 3] * 3)


def test_gross_and_determinism_and_truncation():
    d = make_view()
    w1 = target_weights(d, PARAMS)
    w2 = target_weights(d, PARAMS)
    np.testing.assert_array_equal(w1, w2)
    gross = np.nansum(np.abs(w1), axis=1)
    assert np.all(gross <= 1.0 + 1e-12)
    cut = 430
    dt = DataView(
        dates=d.dates[:cut], instruments=d.instruments, asset_class=d.asset_class,
        ret_co=d.ret_co[:cut], ret_oc=d.ret_oc[:cut], tradable=d.tradable[:cut], listed=d.listed[:cut],
        delisting=d.delisting[:cut], close=d.close[:cut], dollar_volume=d.dollar_volume[:cut],
        universe=None, extras={}, series={}, cash_ret=d.cash_ret[:cut],
    )
    wt = target_weights(dt, PARAMS)
    np.testing.assert_array_equal(np.isnan(wt), np.isnan(w1[:cut]))
    np.testing.assert_allclose(np.nan_to_num(wt), np.nan_to_num(w1[:cut]))


def test_synthetic_runs():
    d = synthetic()
    w = target_weights(d, PARAMS)
    assert w.shape == (len(d.dates), len(d.instruments))
    assert np.all(np.nansum(np.abs(w), axis=1) <= 1.0 + 1e-12)
