import dataclasses

import numpy as np
import pytest

from strategy import PARAMS, target_weights
from strategy import high_vol_half, lag, raw_features, rolling_sum, week_end_days
from lab.framework.data import synthetic


# ----------------------------------------------------------------------------- helpers

def test_rolling_sum_window_and_nan():
    x = np.arange(1.0, 8.0).reshape(-1, 1)            # 1..7
    s = rolling_sum(x, 3)
    assert np.all(np.isnan(s[:2]))
    assert s[2, 0] == 6.0 and s[3, 0] == 9.0 and s[6, 0] == 18.0
    x[3, 0] = np.nan
    s = rolling_sum(x, 3)
    assert np.all(np.isnan(s[3:6, 0])) and s[6, 0] == 5.0 + 6.0 + 7.0
    assert np.all(np.isnan(rolling_sum(np.ones((2, 1)), 3)))   # view shorter than the window


def test_lag():
    x = np.arange(5.0).reshape(-1, 1)
    l2 = lag(x, 2)
    assert np.all(np.isnan(l2[:2])) and list(l2[2:, 0]) == [0.0, 1.0, 2.0]
    assert np.all(np.isnan(lag(np.ones((2, 1)), 5)))


def test_week_end_days_are_fridays():
    dates = np.arange(0, 14).astype("datetime64[D]")    # starts on Thursday 1970-01-01
    fri = week_end_days(dates)
    assert list(np.nonzero(fri)[0]) == [1, 8]


def test_high_vol_half_median_split():
    vol = np.array([[1.0, 2.0, 3.0, 4.0, np.nan, 9.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    elig = np.array([[True, True, True, True, True, False],
                     [False, False, False, False, False, False]])
    hv = high_vol_half(vol, elig)
    assert list(hv[0]) == [False, False, True, True, False, False]   # median 2.5, non-member 9 excluded
    assert not hv[1].any()
    # odd count: the median name itself is "at or above"
    hv = high_vol_half(np.array([[1.0, 2.0, 3.0]]), np.ones((1, 3), dtype=bool))
    assert list(hv[0]) == [False, True, True]


def test_raw_features_by_hand():
    T = 80
    ret_co = np.zeros((T, 2))
    ret_oc = np.full((T, 2), 0.01)
    ret_co[:, 1] = 0.002
    dv = np.full((T, 2), 1e6)
    dv[T - 5:, 1] = 2e6                                  # last 5 days doubled volume for stock 1
    listed = np.ones((T, 2), dtype=bool)
    listed[:10, 1] = False                               # late listing of stock 1
    f_oc, f_co, f_wk, f_dv, vol = raw_features(ret_co, ret_oc, dv, listed)
    assert np.all(np.isnan(f_oc[:4])) and np.allclose(f_oc[4:, 0], 0.05)
    assert np.allclose(f_co[4:, 1], 0.01)
    assert np.all(np.isnan(f_wk[:9])) and np.allclose(f_wk[9:, 0], 1.01 ** 5 - 1)
    assert np.allclose(f_wk[9:, 1], (1.01 * 1.002) ** 5 - 1)
    assert np.all(np.isnan(f_dv[:67])) and np.allclose(f_dv[67:, 0], 0.0)
    assert np.isclose(f_dv[T - 1, 1], np.log(2.0))
    # vol needs 64 listed rows: stock 0 from row 63, stock 1 (listed at row 10) from row 73
    assert np.all(np.isnan(vol[:63, 0])) and np.allclose(vol[63:, 0], 0.0, atol=1e-7)
    assert np.all(np.isnan(vol[:73, 1])) and np.allclose(vol[73:, 1], 0.0, atol=1e-7)


def test_dollar_volume_nan_or_zero_is_incomplete():
    T = 150
    z = np.zeros((T, 1))
    dv = np.full((T, 1), 1e6)
    dv[70, 0] = np.nan
    f_dv = raw_features(z, z, dv, np.ones((T, 1), dtype=bool))[3]
    # row 70 is in the recent window for t = 70..74 and in the prior window (t-67..t-5) for t = 75..137
    assert np.isfinite(f_dv[69, 0])
    assert np.all(np.isnan(f_dv[70:138])) and np.isfinite(f_dv[138, 0])
    T = 80
    z = np.zeros((T, 1))
    dv = np.full((T, 1), 1e6)
    dv[T - 5:, 0] = 0.0
    f_dv = raw_features(z, z, dv, np.ones((T, 1), dtype=bool))[3]
    assert np.isnan(f_dv[T - 1, 0])


# ----------------------------------------------------------------------------- full pipeline

def _widen(m, copies=3):
    """The synthetic market has 30 names, too few for a 20-name high-vol half; stack time-shifted copies of it."""
    T, N = len(m.dates), len(m.instruments)

    def tile(v):
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape == (T, N):
            return np.concatenate([np.roll(v, 11 * k, axis=0) for k in range(copies)], axis=1)
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.shape[0] == N and T != N:
            return np.concatenate([v] * copies)
        return v

    d = dataclasses.asdict(m)
    changes = {}
    for key, v in d.items():
        if key == "instruments":
            changes[key] = tuple(name + "_" + str(k) for k in range(copies) for name in v)
        elif isinstance(v, dict):
            changes[key] = {kk: tile(vv) for kk, vv in v.items()}
        elif isinstance(v, np.ndarray):
            changes[key] = tile(v)
    return dataclasses.replace(m, **changes)


@pytest.fixture(scope="module")
def market():
    m = _widen(synthetic())
    # point-in-time universe: listed single names, minus a block of rows for S00_0 (a member that leaves and returns)
    uni = np.asarray(m.listed, dtype=bool).copy()
    for k in range(3):
        uni[:, m.col("MKT_" + str(k))] = False
    uni[2000:2500, m.col("S00_0")] = False
    # synthetic dollar volume has no cross-sectional dispersion (K-0135): the volume z-score would be NaN for all
    rng = np.random.default_rng(7)
    dv = np.asarray(m.dollar_volume, dtype=np.float64) * np.exp(0.5 * rng.standard_normal(uni.shape))
    return dataclasses.replace(m, universe=uni, dollar_volume=dv)


@pytest.fixture(scope="module")
def base(market):
    return target_weights(market, PARAMS)


def test_shape_schedule_and_gross(market, base):
    w = base
    assert w.shape == (len(market.dates), len(market.instruments))
    fri = week_end_days(market.dates)
    assert np.all(np.isnan(w[~fri]))
    assert np.all(np.isfinite(w[fri]))
    assert np.all(np.abs(w[fri]).sum(axis=1) <= 1.0 + 1e-9)


def test_dollar_neutral_and_members_only(market, base):
    w = base
    fri = week_end_days(market.dates)
    rows = np.nonzero(fri & (np.abs(np.nan_to_num(w)).sum(axis=1) > 0))[0]
    assert rows.size > 0, "the book never invests on the synthetic market"
    for t in rows:
        assert np.isclose(w[t][w[t] > 0].sum(), 0.5) and np.isclose(w[t][w[t] < 0].sum(), -0.5)
    if market.universe is not None:
        assert np.all(np.nan_to_num(w)[~np.asarray(market.universe, dtype=bool)] == 0.0)


def test_flat_before_first_fit(market, base):
    w = np.nan_to_num(base)
    first = np.nonzero(np.abs(w).sum(axis=1) > 0)[0][0]
    assert first > 63 + 63                            # features need 64 rows, then a training sample


def test_params_change_output(market, base):
    a = target_weights(market, dict(PARAMS, top=0.3))
    assert not np.array_equal(np.nan_to_num(a), np.nan_to_num(base))
    b = target_weights(market, dict(PARAMS, alpha=1e6))
    assert not np.array_equal(np.nan_to_num(b), np.nan_to_num(base))
