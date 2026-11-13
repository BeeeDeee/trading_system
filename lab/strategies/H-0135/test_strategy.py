from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from strategy import target_weights, PARAMS


@dataclass
class Fake:
    dates: Any
    instruments: Any
    close: Any
    dollar_volume: Any
    universe: Any
    tradable: Any


T = 400


def _dates(n):
    return (np.arange(n) + 17000).astype("datetime64[D]")


def _base():
    close = np.ones((T, 5))
    close[100, 1] = 2.0
    close[364:, 1] = 1.5          # NH(364): 1.5/2 = 0.75 with W=365; 1.0 with W=180
    close[50, 2] = 2.0            # NH(364): 0.5 with W=365; 1.0 with W=180
    close[200, 3] = 1.0 / 0.9     # NH(364): 0.9 for both windows
    close[:100, 4] = np.nan       # listed late: too young at row 364
    vol = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), (T, 1))
    uni = np.ones((T, 5), dtype=bool)
    return close, vol, uni


def _make(close, vol, uni, n=None):
    n = close.shape[0] if n is None else n
    N = close.shape[1]
    return Fake(dates=_dates(close.shape[0])[:n], instruments=tuple("C%d" % i for i in range(N)),
                close=close[:n], dollar_volume=vol[:n], universe=None if uni is None else uni[:n],
                tradable=np.ones((n, N), dtype=bool))


def test_first_formation_needs_a_year_and_two_coins():
    w = target_weights(_make(*_base()), PARAMS)
    assert np.isnan(w[:364]).all()
    assert not np.isnan(w[364]).any()
    assert np.isnan(w[365]).all()
    assert not np.isnan(w[371]).any()
    assert w[364, 4] == 0.0       # too young


def test_nh_ranking_and_k_half_of_eligible():
    w = target_weights(_make(*_base()), PARAMS)
    # E = 4 -> k = min(5, 2) = 2: NH 1.0 (C0) and 0.9 (C3)
    assert np.allclose(w[364], [0.125, 0, 0, 0.125, 0])
    w1 = target_weights(_make(*_base()), {**PARAMS, "n_hold": 1})
    assert np.allclose(w1[364], [0.25, 0, 0, 0, 0])


def test_window_param_and_volume_tie_break():
    w = target_weights(_make(*_base()), {**PARAMS, "high_window_days": 180})
    # W=180: C0, C1, C2 all at NH 1 -> highest 30-day volume first: C2, C1
    assert np.allclose(w[364], [0, 0.125, 0.125, 0, 0])


def test_overlapping_cohorts_and_expiry():
    w = target_weights(_make(*_base()), PARAMS)
    assert np.isclose(np.nansum(w[364]), 0.25)
    assert np.isclose(np.nansum(w[371]), 0.5)
    assert np.isclose(np.nansum(w[378]), 0.75)
    assert np.isclose(np.nansum(w[385]), 1.0)
    assert np.isclose(np.nansum(w[392]), 1.0)   # first cohort dropped, a new one added
    assert np.allclose(w[392], [0.5, 0, 0, 0.5, 0])


def test_universe_exit_sells_and_slice_stays_cash():
    close, vol, uni = _base()
    uni[366, 0] = False
    w = target_weights(_make(close, vol, uni), PARAMS)
    assert np.isnan(w[365]).all()
    assert np.allclose(w[366], [0, 0, 0, 0.125, 0])   # C0 sold, C3 kept at its target
    assert np.isnan(w[367]).all()                  # re-entry does not revive the slice
    assert np.allclose(w[371], [0.125, 0, 0, 0.25, 0])
    assert np.allclose(w[385], [0.375, 0, 0, 0.5, 0])
    assert np.allclose(w[392], [0.5, 0, 0, 0.5, 0])


def test_empty_cohort_when_fewer_than_two_eligible():
    close, vol, uni = _base()
    uni[371, 1:] = False                            # only C0 eligible on 371 -> cash cohort
    w = target_weights(_make(close, vol, uni), PARAMS)
    assert np.allclose(w[371], [0.125, 0, 0, 0, 0])  # C3 left the universe; no new cohort
    assert np.isclose(np.nansum(w[378]), 0.25 + 0.125)


def test_fallback_universe_top20_by_volume():
    rng = np.random.default_rng(1)
    N = 25
    close = np.exp(np.cumsum(rng.normal(0, 0.03, (T, N)), axis=0))
    vol = np.tile(np.arange(1.0, N + 1.0), (T, 1))   # columns 5..24 are the top 20
    w = target_weights(_make(close, vol, None), PARAMS)
    held = np.nan_to_num(w) > 0
    assert held.any()
    assert not held[:, :5].any()


def test_gross_long_only_determinism_and_point_in_time():
    rng = np.random.default_rng(7)
    N, TT = 30, 700
    close = np.exp(np.cumsum(rng.normal(0, 0.04, (TT, N)), axis=0))
    close[:300, :6] = np.nan
    vol = rng.uniform(1, 100, (TT, N))
    uni = rng.uniform(size=(TT, N)) < 0.7
    d = _make(close, vol, uni)
    w = target_weights(d, PARAMS)
    w2 = target_weights(d, PARAMS)
    assert np.array_equal(w, w2, equal_nan=True)
    assert (np.nan_to_num(w) >= 0).all()
    assert (np.nansum(w, axis=1) <= 1.0 + 1e-12).all()
    assert (np.nan_to_num(w)[~uni] == 0).all()
    for cut in (370, 371, 450, 500, 640):
        wc = target_weights(_make(close, vol, uni, n=cut + 1), PARAMS)
        assert np.array_equal(wc, w[:cut + 1], equal_nan=True)
    for p in ({"high_window_days": 180}, {"high_window_days": 540}, {"n_hold": 3}, {"n_hold": 8}):
        wp = target_weights(d, {**PARAMS, **p})
        assert not np.array_equal(wp, w, equal_nan=True)


def test_short_view_runs():
    close, vol, uni = _base()
    w = target_weights(_make(close, vol, uni, n=18), PARAMS)
    assert w.shape == (18, 5) and np.isnan(w).all()
