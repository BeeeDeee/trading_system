import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS, UNIVERSE, events

N = len(UNIVERSE)
T = 420
SMALL = {"formation_months": 1, "leg_size": 2}   # F = 21 sessions, need >= 7 eligible
F_SMALL = 21


def _dates(T):
    base = np.array([10957]).astype("datetime64[D]")[0]   # 2000-01-01
    return np.busday_offset(base, np.arange(T), roll="forward")


def _month_starts(dates):
    m = dates.astype("datetime64[M]")
    s = np.zeros(len(dates), dtype=bool)
    s[1:] = m[1:] != m[:-1]
    s[0] = True
    return s


DATES = _dates(T)
STARTS = _month_starts(DATES)
T0 = int(np.flatnonzero(STARTS & (np.arange(T) >= F_SMALL + 252))[0])


def _view(ret_oc, close, T_cut=None):
    T_ = ret_oc.shape[0] if T_cut is None else T_cut
    ret_oc = ret_oc[:T_]
    close = close[:T_]
    listed = np.isfinite(close)
    return DataView(
        dates=DATES[:T_],
        instruments=tuple(UNIVERSE),
        asset_class=tuple(["us_etf"] * N),
        ret_co=np.zeros_like(ret_oc),
        ret_oc=np.where(listed, ret_oc, 0.0),
        tradable=listed.copy(),
        listed=listed.copy(),
        delisting=np.zeros_like(listed),
        close=close,
        dollar_volume=np.where(listed, 1e8, np.nan),
        universe=None,
        extras={},
        series={},
        cash_ret=np.zeros(T_),
    )


def _market(a, s):
    """Rows <= T0-252: constant drift a_i. Rows > T0-252: alternating +-s_i (vol window)."""
    ret = np.tile(np.asarray(a, float), (T, 1))
    k = np.arange(T)[:, None]
    alt = np.where(k % 2 == 0, 1.0, -1.0) * np.asarray(s, float)[None, :]
    ret[T0 - 252 + 1:] = alt[T0 - 252 + 1:]
    close = np.full((T, N), 50.0)
    return ret, close


A = np.linspace(-0.002, 0.002, N)[np.random.default_rng(7).permutation(N)]
S = np.linspace(0.005, 0.025, N)[np.random.default_rng(11).permutation(N)]


def _expected(a, s, L, elig=None):
    idx = np.arange(N) if elig is None else np.flatnonzero(elig)
    order = idx[np.argsort(a[idx], kind="stable")]
    longs, shorts = order[:L], order[::-1][:L]
    raw = np.zeros(N)
    raw[longs] = 1 / s[longs]
    raw[shorts] = -1 / s[shorts]
    return raw / np.abs(raw).sum()


def test_params_match_card():
    assert PARAMS == {"formation_months": 48, "leg_size": 5}


def test_weights_on_decision_row():
    ret, close = _market(A, S)
    w = target_weights(_view(ret, close), SMALL)
    np.testing.assert_allclose(w[T0], _expected(A, S, 2), rtol=1e-9, atol=1e-12)
    assert np.isclose(np.abs(w[T0]).sum(), 1.0)
    assert (w[T0] > 0).sum() == 2 and (w[T0] < 0).sum() == 2


def test_nan_between_decisions_and_flat_before_lookback():
    ret, close = _market(A, S)
    w = target_weights(_view(ret, close), SMALL)
    dec = STARTS.copy()
    assert np.all(np.isnan(w[~dec]))
    assert np.all(np.isfinite(w[dec]))
    early = np.flatnonzero(dec & (np.arange(T) < F_SMALL + 252))
    assert len(early) > 0 and np.all(w[early] == 0.0)
    gross = np.abs(w[dec]).sum(axis=1)
    assert np.all(gross <= 1.0 + 1e-12)


def test_ties_broken_by_universe_order():
    a = np.linspace(-0.001, 0.001, N)
    a[[0, 1, 2]] = -0.003      # SPY, EWA, EWC tie lowest -> long SPY, EWA
    a[[18, 19, 20]] = 0.003    # EWY, EWT, EWZ tie highest -> short EWY, EWT
    s = np.full(N, 0.01)
    ret, close = _market(a, s)
    w = target_weights(_view(ret, close), SMALL)[T0]
    assert set(np.flatnonzero(w > 0)) == {0, 1}
    assert set(np.flatnonzero(w < 0)) == {18, 19}
    np.testing.assert_allclose(np.abs(w[w != 0]), 0.25)


def test_late_listing_is_ineligible():
    ret, close = _market(A, S)
    lowest = int(np.argmin(A))
    s0 = T0 - (F_SMALL + 252)
    close[: s0 + 1, lowest] = np.nan        # no price at close of t-(F+252)
    w = target_weights(_view(ret, close), SMALL)[T0]
    elig = np.ones(N, bool)
    elig[lowest] = False
    assert w[lowest] == 0.0
    np.testing.assert_allclose(w, _expected(A, S, 2, elig), rtol=1e-9, atol=1e-12)


def test_too_few_eligible_is_flat():
    ret, close = _market(A, S)
    s0 = T0 - (F_SMALL + 252)
    close[: s0 + 1, :15] = np.nan            # 6 eligible < 2*2+3
    w = target_weights(_view(ret, close), SMALL)[T0]
    assert np.all(w == 0.0)
    close[: s0 + 1, :14] = np.nan
    close[: s0 + 1, 14] = 50.0               # 7 eligible -> book formed
    w = target_weights(_view(ret, close), SMALL)[T0]
    assert np.isclose(np.abs(w).sum(), 1.0)


def test_parameters_change_output():
    ret, close = _market(A, S)
    v = _view(ret, close)
    w = target_weights(v, SMALL)[T0]
    w3 = target_weights(v, {"formation_months": 1, "leg_size": 3})[T0]
    assert (w3 != 0).sum() == 6 and not np.allclose(w, w3)
    np.testing.assert_allclose(w3, _expected(A, S, 3), rtol=1e-9, atol=1e-12)
    w2 = target_weights(v, {"formation_months": 2, "leg_size": 2})
    # F = 42: the first month start with a full lookback is later than T0
    assert np.all(w2[T0] == 0.0)
    assert not np.allclose(np.nan_to_num(w2), np.nan_to_num(target_weights(v, SMALL)))


def test_point_in_time_truncation_and_noise():
    rng = np.random.default_rng(3)
    ret = rng.normal(0.0003, 0.01, (T, N))
    close = np.full((T, N), 50.0)
    close[:100, 5] = np.nan
    full = target_weights(_view(ret, close), SMALL)
    for cut in (T0, T0 + 1, T0 + 40, 380):
        part = target_weights(_view(ret, close, T_cut=cut + 1), SMALL)
        np.testing.assert_array_equal(part, full[: cut + 1])
        noisy = ret.copy()
        noisy[cut + 1:] = rng.normal(0, 0.05, noisy[cut + 1:].shape)
        w2 = target_weights(_view(noisy, close), SMALL)
        np.testing.assert_array_equal(w2[: cut + 1], full[: cut + 1])


def test_diagnostic_matches_strategy_legs():
    ret, close = _market(A, S)
    v = _view(ret, close)
    w = target_weights(v, SMALL)
    mask, side = events(v, SMALL)
    dec = np.isfinite(w).all(axis=1)
    assert not mask[~dec].any()
    np.testing.assert_array_equal(mask, np.nan_to_num(w) != 0)
    np.testing.assert_array_equal(side[mask], np.sign(w[mask]))
