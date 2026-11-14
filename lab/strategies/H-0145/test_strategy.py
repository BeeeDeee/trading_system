import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS, _cs_residual, _rolling_median, _universe, _formations

NAMES = ("BTCUSDT.P",) + tuple("C%02dUSDT.P" % j for j in range(1, 12))
# betas: BTC 1.0, C01..C11 = 0.2, 0.35, ..., 1.7 (C01..C06 below BTC, C07..C11 above)
BETAS = np.array([1.0] + [0.15 * j + 0.05 for j in range(1, 12)])
# recent basis level (x 0.001) per column; ref window is +-0.001 noise, the same for every coin
RECENT = np.array([5, 3, 9, 1, 7, 2, 8, 11, 0, 6, 4, 10], dtype=float) * 0.001
T0 = 154          # first formation row with full history (dates start on row 0; 154 = 22 x 7)


def make_view(T=160, universe=None, names=NAMES, basis=None, close=None, dollar_volume=None):
    N = len(names)
    dates = (np.arange(T) + 18262).astype("datetime64[D]")     # day 18262 after the epoch
    if close is None:
        btc_lr = np.where(np.arange(T) % 2 == 0, 0.01, -0.01)
        btc_lr[0] = 0.0
        btc_lr[T0 - 29:] = 0.0                # flat closes over the last 30 days: r30 = 0 exactly
        lr = btc_lr[:, None] * BETAS[None, :N]
        close = 100.0 * np.exp(np.cumsum(lr, axis=0))
    if basis is None:
        noise = np.where(np.arange(T) % 2 == 0, 0.001, -0.001)
        basis = np.repeat(noise[:, None], N, axis=1)
        basis[:25] = np.nan                   # first formation with 96 valid ref days is row 154
        basis[T0 - 29:] = RECENT[None, :N]
    if dollar_volume is None:
        dollar_volume = np.ones((T, N)) * 1e8
    if universe is None:
        universe = np.ones((T, N), dtype=bool)
    z = np.zeros((T, N))
    return DataView(
        dates=dates,
        instruments=tuple(names),
        asset_class=tuple("crypto_perp" for _ in names),
        ret_co=z.copy(),
        ret_oc=z.copy(),
        tradable=np.ones((T, N), dtype=bool),
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=dollar_volume,
        universe=universe if universe is not False else None,
        extras={"basis": basis, "funding": z.copy()},
        series={},
        cash_ret=np.zeros(T),
    )


def expected_cohort(N=12):
    # low-beta half C01..C06: recent 3, 9, 1, 7, 2, 8 -> short C02, long C03
    # high-beta half BTC, C07..C11: 5, 11, 0, 6, 4, 10 -> short C07, long C08
    w = np.zeros(N)
    w[3] = w[8] = 0.25
    w[2] = w[7] = -0.25
    return w


def test_hand_derived_cohort_and_timing():
    data = make_view()
    W = target_weights(data, PARAMS)
    assert W.shape == (160, 12)
    assert np.all(W[:T0] == 0.0)              # 147 and earlier lack 96 valid ref basis days
    for r in range(T0, 160):
        np.testing.assert_allclose(W[r], expected_cohort(), atol=1e-12)
    assert abs(W[T0].sum()) < 1e-12           # dollar-neutral
    assert np.abs(W).sum(axis=1).max() <= 1.0 + 1e-12


def test_hold_expiry_and_hold_param():
    T = 175
    data = make_view(T=T)
    data.extras["basis"][161] = np.nan        # missing basis on 161: no cohort that week
    W1 = target_weights(data, {"hold_weeks": 1, "lookback_days": 30})
    np.testing.assert_allclose(W1[160], expected_cohort(), atol=1e-12)
    assert np.all(W1[161:168] == 0.0)         # removed after 7 days, nothing new at 161
    np.testing.assert_allclose(W1[168], expected_cohort(), atol=1e-12)   # cohort of 168
    W2 = target_weights(data, {"hold_weeks": 2, "lookback_days": 30})
    np.testing.assert_allclose(W2[167], expected_cohort(), atol=1e-12)


def test_universe_exit_and_cohort_average():
    T = 170
    uni = np.ones((T, 12), dtype=bool)
    uni[158:165, 3] = False                   # C03 leaves the universe on 158, comes back on 165
    data = make_view(T=T, universe=uni)
    W = target_weights(data, {"hold_weeks": 2, "lookback_days": 30})
    np.testing.assert_allclose(W[157], expected_cohort(), atol=1e-12)
    w1 = expected_cohort()
    w1[3] = 0.0                               # exited, others not rescaled
    np.testing.assert_allclose(W[158], w1, atol=1e-12)
    # cohort of 161 without C03: low half C01, C02, C04, C05, C06 -> short C02, long C05
    w2 = np.zeros(12)
    w2[5] = w2[8] = 0.25
    w2[2] = w2[7] = -0.25
    np.testing.assert_allclose(W[161], (w1 + w2) / 2.0, atol=1e-12)
    np.testing.assert_allclose(W[166], (w1 + w2) / 2.0, atol=1e-12)   # no re-entry after return
    assert np.all(W[158:168, 3] == 0.0)


def test_fewer_than_ten_eligible():
    data = make_view()
    data.extras["basis"][T0, :3] = np.nan     # 9 eligible coins on 154
    W = target_weights(data, PARAMS)
    assert np.all(W == 0.0)


def test_lookback_param_changes_output():
    data = make_view(T=200)
    W30 = target_weights(data, {"hold_weeks": 26, "lookback_days": 30})
    W14 = target_weights(data, {"hold_weeks": 26, "lookback_days": 14})
    assert not np.allclose(W30, W14)


def test_cs_residual_by_hand():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 5.0, 8.0])
    # slope = sum(xm*ym)/sum(xm^2) = 11.5/5 = 2.3, intercept = 4.25 - 2.3*1.5 = 0.8
    np.testing.assert_allclose(_cs_residual(x, y), y - (0.8 + 2.3 * x), atol=1e-12)
    np.testing.assert_allclose(_cs_residual(np.ones(3), np.array([1.0, 2.0, 6.0])), [-2.0, -1.0, 3.0])


def test_rolling_median_and_fallback_universe():
    x = np.array([[1.0], [2.0], [np.nan], [4.0], [10.0]])
    np.testing.assert_allclose(_rolling_median(x, 3)[:, 0], [1.0, 1.5, 1.5, 3.0, 7.0])
    names = tuple("C%02dUSDT.P" % j for j in range(35)) + ("BTCUSDT.P",)
    T = 50
    dv = np.tile(np.arange(36, dtype=float)[None, :] + 1.0, (T, 1))
    dv[30:, 0] = 1e12                         # C00 jumps to the top on row 30
    data = make_view(T=T, universe=False, names=names, dollar_volume=dv,
                     basis=np.zeros((T, 36)), close=np.ones((T, 36)))
    uni = _universe(data)
    assert np.all(uni.sum(axis=1) == 30)
    assert uni[0, 35] and not uni[0, 0] and not uni[0, 5] and uni[0, 6]
    assert not uni[43, 0] and uni[43, 6]      # rows 14..43: 14 of 30 at 1e12, median still 1
    assert uni[44, 0] and not uni[44, 6]      # rows 15..44: 15 of 30, median (1 + 1e12) / 2


def test_short_views_and_no_universe():
    for T in (1, 5, 18, 40):
        data = make_view(T=T, universe=False)
        W = target_weights(data, PARAMS)
        assert W.shape == (T, 12)
        assert np.all(np.isfinite(W))


def test_truncation_point_in_time():
    data = make_view(T=175)
    W = target_weights(data, {"hold_weeks": 2, "lookback_days": 30})
    for cut in (150, 158, 165):
        d2 = make_view(T=cut)
        W2 = target_weights(d2, {"hold_weeks": 2, "lookback_days": 30})
        np.testing.assert_allclose(W2, W[:cut], atol=1e-12)


def test_formation_sides_match_cohort():
    # the diagnostic marks exactly these members with these sides (same code, copied)
    data = make_view()
    forms = _formations(data, PARAMS, _universe(data))
    assert len(forms) == 1
    t, members, weights, side = forms[0]
    assert t == T0
    got = dict(zip(members.tolist(), side.tolist()))
    assert got == {3: 1.0, 8: 1.0, 2: -1.0, 7: -1.0}
    np.testing.assert_allclose(weights, 0.25 * side)
