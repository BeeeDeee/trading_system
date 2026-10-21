"""Mechanics tests for H-0071 on a hand-made perp + spot panel.

Scenario (T = 96 rows): six alt perps A..F plus BTC and ETH perps, and the matching spot pairs. Base daily
return alternates +/-1 % (sigma ~ 1 %). On days 40, 50, 60, 70, 80 and 90 every alt moves +/-10 % (always a
2-sigma move: with three earlier 10 % moves in the 30-day window sigma is ~3.4 %). Perp volume is constant;
spot volume equals perp volume except on move days, where it is perp * 2^-x, so the venue ratio v is x log 2 on
move days and 0 otherwise; its trailing 30-day median is 0, so the abnormal ratio a = x log 2.
Days 40..80 give 30 big moves with x = +1 / -1 (15 each): pooled median m = 0 at day 90 (exactly 30 prior
big moves, the minimum). Day 90 (r and x per coin):
  A: r +10 %, x +0.5 -> leverage-led -> side -1
  B: r +10 %, x -0.5 -> spot-led     -> side +1
  C: r -10 %, x +0.5 -> leverage-led -> side +1
  D: r +10 %, x -0.5 -> spot-led     -> side +1
  E: r +10 %, x  0   -> a == m       -> no event
  F: r +10 %, spot volume 0          -> no event
Net = (-1 + 1 + 1 + 1) * 0.125 = 0.25 -> BTC hedge -0.25, gross 0.75.
"""
import numpy as np
import pytest

from lab.framework.data import DataView, synthetic
from strategy import target_weights, PARAMS
import strategy as S

ALTS = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT", "FFFUSDT"]
MAJORS = ["BTCUSDT", "ETHUSDT"]
T = 96
DAY90_R = [0.10, 0.10, -0.10, 0.10, 0.10, 0.10]
DAY90_X = [0.5, -0.5, 0.5, -0.5, 0.0, 0.0]


def make_view(n_rows=T, with_spot=True, with_btc=True, universe=None):
    perps = [a + ".P" for a in ALTS] + [m + ".P" for m in MAJORS if (with_btc or m != "BTCUSDT")]
    spots = (ALTS + [m for m in MAJORS if (with_btc or m != "BTCUSDT")]) if with_spot else []
    names = tuple(perps + spots)
    n_p, n = len(perps), len(perps) + len(spots)
    ret_co = np.zeros((T, n))
    base = np.where(np.arange(T) % 2 == 0, 0.01, -0.01)
    ret_co[:, :n_p] = base[:, None]
    dv = np.full((T, n), 1e6)
    for k, d in enumerate(range(40, 90, 10)):
        for c in range(6):
            ret_co[d, c] = 0.10 if k % 2 == 0 else -0.10
            x = 1.0 if (k + c) % 2 == 0 else -1.0
            if with_spot:
                dv[d, n_p + c] = 1e6 * 2.0 ** (-x)     # exact: a = +/- log 2, pooled median exactly 0
    for c in range(6):
        ret_co[90, c] = DAY90_R[c]
        if with_spot:
            dv[90, n_p + c] = 1e6 * 2.0 ** (-DAY90_X[c])
    if with_spot:
        dv[90, n_p + 5] = 0.0
    ret_oc = np.zeros((T, n))
    close = 10.0 * np.cumprod(1.0 + ret_co, axis=0)
    uni = np.zeros((T, n), dtype=bool)
    uni[:, :n_p] = True
    if universe is not None:
        uni = universe(uni, names)
    sl = slice(0, n_rows)
    return DataView(
        dates=synthetic().dates[:T][sl],
        instruments=names,
        asset_class=tuple(["crypto_perp"] * n_p + ["crypto_spot"] * len(spots)),
        ret_co=ret_co[sl],
        ret_oc=ret_oc[sl],
        tradable=np.ones((T, n), dtype=bool)[sl],
        listed=np.ones((T, n), dtype=bool)[sl],
        delisting=np.zeros((T, n), dtype=bool)[sl],
        close=close[sl],
        dollar_volume=dv[sl],
        universe=uni[sl],
        extras={},
        series={},
        cash_ret=np.zeros(T)[sl],
    )


def test_events_on_known_rows():
    v = make_view()
    mask, side, uni, hedge = S.compute_events(v, PARAMS)
    assert mask[:90].sum() == 0          # fewer than 30 prior big moves before day 90
    assert mask[90].sum() == 4
    assert list(mask[90, :6]) == [True, True, True, True, False, False]
    assert list(side[90, :4]) == [-1.0, 1.0, 1.0, 1.0]
    assert hedge == v.col("BTCUSDT.P")
    assert not mask[:, v.col("BTCUSDT.P")].any() and not mask[:, v.col("ETHUSDT.P")].any()


def test_weights_on_known_rows():
    v = make_view()
    w = target_weights(v, PARAMS)
    assert w.shape == (T, len(v.instruments))
    assert np.all(w[:90] == 0)
    exp = np.zeros(len(v.instruments))
    exp[:4] = [-0.125, 0.125, 0.125, 0.125]
    exp[v.col("BTCUSDT.P")] = -0.25
    for t in (90, 91, 92):              # hold_days = 3: rows t..t+2, closed at the open of t+4
        assert np.allclose(w[t], exp)
    assert np.all(w[93:] == 0)
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)
    assert np.allclose(w.sum(axis=1), 0.0)  # dollar-neutral
    assert np.all(w[:, 8:] == 0)        # spot pairs never held


def test_hold_days_param():
    v = make_view()
    w2 = target_weights(v, dict(PARAMS, hold_days=2))
    assert np.any(w2[91] != 0) and np.all(w2[92] == 0)
    w5 = target_weights(v, dict(PARAMS, hold_days=5))
    assert np.any(w5[94] != 0) and np.all(w5[95] == 0)


def test_k_sigma_param():
    v = make_view()
    # day-90 move ~9.5 % log against sigma ~3.4 %: big at 2.0 and 2.5, not at 3.0
    assert S.compute_events(v, dict(PARAMS, k_sigma=2.5))[0][90].sum() == 4
    assert S.compute_events(v, dict(PARAMS, k_sigma=3.0))[0].sum() == 0


def test_universe_exit_closes_position():
    def drop_a(uni, names):
        uni = uni.copy()
        uni[91, names.index("AAAUSDT.P")] = False
        return uni
    v = make_view(universe=drop_a)
    w = target_weights(v, PARAMS)
    a = v.col("AAAUSDT.P")
    assert w[90, a] == -0.125
    assert w[91, a] == 0.0 and w[92, a] == 0.0   # stays closed after re-entering
    assert np.isclose(w[91, v.col("BTCUSDT.P")], -0.375)


def test_point_in_time_truncation():
    full = target_weights(make_view(), PARAMS)
    for cut in (60, 90, 91, 93):
        part = target_weights(make_view(n_rows=cut + 1), PARAMS)
        assert np.array_equal(part, full[: cut + 1])


def test_missing_inputs_raise():
    with pytest.raises(KeyError):
        target_weights(make_view(with_spot=False), PARAMS)
    with pytest.raises(Exception):
        target_weights(make_view(with_btc=False), PARAMS)


def test_replacement_and_gross_scaling(monkeypatch):
    v = make_view()
    n = len(v.instruments)
    mask = np.zeros((T, n), dtype=bool)
    side = np.zeros((T, n))
    uni = np.zeros((T, n), dtype=bool)
    uni[:, :8] = True
    mask[10, :6] = True
    side[10, :6] = 1.0                  # 6 longs: 0.75 + hedge 0.75 = 1.5 -> scaled by 1/1.5
    mask[11, 0] = True
    side[11, 0] = -1.0                  # A flips, hold restarts
    monkeypatch.setattr(S, "compute_events", lambda d, p: (mask, side, uni, 6))
    w = target_weights(v, PARAMS)
    assert np.allclose(w[10, :6], 0.125 / 1.5) and np.isclose(w[10, 6], -0.75 / 1.5)
    assert np.isclose(np.abs(w[10]).sum(), 1.0)
    # row 11: A short, B..F long: net 0.5, gross 0.75 + 0.5 = 1.25
    assert np.isclose(w[11, 0], -0.125 / 1.25) and np.isclose(w[11, 6], -0.5 / 1.25)
    assert np.all(w[13, 1:6] == 0) and w[13, 0] != 0     # B..F end after row 12, A after row 13
    assert np.all(w[14] == 0)


def test_lab_synthetic_runs():
    v = synthetic()
    w = target_weights(v, PARAMS)
    assert w.shape == (len(v.dates), len(v.instruments))
    assert np.all(np.isfinite(w))
