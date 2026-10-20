import dataclasses
import typing

import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import _dedupe, _window_sum, _sides_by_bin, build_weights, event_table, SLOT


@dataclasses.dataclass
class FakeData:
    ret_co: np.ndarray
    ret_oc: np.ndarray
    dollar_volume: np.ndarray
    tradable: np.ndarray
    listed: np.ndarray
    universe: typing.Any
    extras: dict


def make_data(T=420, N=40, seed=0, nan_rows=0):
    rng = np.random.default_rng(seed)
    ret_co = rng.normal(0, 0.005, (T, N))
    ret_oc = rng.normal(0, 0.015, (T, N))
    dv = np.exp(rng.normal(17, 0.4, (T, N)))
    ea = np.zeros((T, N))
    for j in range(N):
        start = int(rng.integers(0, 9))
        ea[start::9, j] = 1.0
    ea[:nan_rows] = np.nan
    liq = np.tile(np.arange(1, N + 1, dtype=float), (T, 1))
    return FakeData(ret_co=ret_co, ret_oc=ret_oc, dollar_volume=dv,
                    tradable=np.ones((T, N), bool), listed=np.ones((T, N), bool),
                    universe=np.ones((T, N), bool),
                    extras={"ea_reaction_day": ea, "liq_rank": liq})


# ---------------------------------------------------------------- helpers

def test_dedupe_ignores_second_filing_within_hold():
    rt = np.array([10, 12, 16, 17, 30, 11])
    rj = np.array([0, 0, 0, 0, 0, 1])
    keep = _dedupe(rt, rj, 5)
    # 12 is 2 after 10 (blocked), 16 is 6 after 10 (kept), 17 is 1 after 16 (blocked), 30 kept, other stock kept
    assert keep.tolist() == [True, False, True, False, True, True]
    keep3 = _dedupe(rt, rj, 3)
    assert keep3.tolist() == [True, False, True, False, True, True]
    keep7 = _dedupe(rt, rj, 7)
    assert keep7.tolist() == [True, False, False, False, True, True]   # 17 is 7 after 10: blocked


def test_window_sum_covers_t68_to_t6():
    T, N = 200, 2
    x = np.tile(np.arange(T, dtype=float)[:, None], (1, N))
    got = _window_sum(x, np.array([100, 10, 3]), np.array([1, 0, 0]))
    assert got[0] == pytest.approx(sum(range(32, 95)))      # rows 32..94: 63 sessions
    assert got[1] == pytest.approx(sum(range(0, 5)))        # clipped at row 0: rows 0..4
    assert got[2] == 0.0                                     # no row before t-6


def test_sides_by_bin_quantiles_and_min_bin():
    # pool: 60 candidates at rows 100..159, all same |z| and positive sign -> one bin, AV = 0..59
    ct = list(range(100, 160)) + [300, 300, 300, 300]
    absz = [1.0] * 60 + [1.0, 1.0, 1.0, 1.0]
    pos = [1] * 60 + [1, 1, 1, 0]
    av = list(range(60)) + [50.0, 10.0, 30.0, 50.0]
    side = _sides_by_bin(np.array(ct), np.array(absz), np.array(pos), np.array(av, float), 0, 0.3)
    # 0.7 quantile of 0..59 = 41.3, 0.3 quantile = 17.7
    assert side[-4:].tolist() == [1, -1, 0, 0]               # last one: negative-z bin is empty
    assert (side[:60] == 0).all()                            # pool rows have no full pool before row 252
    side2 = _sides_by_bin(np.array(ct), np.array(absz), np.array(pos), np.array(av, float), 0, 0.2)
    assert side2[-4:].tolist() == [1, -1, 0, 0]
    side4 = _sides_by_bin(np.array(ct), np.array(absz), np.array(pos), np.array(av, float), 0, 0.4)
    # with q=0.4: 0.6 quantile = 35.4, 0.4 quantile = 23.6 -> AV 30 is still neutral, 50 long, 10 short
    assert side4[-4:].tolist() == [1, -1, 0, 0]
    # fewer than 30 pool candidates in the bin -> no event
    ct_s = list(range(100, 129)) + [300]
    side5 = _sides_by_bin(np.array(ct_s), np.ones(30), np.ones(30, np.int64),
                          np.array(list(range(29)) + [50.0]), 0, 0.3)
    assert side5[-1] == 0
    # 8-K data starting later: no events until the pool covers 252 sessions
    side6 = _sides_by_bin(np.array(ct), np.array(absz), np.array(pos), np.array(av, float), 60, 0.3)
    assert side6[-4:].tolist() == [0, 0, 0, 0]


def test_sides_by_bin_terciles_split_sign_and_size():
    # pool: 90 positive-z candidates, |z| = 1..90 -> terciles (<=30.67, <=60.33, above)
    ct = np.array(list(range(100, 190)) + [300, 300])
    absz = np.array(list(range(1, 91)) + [5.0, 85.0], float)
    pos = np.ones(92, np.int64)
    # AV high in the low tercile, low in the high tercile
    av = np.array([100.0 if z <= 30 else (0.0 if z > 60 else 50.0) for z in range(1, 91)] + [100.0, 100.0])
    side = _sides_by_bin(ct, absz, pos, av, 0, 0.3)
    # candidate 1 sits in the low tercile where every pool AV = 100: >= and <= both hold -> ambiguous, no event
    assert side[-2] == 0
    # candidate 2 sits in the high tercile where pool AV = 0: AV 100 is long
    assert side[-1] == 1


# ---------------------------------------------------------------- positions

def _ones(T, N):
    return np.ones((T, N), bool)


def test_slot_held_hold_days_and_small_net_unhedged():
    T, N = 20, 60
    liq = np.tile(np.arange(1, N + 1, dtype=float), (T, 1))
    W = build_weights(T, N, np.array([2]), np.array([55]), np.array([1], np.int8), _ones(T, N), _ones(T, N), liq, 5)
    assert np.allclose(W[2:7, 55], SLOT)
    assert np.allclose(W[:2], 0) and np.allclose(W[7:], 0)
    assert np.allclose(np.delete(W, 55, axis=1), 0)           # net 1/30 < 5 % band: no hedge
    W3 = build_weights(T, N, np.array([2]), np.array([55]), np.array([1], np.int8), _ones(T, N), _ones(T, N), liq, 3)
    assert np.allclose(W3[2:5, 55], SLOT) and np.allclose(W3[5:, 55], 0)


def test_hedge_basket_top50_excludes_slot_names():
    T, N = 20, 60
    liq = np.tile(np.arange(1, N + 1, dtype=float), (T, 1))
    # two longs: one on col 0 (most liquid) and one on col 55 -> net 2/30 > 5 %
    W = build_weights(T, N, np.array([2, 2]), np.array([0, 55]), np.array([1, 1], np.int8),
                      _ones(T, N), _ones(T, N), liq, 5)
    hedge_cols = list(range(1, 51))
    assert np.allclose(W[2, hedge_cols], -2 * SLOT / 50)
    assert W[2, 0] == pytest.approx(SLOT) and W[2, 55] == pytest.approx(SLOT)
    assert np.allclose(W[2, 51:55], 0)
    assert W[2].sum() == pytest.approx(0.0)
    # held unchanged while slots are open, all closed with the slots (book in cash)
    assert np.allclose(W[6], W[2])
    assert np.allclose(W[7], 0)


def test_universe_exit_closes_slot():
    T, N = 20, 60
    liq = np.tile(np.arange(1, N + 1, dtype=float), (T, 1))
    U = _ones(T, N)
    U[4:, 55] = False
    trad = _ones(T, N)
    trad[3, 56] = False
    W = build_weights(T, N, np.array([2, 2]), np.array([55, 56]), np.array([1, -1], np.int8), U, trad, liq, 5)
    assert W[3, 55] == pytest.approx(SLOT) and W[4, 55] == 0
    assert W[2, 56] == pytest.approx(-SLOT) and np.allclose(W[3:, 56], 0)


def test_gross_cap_scales_new_slots():
    T, N = 10, 200
    liq = np.tile(np.arange(1, N + 1, dtype=float), (T, 1))
    cols = np.arange(100, 140)
    W = build_weights(T, N, np.full(40, 2), cols, np.ones(40, np.int8), _ones(T, N), _ones(T, N), liq, 5)
    # 40 longs: s = 1 / (2 * 40/30) = 0.375; hedge equals minus the net
    assert np.allclose(W[2, cols], 0.375 * SLOT, rtol=1e-6)
    assert np.abs(W[2]).sum() == pytest.approx(1.0, abs=1e-6)
    assert W[2].sum() == pytest.approx(0.0, abs=1e-9)
    # 20 long + 20 short: net 0, s = 0.75, no hedge
    sides = np.array([1] * 20 + [-1] * 20, np.int8)
    W2 = build_weights(T, N, np.full(40, 2), cols, sides, _ones(T, N), _ones(T, N), liq, 5)
    assert np.allclose(np.abs(W2[2, cols]), 0.75 * SLOT, rtol=1e-6)
    assert np.allclose(W2[2, :100], 0)


def test_existing_slots_not_resized():
    T, N = 12, 200
    liq = np.tile(np.arange(1, N + 1, dtype=float), (T, 1))
    et = np.array([2] * 20 + [3] * 20)
    cols = np.arange(100, 140)
    sides = np.array([1, -1] * 20, np.int8)
    W = build_weights(T, N, et, cols, sides, _ones(T, N), _ones(T, N), liq, 5)
    assert np.allclose(np.abs(W[2, 100:120]), SLOT)
    assert np.allclose(np.abs(W[3, 100:120]), SLOT)            # day-2 slots unchanged
    assert np.abs(W[3]).sum() <= 1.0 + 1e-9
    assert np.allclose(np.abs(W[3, 120:140]), 0.5 * SLOT)     # day-3 slots fill the remaining 1/3


# ---------------------------------------------------------------- full pipeline

def test_full_pipeline_shape_gross_determinism():
    d = make_data()
    W = target_weights(d, PARAMS)
    T, N = d.universe.shape
    assert W.shape == (T, N)
    assert np.all(np.abs(W).sum(axis=1) <= 1.0 + 1e-9)
    assert np.allclose(W[:252], 0)                            # pool not yet covering 252 sessions
    assert np.abs(W[252:]).sum() > 0
    assert np.array_equal(W, target_weights(d, PARAMS))


def test_point_in_time_truncation():
    d = make_data()
    W = target_weights(d, PARAMS)
    for cut in (270, 330, 400):
        dt = FakeData(ret_co=d.ret_co[:cut], ret_oc=d.ret_oc[:cut], dollar_volume=d.dollar_volume[:cut],
                      tradable=d.tradable[:cut], listed=d.listed[:cut], universe=d.universe[:cut],
                      extras={k: v[:cut] for k, v in d.extras.items()})
        Wt = target_weights(dt, PARAMS)
        assert np.array_equal(Wt, W[:cut])


def test_late_8k_start_delays_events():
    d = make_data(nan_rows=60)
    et, _, _ = event_table(d, PARAMS)
    assert et.min() >= 60 + 252


def test_params_change_output():
    d = make_data()
    W = target_weights(d, PARAMS)
    for k, v in (("hold_days", 3), ("hold_days", 7), ("leg_quantile", 0.2), ("leg_quantile", 0.4)):
        p = dict(PARAMS)
        p[k] = v
        assert not np.array_equal(W, target_weights(d, p)), k


def test_missing_input_raises():
    d = make_data()
    del d.extras["ea_reaction_day"]
    with pytest.raises(KeyError):
        target_weights(d, PARAMS)


def test_nan_flag_and_short_history_give_no_event():
    d = make_data()
    d.extras["ea_reaction_day"][:, 3] = np.nan
    d.dollar_volume[:, 4] = np.nan
    d.dollar_volume[300:, 4] = 1e7                            # stock 4: only from row 300 -> < 50 valid sessions until 355+
    et, ej, _ = event_table(d, PARAMS)
    assert not (ej == 3).any()
    assert not ((ej == 4) & (et < 300 + 56)).any()
