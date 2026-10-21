import dataclasses
from typing import Any, Optional

import numpy as np
import pytest

from strategy import target_weights, PARAMS, compute_signal


@dataclasses.dataclass
class FakeView:
    dates: Any
    instruments: tuple
    ret_co: Any
    ret_oc: Any
    tradable: Any
    listed: Any
    close: Any
    universe: Optional[Any]
    extras: dict

    def col(self, name):
        return self.instruments.index(name)


def make_view(T, N, base=0.01):
    """All stocks: r[t] = base * (-1)^t (close-to-close, all in ret_co)."""
    sign = np.where(np.arange(T) % 2 == 0, 1.0, -1.0)
    r = np.repeat((base * sign)[:, None], N, axis=1)
    dates = np.arange(T).astype("datetime64[D]")
    names = tuple("E%d" % (j + 1) for j in range(N))
    liq = np.repeat((np.arange(N) + 1.0)[None, :], T, axis=0)
    return FakeView(
        dates=dates,
        instruments=names,
        ret_co=r.copy(),
        ret_oc=np.zeros((T, N)),
        tradable=np.ones((T, N), dtype=bool),
        listed=np.ones((T, N), dtype=bool),
        close=np.full((T, N), 10.0),
        universe=np.ones((T, N), dtype=bool),
        extras={"liq_rank": liq},
    )


P6 = {"k_sigma": 3.0, "lookback_days": 6}
SD_ALT = np.sqrt(6 * 0.01 ** 2 / 5)  # sample std of six alternating +-0.01


def test_params_match_card():
    assert PARAMS == {"k_sigma": 3.0, "lookback_days": 60}


def test_up_and_down_movers_both_shorted_and_hedged():
    v = make_view(14, 12)
    v.ret_co[10, 0] = 0.10    # up-move
    v.ret_co[10, 1] = -0.10   # down-move
    w = target_weights(v, P6)
    z, elig, ev = compute_signal(v, P6)
    # z uses the median (0.01) and sigma of rows 4..9 (day t excluded)
    assert z[10, 0] == pytest.approx((0.10 - 0.01) / SD_ALT)
    assert z[10, 1] == pytest.approx((-0.10 - 0.01) / SD_ALT)
    assert ev[10, 0] and ev[10, 1] and ev[10].sum() == 2
    assert w[10, 0] == pytest.approx(-0.025)
    assert w[10, 1] == pytest.approx(-0.025)
    # hedge: the 10 other members, total 0.05
    assert np.allclose(w[10, 2:], 0.05 / 10)
    assert w[10].sum() == pytest.approx(0.0)
    # rows without events are flat
    other = np.ones(14, dtype=bool)
    other[10] = False
    assert np.all(w[other] == 0.0)
    assert not np.isnan(w).any()


def test_cap_twenty_by_abs_z_and_hedge_best_liq_rank():
    N = 130
    v = make_view(14, N)
    for j in range(25):
        s = 1.0 if j % 2 == 0 else -1.0
        v.ret_co[10, j] = 0.01 + s * (0.04 + 0.002 * j)
    w = target_weights(v, P6)
    _, _, ev = compute_signal(v, P6)
    assert ev[10, :25].all() and ev[10].sum() == 25
    # kept: the 20 largest |z| -> stocks 5..24
    assert np.all(w[10, 5:25] == pytest.approx(-0.025))
    assert np.all(w[10, :5] == 0.0)
    # hedge: best liq_rank among non-events = stocks 25..124, total 0.5
    assert np.allclose(w[10, 25:125], 0.5 / 100)
    assert np.all(w[10, 125:] == 0.0)
    assert np.abs(w[10]).sum() == pytest.approx(1.0)
    assert w[10].sum() == pytest.approx(0.0)


def test_hedge_uses_liq_rank_not_column_order():
    N = 130
    v = make_view(14, N)
    v.extras["liq_rank"] = np.repeat((N - np.arange(N, dtype=float))[None, :], 14, axis=0)
    v.ret_co[10, 129] = 0.10   # the most liquid stock is the event
    w = target_weights(v, P6)
    assert w[10, 129] == pytest.approx(-0.025)
    assert np.allclose(w[10, 29:129], 0.025 / 100)
    assert np.all(w[10, :29] == 0.0)


def test_nan_liq_rank_not_a_hedge():
    v = make_view(14, 12)
    v.ret_co[10, 0] = 0.10
    v.extras["liq_rank"][10, 5] = np.nan
    w = target_weights(v, P6)
    assert w[10, 5] == 0.0
    assert np.allclose(np.delete(w[10], [0, 5]), 0.025 / 10)


def test_short_history_late_listing_not_event_nor_hedge():
    v = make_view(14, 12)
    v.close[:8, 3] = np.nan
    v.listed[:8, 3] = False
    v.tradable[:8, 3] = False
    v.ret_co[:8, 3] = 0.0
    v.ret_co[10, 3] = 0.20
    v.ret_co[10, 0] = 0.10
    _, elig, ev = compute_signal(v, P6)
    # valid returns of stock 3 in rows 4..9: rows 9 only (row 8 has no previous price) -> 1 < 4
    assert not elig[10, 3] and not ev[10, 3]
    w = target_weights(v, P6)
    assert w[10, 3] == 0.0
    assert w[10, 0] == pytest.approx(-0.025)
    # later, with 4 valid returns in rows 7..12 (9..12), stock 3 becomes eligible
    assert elig[13, 3]


def test_first_rows_before_history_are_flat():
    v = make_view(14, 12)
    v.ret_co[3, 0] = 0.30   # rows 1..2 valid only -> 2 < 4
    w = target_weights(v, P6)
    _, elig, _ = compute_signal(v, P6)
    assert not elig[:5].any()   # row 0 has no previous close; row 4 has 3 valid (1..3)
    assert elig[5].all()
    assert np.all(w[:5] == 0.0)


def test_non_member_and_non_tradable_excluded():
    v = make_view(14, 12)
    v.ret_co[10, 0] = 0.10
    v.ret_co[10, 1] = 0.10
    v.ret_co[10, 2] = 0.10
    v.universe[10, 1] = False
    v.tradable[10, 2] = False
    v.universe[10, 7] = False
    w = target_weights(v, P6)
    _, _, ev = compute_signal(v, P6)
    assert ev[10, 0] and not ev[10, 1] and not ev[10, 2]
    assert w[10, 1] == 0.0 and w[10, 2] == 0.0 and w[10, 7] == 0.0
    assert w[10, 0] == pytest.approx(-0.025)
    hedge = [j for j in range(12) if j not in (0, 1, 2, 7)]
    assert np.allclose(w[10, hedge], 0.025 / len(hedge))


def test_missing_universe_raises():
    v = make_view(14, 12)
    v.universe = None
    with pytest.raises(ValueError):
        target_weights(v, P6)


def test_missing_liq_rank_raises():
    v = make_view(14, 12)
    v.extras = {}
    with pytest.raises(KeyError):
        target_weights(v, P6)


def test_k_sigma_changes_output():
    v = make_view(14, 12)
    v.ret_co[10, 0] = 0.05   # z = 0.04 / 0.010954 = 3.65
    w3 = target_weights(v, {"k_sigma": 3.0, "lookback_days": 6})
    w4 = target_weights(v, {"k_sigma": 4.0, "lookback_days": 6})
    assert w3[10, 0] == pytest.approx(-0.025)
    assert np.all(w4 == 0.0)


def test_lookback_changes_output():
    v = make_view(30, 12)
    v.ret_co[20, 0] = 0.10    # stock 0: big move on row 20, z on row 24 depends on the window
    v.ret_co[24, 0] = 0.06
    wa = target_weights(v, {"k_sigma": 3.0, "lookback_days": 6})
    wb = target_weights(v, {"k_sigma": 3.0, "lookback_days": 3})
    # lookback 6: window 18..23 contains the 0.10 spike -> large sigma -> no event
    # lookback 3: window 21..23 without the spike -> event
    assert wa[24, 0] == 0.0
    assert wb[24, 0] == pytest.approx(-0.025)


def test_ret_co_and_ret_oc_compound():
    v = make_view(14, 12)
    v.ret_co[10, 0] = 0.05
    v.ret_oc[10, 0] = 0.05     # r = 0.1025
    z, _, _ = compute_signal(v, P6)
    assert z[10, 0] == pytest.approx((1.05 * 1.05 - 1 - 0.01) / SD_ALT)


def _random_view(T, N, seed):
    rng = np.random.default_rng(seed)
    v = make_view(T, N)
    v.ret_co = rng.standard_t(3, size=(T, N)) * 0.01
    v.ret_oc = rng.standard_t(3, size=(T, N)) * 0.01
    v.universe = rng.random((T, N)) < 0.8
    v.tradable = rng.random((T, N)) < 0.97
    v.extras = {"liq_rank": rng.permuted(np.tile(np.arange(1.0, N + 1), (T, 1)), axis=1)}
    return v


def test_point_in_time_truncation_and_perturbation():
    T, N, cut = 300, 150, 170
    v = _random_view(T, N, 7)
    w = target_weights(v, PARAMS)
    assert (w < 0).sum() > 0
    vt = FakeView(dates=v.dates[:cut], instruments=v.instruments, ret_co=v.ret_co[:cut],
                  ret_oc=v.ret_oc[:cut], tradable=v.tradable[:cut], listed=v.listed[:cut],
                  close=v.close[:cut], universe=v.universe[:cut],
                  extras={"liq_rank": v.extras["liq_rank"][:cut]})
    assert np.array_equal(w[:cut], target_weights(vt, PARAMS))
    vp = _random_view(T, N, 7)
    rng = np.random.default_rng(99)
    vp.ret_co[cut:] = rng.normal(0, 0.05, size=(T - cut, N))
    vp.extras["liq_rank"][cut:] = rng.random((T - cut, N)) * N
    assert np.array_equal(w[:cut], target_weights(vp, PARAMS)[:cut])


def test_deterministic_and_gross():
    v = _random_view(300, 150, 3)
    w1 = target_weights(v, PARAMS)
    w2 = target_weights(v, PARAMS)
    assert np.array_equal(w1, w2)
    assert not np.isnan(w1).any()
    assert np.all(np.abs(w1).sum(axis=1) <= 1.0 + 1e-9)
    assert np.all(w1.min(axis=1) >= -0.025 - 1e-12)
    assert np.allclose(w1.sum(axis=1), 0.0, atol=1e-9)   # dollar neutral
    assert np.all((w1 < 0).sum(axis=1) <= 20)
    assert np.all((w1 > 0).sum(axis=1) <= 100)
    # only members hold weight
    assert not np.any((w1 != 0) & ~v.universe)
