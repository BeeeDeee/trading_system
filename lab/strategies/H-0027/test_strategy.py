import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

N = 150
T = 20


def make_view(T=T, N=N, ncf=None, sbc=None, mcap=None, ea=None, universe=None, tradable=None):
    dates = (np.arange(T) + 12000).astype("datetime64[D]")
    instruments = tuple("X%03d" % i for i in range(N))
    if ncf is None:
        # net_flow = -ncf / mcap = i / 1000: stock i ranks i-th
        ncf = np.tile(-np.arange(N, dtype=np.float64), (T, 1))
    if sbc is None:
        sbc = np.zeros((T, N))
    if mcap is None:
        mcap = np.full((T, N), 1000.0)
    if ea is None:
        ea = np.zeros((T, N))
    if universe is None:
        universe = np.ones((T, N), dtype=bool)
    if tradable is None:
        tradable = np.ones((T, N), dtype=bool)
    extras = {
        "sf1_art_ncfcommon": np.asarray(ncf, dtype=np.float32),
        "sf1_art_sbcomp": np.asarray(sbc, dtype=np.float32),
        "sf1_art_marketcap": np.asarray(mcap, dtype=np.float32),
        "ea_reaction_day": np.asarray(ea, dtype=np.float32),
        "liq_rank": np.tile(np.arange(1, N + 1, dtype=np.float32), (T, 1)),
    }
    return DataView(
        dates=dates,
        instruments=instruments,
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=np.zeros((T, N)),
        ret_oc=np.zeros((T, N)),
        tradable=tradable,
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=np.full((T, N), 10.0),
        dollar_volume=np.full((T, N), 1e8),
        universe=universe,
        extras=extras,
        series={},
        cash_ret=np.zeros(T),
    )


def ea_with(events):
    ea = np.zeros((T, N))
    for t, j in events:
        ea[t, j] = 1.0
    return ea


def test_long_and_short_event_hold_window():
    # q=0.2 over scores 0..149: lower = 29.8/1000, upper = 119.2/1000
    d = make_view(ea=ea_with([(2, 140), (2, 5), (2, 70)]))
    w = target_weights(d, PARAMS)
    assert w.shape == (T, N)
    assert np.all(w[:2] == 0)
    for t in range(2, 7):  # decision rows 2..6 -> held 5 sessions
        assert w[t, 140] == pytest.approx(0.05)
        assert w[t, 5] == pytest.approx(-0.05)
        assert w[t, 70] == 0.0          # middle stock is no event
        assert np.abs(w[t]).sum() == pytest.approx(0.10)  # net 0 -> no basket
    assert np.all(w[7:] == 0)


def test_threshold_boundaries():
    # stock 29 (29.8 lower) is short, 30 is not; 120 (>= 119.2) long, 119 not
    d = make_view(ea=ea_with([(3, 29), (3, 30), (3, 119), (3, 120)]))
    w = target_weights(d, PARAMS)
    assert w[3, 29] < 0 and w[3, 120] > 0
    assert w[3, 30] == w[3, 119]       # both only in the basket (net 0 -> 0)
    assert w[3, 30] == 0.0


def test_basket_offsets_net():
    d = make_view(ea=ea_with([(1, 145)]))
    w = target_weights(d, PARAMS)
    assert w[1, 145] == pytest.approx(0.05)
    others = np.delete(w[1], 145)
    assert np.allclose(others, -0.05 / (N - 1))
    assert w[1].sum() == pytest.approx(0.0, abs=1e-12)


def test_leg_weights_and_gross():
    longs = list(range(120, 150))   # 30 longs -> 0.5/30 each
    d = make_view(ea=ea_with([(4, j) for j in longs] + [(4, 0)]))
    w = target_weights(d, PARAMS)
    assert np.allclose(w[4, longs], 0.5 / 30)
    assert w[4, 0] == pytest.approx(-0.05)
    net = 0.5 - 0.05
    assert np.allclose(w[4, 1:120], -net / 119)
    assert np.abs(w).sum(axis=1).max() <= 1.0 + 1e-9
    assert w[4].sum() == pytest.approx(0.0, abs=1e-12)


def test_restart_and_side_flip():
    ncf = np.tile(-np.arange(N, dtype=np.float64), (T, 1))
    ncf[5:, 140] = 500.0     # new filing: big issuer -> bottom leg
    d = make_view(ncf=ncf, ea=ea_with([(2, 140), (5, 140)]))
    w = target_weights(d, PARAMS)
    assert w[2, 140] > 0 and w[4, 140] > 0
    for t in range(5, 10):
        assert w[t, 140] == pytest.approx(-0.05)
    assert w[10, 140] == 0.0


def test_restart_same_side_extends():
    d = make_view(ea=ea_with([(2, 140), (5, 140)]))
    w = target_weights(d, PARAMS)
    assert np.all(w[2:10, 140] > 0)
    assert np.all(w[10:] == 0)


def test_not_tradable_closes_position():
    tradable = np.ones((T, N), dtype=bool)
    tradable[4, 140] = False
    d = make_view(ea=ea_with([(2, 140)]), tradable=tradable)
    w = target_weights(d, PARAMS)
    assert w[3, 140] > 0
    assert np.all(w[4:, 140] == 0)
    assert np.all(w[4:] == 0)


def test_leaving_universe_closes_and_nonmember_no_event():
    uni = np.ones((T, N), dtype=bool)
    uni[4:, 140] = False
    uni[:, 3] = False
    d = make_view(ea=ea_with([(2, 140), (2, 3)]), universe=uni)
    w = target_weights(d, PARAMS)
    assert w[2, 140] > 0
    assert np.all(w[:, 3] == 0)
    assert np.all(w[4:, 140] == 0)


def test_fewer_than_100_scored_no_event():
    ncf = np.tile(-np.arange(N, dtype=np.float64), (T, 1))
    ncf[:, :51] = np.nan            # 99 scored
    d = make_view(ncf=ncf, ea=ea_with([(2, 140)]))
    assert np.all(target_weights(d, PARAMS) == 0)
    ncf[:, 50] = -50.0              # 100 scored
    d = make_view(ncf=ncf, ea=ea_with([(2, 140)]))
    assert target_weights(d, PARAMS)[2, 140] > 0


def test_missing_inputs_and_sbcomp():
    mcap = np.full((T, N), 1000.0)
    mcap[:, 145] = 0.0              # no score
    mcap[:, 146] = np.nan           # no score
    sbc = np.zeros((T, N))
    sbc[:, 147] = np.nan            # treated as 0 -> still top
    sbc[:, 148] = 1000.0            # score = (-(-148) - 1000)/1000 < 0 -> bottom leg
    d = make_view(mcap=mcap, sbc=sbc, ea=ea_with([(2, 145), (2, 146), (2, 147), (2, 148)]))
    w = target_weights(d, PARAMS)
    assert w[2, 147] == pytest.approx(0.05)
    assert w[2, 148] == pytest.approx(-0.05)
    assert w[2, 145] == 0.0 and w[2, 146] == 0.0


def test_nan_ea_is_no_event():
    ea = np.full((T, N), np.nan)
    d = make_view(ea=ea)
    assert np.all(target_weights(d, PARAMS) == 0)


def test_params_change_output():
    d = make_view(ea=ea_with([(2, 140), (2, 35)]))
    base = target_weights(d, PARAMS)
    w3 = target_weights(d, dict(PARAMS, hold_days=3))
    assert base[5, 140] > 0 and w3[5, 140] == 0 and w3[4, 140] > 0
    assert base[2, 35] <= 0 and abs(base[2, 35]) < 0.05  # stock 35 not in 0.2 leg
    w03 = target_weights(d, dict(PARAMS, leg_quantile=0.3))
    assert w03[2, 35] == pytest.approx(-0.05)


def test_point_in_time_truncation():
    rng = np.random.default_rng(0)
    ea = (rng.random((T, N)) < 0.05).astype(float)
    ncf = rng.normal(size=(T, N)) * 100
    d = make_view(ea=ea, ncf=ncf)
    full = target_weights(d, PARAMS)
    for cut in (5, 11, 17):
        dc = make_view(T=cut, ea=ea[:cut], ncf=ncf[:cut])
        part = target_weights(dc, PARAMS)
        assert np.allclose(part, full[:cut])
    assert np.abs(full).sum(axis=1).max() <= 1.0 + 1e-9


def test_point_in_time_perturbation():
    rng = np.random.default_rng(1)
    ea = (rng.random((T, N)) < 0.05).astype(float)
    ncf = rng.normal(size=(T, N)) * 100
    full = target_weights(make_view(ea=ea, ncf=ncf), PARAMS)
    assert np.abs(full).sum() > 0
    for cut in (4, 9, 15):
        ea2 = ea.copy()
        ncf2 = ncf.copy()
        ea2[cut + 1:] = (rng.random((T - cut - 1, N)) < 0.3).astype(float)
        ncf2[cut + 1:] = rng.normal(size=(T - cut - 1, N)) * 100
        pert = target_weights(make_view(ea=ea2, ncf=ncf2), PARAMS)
        assert np.allclose(pert[:cut + 1], full[:cut + 1])
