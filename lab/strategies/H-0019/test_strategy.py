import numpy as np
import pytest

from strategy import target_weights, PARAMS, events

T, N = 20, 30


class Stub:
    """Minimal stand-in for DataView with the fields the strategy reads."""

    def __init__(self, universe, tradable, listed, extras):
        self.universe = universe
        self.tradable = tradable
        self.listed = listed
        self.extras = extras
        self.instruments = tuple("S%02d" % j for j in range(universe.shape[1]))

    def truncate(self, k):
        return Stub(self.universe[:k], self.tradable[:k], self.listed[:k],
                    {key: v[:k] for key, v in self.extras.items()})


def make():
    """All 30 stocks are members; filing changes on row 3 for all; F_j = 0.5 * j (rank = j)."""
    j = np.arange(N, dtype=np.float64)
    ncf = np.tile(-100.0 * j, (T, 1))
    sbc = np.zeros((T, N))
    mcap = np.full((T, N), 100.0)
    mcap[3:] = 200.0
    ones = np.ones((T, N), dtype=bool)
    return Stub(ones.copy(), ones.copy(), ones.copy(),
                {"sf1_art_ncfcommon": ncf.astype(np.float32), "sf1_art_sbcomp": sbc.astype(np.float32),
                 "sf1_art_marketcap": mcap.astype(np.float32)})


LONG = list(range(24, 30))   # pct (j+0.5)/30 >= 0.8
SHORT = list(range(0, 6))    # pct <= 0.2


def test_basic_book():
    w = target_weights(make(), PARAMS)
    assert w.shape == (T, N)
    for t in range(T):
        if 3 <= t <= 7:  # decided after close of 3, held opens 4..8, closed at open of 3+1+5 = 9
            assert np.allclose(w[t, LONG], 0.5 / 6)
            assert np.allclose(w[t, SHORT], -0.5 / 6)
            assert np.allclose(np.delete(w[t], LONG + SHORT), 0.0)
        else:
            assert np.allclose(w[t], 0.0)
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-9)
    assert np.allclose(w.sum(axis=1), 0.0)


def test_hold_days_param():
    p = dict(PARAMS, hold_days=3)
    w = target_weights(make(), p)
    assert np.all(np.abs(w[3:6]).sum(axis=1) > 0.99)
    assert np.allclose(w[6:], 0.0)


def test_quantile_param():
    p = dict(PARAMS, quantile=0.3)
    w = target_weights(make(), p)
    assert np.allclose(w[3, 21:30], 0.5 / 9)
    assert np.allclose(w[3, 0:9], -0.5 / 9)
    # quantile 0.1 -> 3 names per leg < 5 -> flat book
    w = target_weights(make(), dict(PARAMS, quantile=0.1))
    assert np.allclose(w, 0.0)


def test_nan_to_value_is_not_event():
    d = make()
    d.extras["sf1_art_marketcap"][:3] = np.nan
    assert np.allclose(target_weights(d, PARAMS), 0.0)
    mask, side = events(d, PARAMS)
    assert not mask.any()


def test_missing_ncf_skips_stock_and_shrinks_cross_section():
    d = make()
    d.extras["sf1_art_ncfcommon"][:, 29] = np.nan   # 29 valid stocks: pct (j+0.5)/29
    w = target_weights(d, PARAMS)
    assert w[3, 29] == 0.0
    # long: (j+0.5)/29 >= 0.8 -> j >= 22.7 -> 23..28 ; short: j <= 5.3 -> 0..5
    assert np.allclose(w[3, 23:29], 0.5 / 6)
    assert np.allclose(w[3, 0:6], -0.5 / 6)


def test_non_member_not_in_cross_section_or_event():
    d = make()
    d.universe[3, 0] = False
    mask, side = events(d, PARAMS)
    assert not mask[3, 0]
    # 29 members, stock j>=1 rank j-1: short (r+0.5)/29 <= 0.2 -> r <= 5 -> j 1..6
    assert np.all(side[3, 1:7] == -1.0)


def test_leaving_universe_closes_for_good():
    d = make()
    d.universe[5, 29] = False      # leaves on row 5, back on row 6
    w = target_weights(d, PARAMS)
    assert np.allclose(w[4, 24:30], 0.5 / 6)
    assert np.all(w[5:, 29] == 0.0)
    assert np.allclose(w[5:8, 24:29], 0.5 / 5)
    d = make()
    d.tradable[6, 0] = False       # stops trading on row 6
    w = target_weights(d, PARAMS)
    assert np.all(w[6:, 0] == 0.0)
    assert np.allclose(w[6:8, 1:6], -0.5 / 5)


def test_new_event_restarts_clock_with_new_side():
    d = make()
    d.extras["sf1_art_marketcap"][6:, 0] = 300.0
    d.extras["sf1_art_ncfcommon"][6:, 0] = -1e9   # F huge -> top
    mask, side = events(d, PARAMS)
    assert mask[6, 0] and side[6, 0] == 1.0
    w = target_weights(d, PARAMS)
    assert w[5, 0] == pytest.approx(-0.5 / 6)
    assert w[6, 0] == pytest.approx(0.5 / 7)      # 7 longs, 5 shorts
    assert np.allclose(w[6, 1:6], -0.5 / 5)
    assert np.allclose(w[8:], 0.0)                # only stock 0 left long -> legs < 5 -> flat


def test_not_tradable_on_filing_day_no_event():
    d = make()
    d.tradable[3, 29] = False
    mask, side = events(d, PARAMS)
    assert not mask[3, 29]
    assert np.all(target_weights(d, PARAMS)[:, 29] == 0.0)


def test_point_in_time_truncation():
    d = make()
    d.extras["sf1_art_marketcap"][9:, 10:20] = 400.0
    rng = np.random.default_rng(0)
    d.extras["sf1_art_ncfcommon"][9:] = rng.normal(size=(T - 9, N)).astype(np.float32) * 100
    full = target_weights(d, PARAMS)
    for k in range(2, T):
        part = target_weights(d.truncate(k), PARAMS)
        assert np.array_equal(part, full[:k])


def test_diagnostic_matches_strategy_entries():
    mask, side = events(make(), PARAMS)
    assert mask.sum() == 12
    assert np.all(side[3, LONG] == 1.0) and np.all(side[3, SHORT] == -1.0)


def test_missing_input_raises():
    d = make()
    del d.extras["sf1_art_sbcomp"]
    with pytest.raises(KeyError):
        target_weights(d, PARAMS)
