import numpy as np
import pytest

from strategy import target_positions, PARAMS

NAMES = ("BTCUSDT", "ETHUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT")


class View:
    def __init__(self, c, real=None, universe=None, listed=None):
        T, N = c.shape
        self.instruments = NAMES[:N]
        self.c = c.astype(np.float32)
        self.real = np.ones((T, N), bool) if real is None else real
        self.listed = np.ones((T, N), bool) if listed is None else listed
        self.universe = np.ones((T, N), bool) if universe is None else universe

    def col(self, name):
        return self.instruments.index(name)


def make_market(T=800, jump_row=700, jump=0.05, follower_moves=(0.0, 0.0, 0.0, 0.0)):
    """BTC alternates +-0.001 per candle (sigma ~0.001); at jump_row BTC jumps by `jump` (log)."""
    N = 5
    lr = np.zeros((T, N))
    alt = np.where(np.arange(T) % 2 == 0, 0.001, -0.001)
    lr[:, 0] = alt
    lr[jump_row, 0] = jump
    for k, m in enumerate(follower_moves):
        lr[jump_row, 1 + k] = m
    c = 100.0 * np.exp(np.cumsum(lr, axis=0))
    return c


def test_entry_hold_and_exit_rows():
    # ETH (col 1) lags but is excluded; AAA lags (0), BBB moved fully (0.05), CCC declined (-0.01)
    c = make_market(follower_moves=(0.0, 0.0, 0.05, -0.01))
    w = target_positions(View(c), PARAMS)
    h = PARAMS["hold_candles"]
    t = 700
    assert np.all(w[:t] == 0)
    assert np.allclose(w[t:t + h, 2], 0.1)
    assert w[t + h, 2] == 0
    assert np.allclose(w[t:t + h, 4], 0.1)  # declines count as lagging
    assert np.all(w[:, 3] == 0)              # moved as much as BTC
    assert np.all(w[:, 1] == 0)              # ETH excluded
    assert np.all(w[:, 0] == 0)              # leader never held
    # BTC 1h return stays elevated for rows t..t+3; re-signals while held are ignored
    assert np.all(w[t + h:, 2] == 0)


def test_no_signal_below_threshold_and_k_sigma_matters():
    # jump 0.0075 -> r_L = 0.0065; sigma ~ 0.001041, s_1h ~ 0.002082: 3.0*s_1h = 0.00625 < r_L < 3.5*s_1h = 0.00729
    c = make_market(jump=0.0075)
    w3 = target_positions(View(c), {"hold_candles": 16, "k_sigma": 3.0})
    w35 = target_positions(View(c), {"hold_candles": 16, "k_sigma": 3.5})
    assert w3[:, 2].max() == pytest.approx(0.1)
    assert np.all(w35 == 0)


def test_hold_candles_param():
    c = make_market()
    w12 = target_positions(View(c), {"hold_candles": 12, "k_sigma": 3.0})
    w24 = target_positions(View(c), {"hold_candles": 24, "k_sigma": 3.0})
    assert (w12[:, 2] > 0).sum() == 12
    assert (w24[:, 2] > 0).sum() == 24


def test_lookback_and_gap_rules():
    # jump before 672 candles of history -> no signal
    c = make_market(T=700, jump_row=600)
    assert np.all(target_positions(View(c), PARAMS) == 0)
    # a non-real BTC candle inside the 672 window -> no signal
    c = make_market()
    real = np.ones(c.shape, bool)
    real[300, 0] = False
    assert np.all(target_positions(View(c, real=real), PARAMS) == 0)
    # a non-real follower candle in t-4..t -> no signal for that follower only
    real = np.ones(c.shape, bool)
    real[700, 2] = False  # blocks t = 700..704, i.e. every row where r_L is elevated
    w = target_positions(View(c, real=real), PARAMS)
    assert np.all(w[:, 2] == 0)
    assert w[:, 3].max() == pytest.approx(0.1)


def test_universe_exit_and_no_resume():
    c = make_market()
    uni = np.ones(c.shape, bool)
    uni[705:708, 2] = False  # leaves for 3 rows, comes back
    w = target_positions(View(c, universe=uni), PARAMS)
    assert np.allclose(w[700:705, 2], 0.1)
    assert np.all(w[705:708, 2] == 0)
    assert np.allclose(w[708:716, 2], 0.1)  # hold clock kept running; exit row 716 unchanged
    assert w[716, 2] == 0
    # not a member at the signal row -> no entry
    uni = np.ones(c.shape, bool)
    uni[700:704, 3] = False  # r_L is elevated on rows 700..703
    w = target_positions(View(c, universe=uni), PARAMS)
    assert np.all(w[:, 3] == 0)


def test_signal_on_exit_row_ignored_and_reentry_after():
    T = 900
    lr = np.zeros((T, 5))
    lr[:, 0] = np.where(np.arange(T) % 2 == 0, 0.001, -0.001)
    lr[700, 0] = 0.05
    lr[716, 0] = 0.05   # signal on exit row (700+16) -> ignored
    lr[740, 0] = 0.05   # later signal -> new entry
    c = 100.0 * np.exp(np.cumsum(lr, axis=0))
    w = target_positions(View(c), PARAMS)
    assert w[716, 2] == 0
    # rows 717..719 still have r_L elevated from the 716 jump -> entry at 717 allowed
    assert w[717, 2] == pytest.approx(0.1)
    assert np.all(w >= 0)
    assert np.all(np.abs(w).sum(axis=1) <= 1 + 1e-12)


def test_point_in_time_truncation():
    c = make_market(T=900)
    full = target_positions(View(c), PARAMS)
    for cut in (650, 701, 710, 800):
        part = target_positions(View(c[:cut]), PARAMS)
        assert np.array_equal(part, full[:cut])
