import numpy as np
import pytest
from lab.framework.data import DataView
from strategy import target_weights, PARAMS, book_weights

T = 80
N = 12
EV = 70
P0 = {"volume_ratio": 3, "ref_window": 10, "hold_days": 3}


class Market:
    def __init__(self, T=T, N=N):
        self.T, self.N = T, N
        self.dv = np.full((T, N), 1e6)
        self.ret_oc = np.zeros((T, N))
        self.ret_co = np.zeros((T, N))
        self.tradable = np.ones((T, N), dtype=bool)
        self.listed = np.ones((T, N), dtype=bool)
        self.universe = np.ones((T, N), dtype=bool)

    def trend(self, i, sign, start=55, end=EV, size=0.01):
        self.ret_oc[start:end, i] = sign * size

    def event(self, i, sign, row=EV, mult=5.0, size=0.05):
        self.dv[row, i] = 1e6 * mult
        self.ret_oc[row, i] = sign * size

    def view(self, with_universe=True):
        names = tuple("S%02d" % k for k in range(self.N))
        close = np.where(self.listed, 10.0, np.nan)
        return DataView(
            dates=np.arange(12000, 12000 + self.T).astype("datetime64[D]"),
            instruments=names,
            asset_class=tuple("us_equity" for _ in names),
            ret_co=self.ret_co.copy(),
            ret_oc=self.ret_oc.copy(),
            tradable=self.tradable.copy(),
            listed=self.listed.copy(),
            delisting=np.zeros((self.T, self.N), dtype=bool),
            close=close,
            dollar_volume=np.where(self.listed, self.dv, np.nan),
            universe=self.universe.copy() if with_universe else None,
            extras={"liq_rank": np.ones((self.T, self.N))},
            series={},
            cash_ret=np.zeros(self.T),
        )


def base_market():
    m = Market()
    m.trend(0, +1); m.event(0, +1)   # A: gains + good news -> long
    m.trend(1, -1); m.event(1, -1)   # B: losses + bad news -> short
    m.trend(2, +1); m.event(2, -1)   # C: gains + bad news -> misaligned
    m.trend(3, -1); m.event(3, +1)   # D: losses + good news -> misaligned
    return m


def test_params_names():
    assert PARAMS == {"volume_ratio": 3, "ref_window": 252, "hold_days": 5}


def test_aligned_events_and_hold():
    W = target_weights(base_market().view(), P0)
    assert W.shape == (T, N)
    assert np.all(W[:EV] == 0)
    for t in range(EV, EV + 3):
        assert W[t, 0] == pytest.approx(0.10)
        assert W[t, 1] == pytest.approx(-0.10)
        assert np.all(W[t, 2:] == 0)
    assert np.all(W[EV + 3:] == 0)


def test_hold_days_param():
    W = target_weights(base_market().view(), dict(P0, hold_days=5))
    assert np.all(W[EV:EV + 5, 0] > 0) and np.all(W[EV + 5:, 0] == 0)


def test_volume_ratio_param():
    m = base_market()
    m.dv[EV, :4] = 2.5e6
    assert np.all(target_weights(m.view(), P0) == 0)
    W = target_weights(m.view(), dict(P0, volume_ratio=2))
    assert W[EV, 0] == pytest.approx(0.10) and W[EV, 1] == pytest.approx(-0.10)


def test_ref_window_param():
    m = Market()
    m.ret_oc[5, 4] = 1.0      # price doubles early
    m.ret_oc[40, 4] = -0.5    # then halves
    m.trend(4, +1, start=60, size=0.005)
    m.event(4, +1)
    W = target_weights(m.view(), P0)            # R over rows 60..69 is below P_69 -> long
    assert W[EV, 4] == pytest.approx(0.10)
    W = target_weights(m.view(), dict(P0, ref_window=60))   # window holds the high prices -> G<0, misaligned
    assert np.all(W == 0)


def test_restart_and_misaligned_close():
    m = base_market()
    m.event(0, +1, row=EV + 2)            # aligned again -> clock restarts
    W = target_weights(m.view(), P0)
    assert np.all(W[EV:EV + 5, 0] > 0) and W[EV + 5, 0] == 0
    m = base_market()
    m.event(0, -1, row=EV + 1)            # still G>0, bad news -> misaligned -> close
    W = target_weights(m.view(), P0)
    assert W[EV, 0] > 0 and np.all(W[EV + 1:, 0] == 0)
    assert np.all(W[EV:EV + 3, 1] < 0)


def test_flip_on_opposite_aligned_event():
    m = Market()
    m.trend(0, +1, start=55, end=EV + 1)
    m.event(0, +1, row=EV)
    # make G<0 at EV+6: a crash and then a bad-news event
    m.ret_oc[EV + 1:EV + 6, 0] = -0.08
    m.event(0, -1, row=EV + 6)
    W = target_weights(m.view(), dict(P0, hold_days=10))
    assert np.all(W[EV:EV + 6, 0] > 0)
    assert np.all(W[EV + 6:, 0] < 0)


def test_market_wide_volume_day_is_not_an_event():
    m = base_market()
    m.dv[EV, :] = 5e6
    assert np.all(target_weights(m.view(), P0) == 0)


def test_market_adjusted_direction():
    m = base_market()
    m.ret_oc[EV, 4:] = 0.10                 # market up 10 %, A up only 5 % -> A's adjusted move < 0
    W = target_weights(m.view(), P0)        # cross-sectional median return = 0.10
    assert np.all(W[:, 0] == 0)             # A: gains, adjusted move -0.05 -> misaligned
    assert np.all(W[:, 2] == 0)             # C: gains, adjusted -0.15 -> misaligned
    assert W[EV, 1] == pytest.approx(-0.10) # B: losses, adjusted -0.15 -> short
    assert W[EV, 3] == pytest.approx(-0.10) # D: losses, adjusted -0.05 -> short (raw move was up)
    assert np.all(W[EV, 4:] == 0)


def test_universe_and_tradable_gate():
    m = base_market()
    m.universe[EV, 0] = False
    m.tradable[EV, 1] = False
    W = target_weights(m.view(), P0)
    assert np.all(W[:, 0] == 0) and np.all(W[:, 1] == 0)
    m = base_market()
    m.universe[EV + 1:, 0] = False           # leaves LIQ-500 after the event: closed (G0 rule)
    W = target_weights(m.view(), P0)
    assert W[EV, 0] == pytest.approx(0.10)
    assert np.all(W[EV + 1:, 0] == 0)
    assert np.all(W[EV:EV + 3, 1] < 0)
    m = base_market()
    m.universe[EV + 1, 0] = False            # out for one day: position is not reopened
    W = target_weights(m.view(), P0)
    assert np.all(W[EV + 1:, 0] == 0)


def test_late_listing_needs_40_baseline_days():
    m = base_market()
    m.listed[:35, 0] = False
    m.tradable[:35, 0] = False
    m.universe[:35, 0] = False
    W = target_weights(m.view(), P0)          # 35 valid baseline days -> skip
    assert np.all(W[:, 0] == 0)
    m = base_market()
    m.listed[:30, 0] = False
    m.tradable[:30, 0] = False
    m.universe[:30, 0] = False
    W = target_weights(m.view(), P0)          # 40 valid baseline days -> event
    assert W[EV, 0] == pytest.approx(0.10)


def test_overhang_needs_half_window():
    m = base_market()
    m.dv[EV - 6:EV, 0] = np.nan              # only 4 of the 10 reference days valid
    W = target_weights(m.view(), P0)
    assert np.all(W[:, 0] == 0)
    assert W[EV, 1] < 0


def test_many_names_split_and_gross():
    m = Market(N=30)
    for i in range(7):
        m.trend(i, +1); m.event(i, +1)
    for i in range(7, 10):
        m.trend(i, -1); m.event(i, -1)
    W = target_weights(m.view(), P0)
    assert np.allclose(W[EV, :7], 0.5 / 7)
    assert np.allclose(W[EV, 7:10], -0.10)
    assert np.all(np.abs(W).sum(axis=1) <= 1.0 + 1e-12)


def test_book_weights():
    s = np.array([1, 1, 1, 0, -1] + [-1] * 9, dtype=np.int8)
    w = book_weights(s)
    assert np.allclose(w[:3], 0.10) and w[3] == 0
    assert np.allclose(w[4:], -0.05)
    assert np.all(book_weights(np.zeros(4, dtype=np.int8)) == 0)


def test_point_in_time_truncation():
    m = base_market()
    m.event(0, +1, row=EV + 2)
    full = target_weights(m.view(), P0)
    for cut in (EV - 1, EV, EV + 1, EV + 3):
        mt = Market(T=cut + 1)
        mt.dv = m.dv[:cut + 1].copy()
        mt.ret_oc = m.ret_oc[:cut + 1].copy()
        Wc = target_weights(mt.view(), P0)
        assert np.array_equal(Wc, full[:cut + 1])


def test_missing_universe_raises():
    with pytest.raises(ValueError):
        target_weights(base_market().view(with_universe=False), P0)
