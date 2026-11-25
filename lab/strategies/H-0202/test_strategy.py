import numpy as np
import pytest

from strategy import PARAMS, target_positions, trade_log

ALTS = tuple("X%02d" % i for i in range(12))
NAMES = ("BTCUSDT", "ETHUSDT") + ALTS
T = 3300
T0 = 3100            # event candle (decision after its close)
JUMP = 0.05


class View:
    def __init__(self, btc_ret, alt_ret, qv=None, real=None, universe=None):
        n = len(NAMES)
        lr = np.zeros((T, n))
        lr[:, 0] = btc_ret
        lr[:, 1] = btc_ret
        lr[:, 2:] = alt_ret
        c = 100.0 * np.exp(np.cumsum(lr, axis=0))
        self.instruments = NAMES
        self.times = (np.arange(T, dtype=np.int64) * 15).astype("datetime64[m]")
        self.c = c
        o = c.copy()
        o[1:] = c[:-1]
        self.o = o
        self.h = c
        self.l = c
        self.qv = np.full((T, n), 1.0e8) if qv is None else qv
        self.real = np.ones((T, n), dtype=bool) if real is None else real
        self.universe = np.ones((T, n), dtype=bool) if universe is None else universe
        self.tradable = self.real.copy()
        self.bars_per_day = 96

    def col(self, name):
        return self.instruments.index(name)


def base_btc():
    r = 0.001 * np.sin(2 * np.pi * np.arange(T) / 7.0)
    r[0] = 0.0
    return r


def scenario(moves, jump=JUMP, betas=None, t0=T0):
    """BTC base path plus a one-candle jump at t0; alt j follows beta_j * BTC base returns and
    moves by moves[j] (log) on candle t0 instead of beta_j * jump."""
    b = base_btc()
    betas = np.ones(len(ALTS)) if betas is None else np.asarray(betas, dtype=float)
    alt = b[:, None] * betas[None, :]
    btc = b.copy()
    btc[t0] += jump
    for j, m in moves.items():
        alt[t0, j] += m
    for j in range(len(ALTS)):
        if j not in moves:
            alt[t0, j] += betas[j] * jump
    return btc, alt


P1 = {"beta_days": 1, "k_sigma": 3.0}


def col(name):
    return NAMES.index(name)


def test_up_event_sizes_by_expected_catch_up():
    btc, alt = scenario({0: 0.01, 2: 0.045, 3: 0.08}, betas=[1, 2] + [1] * 10)
    alt[T0, 1] = 2 * base_btc()[T0] + 0.05          # beta 2, moved 0.05: E = 2 * 0.05 - 0.05
    v = View(btc, alt)
    pos = target_positions(v, P1)
    a0, a1, a2, a3 = col("X00"), col("X01"), col("X02"), col("X03")
    assert np.allclose(pos[T0:T0 + 16, a0], 0.1)                 # E = 0.04 -> capped at 0.1
    assert np.allclose(pos[T0:T0 + 16, a1], 0.1)                 # E = 0.05 -> capped
    assert np.allclose(pos[T0:T0 + 16, a2], 0.025, atol=1e-6)    # E = 0.005 -> 0.005 / 0.02 / 10
    assert np.all(pos[:, a3] == 0)                               # overshot BTC: not traded
    assert np.all(pos[T0 + 16:] == 0) and np.all(pos[:T0] == 0)  # exit at the open of t0 + 17
    assert np.all(pos[:, col("BTCUSDT")] == 0) and np.all(pos[:, col("ETHUSDT")] == 0)
    others = [col(n) for n in ALTS[4:]]
    assert np.all(pos[:, others] == 0)                           # moved with BTC: E = 0
    assert np.abs(pos).sum(axis=1).max() <= 1.0 + 1e-12


def test_down_event_is_short():
    btc, alt = scenario({0: -0.01}, jump=-JUMP)
    pos = target_positions(View(btc, alt), P1)
    assert np.allclose(pos[T0:T0 + 16, col("X00")], -0.1)
    assert np.all(pos[:, col("X01")] == 0)


def test_negative_beta_skipped():
    btc, alt = scenario({0: 0.0}, betas=[-1] + [1] * 11)
    pos = target_positions(View(btc, alt), P1)
    assert np.all(pos[:, col("X00")] == 0)


def test_gap_at_entry_candle_cancels_at_next_row():
    btc, alt = scenario({0: 0.01, 1: 0.01})
    real = np.ones((T, len(NAMES)), dtype=bool)
    real[T0 + 1, col("X00")] = False
    pos = target_positions(View(btc, alt, real=real), P1)
    assert pos[T0, col("X00")] == pytest.approx(0.1)    # order at t0 cannot fill on the gap open
    assert np.all(pos[T0 + 1:, col("X00")] == 0)        # cancelled once the gap is known
    assert np.allclose(pos[T0:T0 + 16, col("X01")], 0.1)


def test_gap_in_measurement_hour_skips():
    btc, alt = scenario({0: 0.01, 1: 0.01})
    real = np.ones((T, len(NAMES)), dtype=bool)
    real[T0 - 2, col("X00")] = False
    pos = target_positions(View(btc, alt, real=real), P1)
    assert np.all(pos[:, col("X00")] == 0)
    assert np.allclose(pos[T0:T0 + 16, col("X01")], 0.1)


def test_real_hour_share_80_percent():
    btc, alt = scenario({0: 0.01})
    for m, traded in ((4, True), (5, False)):        # 4 or 5 of 24 beta hours lost
        real = np.ones((T, len(NAMES)), dtype=bool)
        for k in range(1, m + 1):
            real[T0 - 4 * k, col("X00")] = False
        pos = target_positions(View(btc, alt, real=real), P1)
        assert (pos[T0, col("X00")] != 0) == traded


def test_universe_membership_and_exit():
    btc, alt = scenario({0: 0.01, 1: 0.01})
    uni = np.ones((T, len(NAMES)), dtype=bool)
    uni[T0, col("X01")] = False
    uni[T0 + 5:T0 + 7, col("X00")] = False
    pos = target_positions(View(btc, alt, universe=uni), P1)
    assert np.all(pos[:, col("X01")] == 0)
    assert np.allclose(pos[T0:T0 + 5, col("X00")], 0.1)
    assert np.all(pos[T0 + 5:, col("X00")] == 0)        # no re-entry after the universe exit
    assert np.all(pos[~uni] == 0)


def test_top_ten_by_quote_volume():
    btc, alt = scenario({j: 0.01 for j in range(12)})
    qv = np.full((T, len(NAMES)), 1.0e8)
    qv[:, 2:] = 1.0e8 * (1.0 + np.arange(12)[::-1])[None, :]   # X00 largest, X11 smallest
    qv[:, 1] = 1.0e12                                          # ETH excluded even if largest
    pos = target_positions(View(btc, alt, qv=qv), P1)
    traded = [n for n in ALTS if pos[T0, col(n)] != 0]
    assert traded == list(ALTS[:10])
    assert np.abs(pos[T0]).sum() == pytest.approx(1.0)


def test_cost_floor_uses_slippage_tier():
    btc, alt = scenario({0: 0.047})                   # E = 0.003
    uni = np.zeros((T, len(NAMES)), dtype=bool)
    uni[:, [0, 1, col("X00")]] = True
    pos = target_positions(View(btc, alt, universe=uni), P1)   # daily qv 9.6e9: floor 2 * 6.5 bps
    assert pos[T0, col("X00")] == pytest.approx(0.015, abs=1e-6)
    qv = np.full((T, len(NAMES)), 1.0e8)
    qv[:, col("X00")] = 1.0e5                                  # daily 9.6e6: floor 2 * 20 bps
    pos = target_positions(View(btc, alt, qv=qv, universe=uni), P1)
    assert np.all(pos[:, col("X00")] == 0)


def _jumps(lag_at, full_at):
    b = base_btc()
    btc = b.copy()
    alt = np.repeat(b[:, None], 12, axis=1)
    for t in tuple(lag_at) + tuple(full_at):
        btc[t] += JUMP
        alt[t, :] += JUMP
        if t in lag_at:
            alt[t, 0] += 0.01 - JUMP
    return btc, alt


def test_no_stacking_and_reentry_after_close():
    btc, alt = _jumps(lag_at=(T0, T0 + 17), full_at=(T0 + 10,))
    pos = target_positions(View(btc, alt), P1)
    a0 = col("X00")
    assert np.allclose(pos[T0:T0 + 16, a0], 0.1)
    assert pos[T0 + 16, a0] == 0                 # the event at t0+10 is ignored: book still open
    assert np.all(pos[T0 + 17:T0 + 33, a0] > 0)  # next book from t0+17
    assert np.all(pos[T0 + 33:, a0] == 0)


def test_event_on_exit_candle_is_ignored():
    btc, alt = _jumps(lag_at=(T0, T0 + 16), full_at=())
    pos = target_positions(View(btc, alt), P1)
    assert np.allclose(pos[T0:T0 + 16, col("X00")], 0.1)
    assert np.all(pos[T0 + 16] == 0)             # candle t0+16 ignored, the book closes at the open of t0+17


def test_k_sigma_changes_events():
    b = base_btc()
    r4 = np.full(T, np.nan)
    lp = np.cumsum(b)
    r4[4:] = lp[4:] - lp[:-4]
    sigma = np.std(r4[T0 - 4 - 2879:T0 - 3], ddof=1)
    jump = 2.75 * sigma - r4[T0]
    btc, alt = scenario({0: 0.0}, jump=jump)
    lo = target_positions(View(btc, alt), {"beta_days": 1, "k_sigma": 2.5})
    hi = target_positions(View(btc, alt), {"beta_days": 1, "k_sigma": 3.0})
    assert lo[T0, col("X00")] == pytest.approx(jump / 0.02 / 10, rel=1e-6)
    assert np.all(hi[:T0 + 1] == 0)          # z = 2.75 at t0: no event at k_sigma = 3
    assert np.all(lo[T0 + 16] == 0)


def test_beta_days_changes_beta():
    b = base_btc()
    btc = b.copy()
    btc[T0] += JUMP
    alt = np.repeat(b[:, None], 12, axis=1)
    alt[: T0 - 4 - 96, 0] *= 2.0                   # beta 2 before the last day, 1 within it
    alt[T0, :] += JUMP
    alt[T0, 0] = b[T0] + 0.04
    one = trade_log(View(btc, alt), {"beta_days": 1, "k_sigma": 3.0})
    two = trade_log(View(btc, alt), {"beta_days": 2, "k_sigma": 3.0})
    i1 = list(one["instrument"]).index("X00")
    i2 = list(two["instrument"]).index("X00")
    assert one["beta"][i1] == pytest.approx(1.0, rel=1e-6)
    assert two["beta"][i2] > 1.2
    assert two["E"][i2] > one["E"][i1]


def test_trade_log_fields():
    btc, alt = scenario({0: 0.01})
    v = View(btc, alt)
    log = trade_log(v, P1)
    assert list(log["instrument"]) == ["X00"]
    assert log["entry_row"][0] == T0 + 1 and log["exit_row"][0] == T0 + 17
    assert log["side"][0] == 1.0 and log["E"][0] == pytest.approx(0.04, abs=1e-9)
    j = col("X00")
    assert log["gross_return"][0] == pytest.approx(v.o[T0 + 17, j] / v.o[T0 + 1, j] - 1)


def test_point_in_time_truncation():
    btc, alt = scenario({0: 0.01, 1: 0.02})
    full = View(btc, alt)
    pos = target_positions(full, P1)
    for cut in (T0 - 1, T0, T0 + 1, T0 + 8):
        v = View(btc, alt)
        n = cut + 1
        v.times = v.times[:n]
        v.c = v.c[:n]
        v.o = v.o[:n]
        v.h = v.h[:n]
        v.l = v.l[:n]
        v.qv = v.qv[:n]
        v.real = v.real[:n]
        v.universe = v.universe[:n]
        v.tradable = v.tradable[:n]
        assert np.array_equal(target_positions(v, P1)[:n], pos[:n])


def test_primary_params():
    assert PARAMS == {"beta_days": 30, "k_sigma": 3.0}
