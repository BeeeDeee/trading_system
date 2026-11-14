import numpy as np
import pytest
from dataclasses import dataclass
from typing import Any

from strategy import PARAMS, target_positions, atr_real, book, session_bands, session_clock

DAY0 = 18262  # a UTC day number (days since epoch)


@dataclass
class FakeView:
    times: Any
    instruments: tuple
    o: Any
    h: Any
    l: Any
    c: Any
    qv: Any
    real: Any
    tradable: Any
    universe: Any
    funding: Any
    bars_per_day: int = 288

    def col(self, name):
        return self.instruments.index(name)


def make_times(T, start_minute=0):
    m = np.arange(T, dtype=np.int64) * 5 + DAY0 * 1440 + start_minute
    return m.astype("datetime64[m]")


def scenario(T=300):
    """BTC: one UTC session with a clean long setup at test candle 42, signal 43, target touched at 47."""
    c = np.full(T, 100.0)
    h = c + 0.1
    l = c - 0.1
    for k in range(36):
        c[k] = 99.0 if k % 2 == 0 else 101.0
        h[k] = c[k] + 0.1
        l[k] = c[k] - 0.1
    rows = {
        42: (99.3, 98.8, 99.2),   # test candle: low through L, close back inside
        43: (99.5, 99.2, 99.4),   # confirmation close above h_42
        44: (99.7, 99.3, 99.6),
        45: (99.9, 99.5, 99.8),
        46: (99.9, 99.6, 99.8),
        47: (100.2, 99.8, 100.0),  # touches the target (session VWAP ~ 99.97)
    }
    for k, (hh, ll, cc) in rows.items():
        h[k], l[k], c[k] = hh, ll, cc
    return h, l, c


def view_from(h, l, c, real=None, eth=None):
    T = c.shape[0]
    if real is None:
        real = np.ones(T, bool)
    if eth is None:
        eth = np.full(T, 50.0)
    H = np.stack([h, eth], axis=1).astype(np.float32)
    Lw = np.stack([l, eth], axis=1).astype(np.float32)
    C = np.stack([c, eth], axis=1).astype(np.float32)
    qv = np.ones((T, 2), np.float32)
    qv[:, 0] = np.where(real, 1.0, 0.0)
    R = np.stack([real, np.ones(T, bool)], axis=1)
    return FakeView(times=make_times(T), instruments=("BTCUSDT", "ETHUSDT"), o=C.copy(), h=H, l=Lw, c=C, qv=qv,
                    real=R, tradable=R.copy(), universe=None, funding=np.zeros((T, 2), np.float32))


def run(view, **kw):
    p = dict(PARAMS)
    p.update(kw)
    return target_positions(view, p)


# ------------------------------------------------------------------------------------------ building blocks
def test_session_clock():
    t = make_times(600)
    sess, idx = session_clock(t, 288)
    assert idx[0] == 0 and idx[287] == 287 and idx[288] == 0
    assert sess[287] == sess[0] and sess[288] == sess[0] + 1


def test_bands_match_naive_loop():
    rng = np.random.default_rng(1)
    T = 700
    c = 100 + np.cumsum(rng.normal(0, 0.2, T))
    h = c + rng.uniform(0, 0.3, T)
    l = c - rng.uniform(0, 0.3, T)
    qv = rng.uniform(0, 5, T)
    qv[rng.random(T) < 0.1] = 0.0
    real = rng.random(T) > 0.05
    sess = np.arange(T) // 288
    V, sg = session_bands(h, l, c, qv, real, sess)
    for t in range(T):
        a = sess[t] * 288
        sw = swp = swp2 = 0.0
        for k in range(a, t + 1):
            w = qv[k] if (real[k] and qv[k] > 0) else 0.0
            tp = (h[k] + l[k] + c[k]) / 3
            sw += w
            swp += w * tp
            swp2 += w * tp * tp
        if sw == 0:
            assert np.isnan(V[t])
        else:
            v = swp / sw
            assert abs(V[t] - v) < 1e-9
            assert abs(sg[t] - np.sqrt(max(0.0, swp2 / sw - v * v))) < 1e-5


def test_atr_last_14_real_candles():
    rng = np.random.default_rng(2)
    T = 120
    c = 50 + np.cumsum(rng.normal(0, 0.5, T))
    h = c + rng.uniform(0, 1, T)
    l = c - rng.uniform(0, 1, T)
    real = rng.random(T) > 0.2
    a = atr_real(h, l, c, real, 14)
    trs = []
    for t in range(T):
        if real[t]:
            trs.append(np.nan if t == 0 else max(h[t], c[t - 1]) - min(l[t], c[t - 1]))
        last = trs[-14:]
        if len(last) < 14 or np.isnan(last).any():
            assert np.isnan(a[t])
        else:
            assert abs(a[t] - np.mean(last)) < 1e-9


# ------------------------------------------------------------------------------------------ the setup
def test_long_setup_confirm3_mid_target():
    w = run(view_from(*scenario()))
    btc = w[:, 0]
    assert np.all(btc[:43] == 0)
    assert np.all(btc[43:47] == 0.5)      # signal at 43, held, target touched at the close of 47
    assert np.all(btc[47:] == 0)
    assert np.all(w[:, 1] == 0)           # flat ETH: no band width, never passes the cost filter


def test_confirm0_enters_at_test_candle():
    w = run(view_from(*scenario()), confirm_candles=0)
    btc = w[:, 0]
    assert np.all(btc[:42] == 0)
    assert np.all(btc[42:47] == 0.5)
    assert np.all(btc[47:] == 0)


def test_target_frac_quarter_fails_cost_filter():
    w = run(view_from(*scenario()), target_frac=0.25)
    assert np.all(w == 0)


def test_target_frac_one_exits_at_session_end():
    w = run(view_from(*scenario()), target_frac=1.0)
    btc = w[:, 0]
    assert np.all(btc[:43] == 0)
    assert np.all(btc[43:287] == 0.5)     # opposite band never touched: flat after the close of candle 287
    assert np.all(btc[287:] == 0)


def test_short_is_exact_mirror():
    h, l, c = scenario()
    w = run(view_from(200 - l, 200 - h, 200 - c))
    w_long = run(view_from(h, l, c))
    assert np.array_equal(w[:, 0], -w_long[:, 0])


def test_stop_exit():
    h, l, c = scenario()
    l[44], c[44] = 98.0, 99.3              # stop = min(l_42, l_43) - 0.25 ATR ~ 98.5
    w = run(view_from(h, l, c))
    btc = w[:, 0]
    assert btc[43] == 0.5
    assert np.all(btc[44:] == 0)


def test_gap_candle_is_never_the_signal():
    h, l, c = scenario()
    real = np.ones(c.shape[0], bool)
    real[43] = False                        # a gap candle whose close is above h_42 must not trigger
    w = run(view_from(h, l, c, real=real))
    btc = w[:, 0]
    assert np.all(btc[:44] == 0)
    assert np.all(btc[44:47] == 0.5)
    assert np.all(btc[47:] == 0)


def test_close_below_band_cancels():
    h, l, c = scenario()
    h[43], l[43], c[43] = 99.0, 97.0, 97.5  # closes below L before any confirmation
    w = run(view_from(h, l, c), confirm_candles=3)
    assert np.all(w[40:46, 0] == 0)


def test_no_test_before_index_36():
    h, l, c = scenario()
    # move the whole pattern so the test candle falls at session index 30
    sh = 12
    h2, l2, c2 = np.roll(h, -sh), np.roll(l, -sh), np.roll(c, -sh)
    w = run(view_from(h2, l2, c2), confirm_candles=0)
    assert np.all(w[:, 0] == 0)


def test_book_limits():
    T = 60
    h = np.full(T, 100.5)
    l = np.full(T, 99.5)
    sess = np.zeros(T, np.int64)
    idx = np.arange(T)
    member = np.ones(T, bool)
    h[12] = 102.0                            # long target 101 touched at 12
    ev = [(10, 9, 1, 101.0, 98.0),
          (11, 9, -1, 99.0, 102.0),          # arrives while long: ignored
          (20, 19, 1, 101.0, 98.0),          # second long in the same session: ignored
          (30, 29, -1, 99.0, 103.0)]         # first short of the session: taken, never exits (end of data)
    pos = book(ev, h, l, sess, idx, member, T)
    assert np.all(pos[10:12] == 0.5) and pos[12] == 0
    assert np.all(pos[13:30] == 0)
    assert np.all(pos[30:] == -0.5)
    both = book([(10, 9, 1, 101.0, 98.0), (10, 8, -1, 99.0, 102.0)], h, l, sess, idx, member, T)
    assert np.all(both == 0)                 # simultaneous long and short signals: ambiguous, skipped


# ------------------------------------------------------------------------------------------ integrity
def random_view(T=288 * 4, seed=3):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(2):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, T)))
        h = c * (1 + rng.uniform(0, 0.004, T))
        l = c * (1 - rng.uniform(0, 0.004, T))
        v.append((h, l, c))
    real = rng.random(T) > 0.02
    view = view_from(*v[0], real=real, eth=v[1][2])
    view.h[:, 1] = v[1][0]
    view.l[:, 1] = v[1][1]
    view.qv[:] = rng.uniform(0.5, 5, (T, 2)).astype(np.float32)
    return view


def test_gross_and_values():
    for n in (0, 3, 6):
        for f in (0.25, 0.5, 1.0):
            w = run(random_view(), confirm_candles=n, target_frac=f)
            assert np.all(np.abs(w).sum(axis=1) <= 1 + 1e-12)
            assert set(np.unique(w)).issubset({-0.5, 0.0, 0.5})


def test_truncation_point_in_time():
    v = random_view()
    full = run(v)
    T = full.shape[0]
    for cut in (100, 400, 700, 1000):
        tv = FakeView(times=v.times[:cut], instruments=v.instruments, o=v.o[:cut], h=v.h[:cut], l=v.l[:cut],
                      c=v.c[:cut], qv=v.qv[:cut], real=v.real[:cut], tradable=v.tradable[:cut], universe=None,
                      funding=v.funding[:cut])
        part = run(tv)
        assert np.array_equal(part, full[:cut])
    assert T == 288 * 4


def test_missing_instrument_raises():
    v = view_from(*scenario())
    v.instruments = ("BTCUSDT", "XRPUSDT")
    with pytest.raises(ValueError):
        run(v)
