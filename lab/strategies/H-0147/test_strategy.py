import numpy as np
import pytest

from strategy import PARAMS, simulate_instrument, session_bands, target_positions

BIG = 1e6
SMALL = 1e-3
DAY = 288


def P(**kw):
    p = dict(PARAMS)
    p.update(kw)
    return p


def alt_candle(i):
    p = 99.0 if i % 2 == 0 else 101.0
    return (p, p, p, p, BIG)


FILL_LONG = (102.1, 102.2, 102.0, 102.1, SMALL)

BREAK = [
    (101.0, 101.9, 101.0, 101.8, SMALL),    # 14: close > U (count 1)
    (101.8, 101.85, 101.6, 101.7, SMALL),   # 15: count 2
    (101.7, 102.0, 101.6, 101.9, SMALL),    # 16: count 3 -> acceptance (cannot be b itself)
]
BT = (101.9, 101.95, 101.3, 101.6, SMALL)    # 17: backtest candle b, high_b = 101.95
TRIG = (101.6, 102.1, 101.5, 102.05, SMALL)  # 18: close > high_b -> entry signal


def build(day1_rows, filler=FILL_LONG, start=14, unreal=()):
    """Day 0: 288 alternating 99/101 candles with big volume (V = 100, sigma = 1); day 1: the first `start`
    candles alternate the same way, then `day1_rows`, then `filler`."""
    rows = [alt_candle(i) for i in range(DAY)]
    d1 = [alt_candle(i) for i in range(start)] + list(day1_rows)
    d1 += [filler] * (DAY - len(d1))
    rows += d1
    a = np.array(rows, dtype=np.float64)
    o, h, l, c, vol = a.T
    tp = (h + l + c) / 3.0
    qv = vol * tp
    n = len(rows)
    times = np.arange(n, dtype=np.int64) * 5
    real = np.ones(n, dtype=bool)
    for r in unreal:
        real[DAY + r] = False
    tradable = real.copy()
    member = np.ones(n, dtype=bool)
    return dict(times_min=times, o=o, h=h, l=l, c=c, qv=qv, real=real, tradable=tradable, member=member)


def run(arrs, params=None):
    return simulate_instrument(params=params if params is not None else P(), **arrs)


def mirror(arrs):
    m = dict(arrs)
    m["o"] = 200.0 - arrs["o"]
    m["h"] = 200.0 - arrs["l"]
    m["l"] = 200.0 - arrs["h"]
    m["c"] = 200.0 - arrs["c"]
    tp_old = (arrs["h"] + arrs["l"] + arrs["c"]) / 3.0
    m["qv"] = arrs["qv"] / tp_old * (200.0 - tp_old)
    return m


def ref_atr(h, l, c, real):
    atr = np.nan
    trs = []
    pc = np.nan
    out = np.full(len(h), np.nan)
    for t in range(len(h)):
        if not real[t]:
            out[t] = atr
            continue
        tr = h[t] - l[t] if np.isnan(pc) else max(h[t] - l[t], abs(h[t] - pc), abs(l[t] - pc))
        pc = c[t]
        if len(trs) < 14:
            trs.append(tr)
            if len(trs) == 14:
                atr = sum(trs) / 14
        else:
            atr = (atr * 13 + tr) / 14
        out[t] = atr
    return out


def test_bands_known_values():
    a = build([])
    V, sig, U, L = session_bands(a["times_min"], a["h"], a["l"], a["c"], a["qv"], a["real"], 1.5)
    assert V[DAY + 13] == pytest.approx(100.0, abs=1e-9)
    assert sig[DAY + 13] == pytest.approx(1.0, abs=1e-9)
    assert U[DAY + 13] == pytest.approx(101.5, abs=1e-9)
    assert L[DAY + 13] == pytest.approx(98.5, abs=1e-9)
    # first candle of a session: V = its own tp, sigma = 0 (reset at 00:00)
    assert V[DAY] == pytest.approx(99.0)
    assert sig[DAY] == pytest.approx(0.0, abs=1e-12)


def test_bands_point_in_time():
    a = build(BREAK + [BT, TRIG])
    full = session_bands(a["times_min"], a["h"], a["l"], a["c"], a["qv"], a["real"], 1.5)
    cut = DAY + 20
    part = session_bands(a["times_min"][:cut], a["h"][:cut], a["l"][:cut], a["c"][:cut], a["qv"][:cut],
                         a["real"][:cut], 1.5)
    for f, p in zip(full, part):
        np.testing.assert_array_equal(f[:cut], p)


def test_long_backtest_entry_and_session_end():
    w = run(build(BREAK + [BT, TRIG]))
    d1 = w[DAY:]
    assert np.all(w[:DAY] == 0)
    assert np.all(d1[:18] == 0)          # acceptance candle 16 is not a backtest; 17 is b; 18 is the signal
    assert np.all(d1[18:287] == 0.5)
    assert d1[287] == 0.0                # closed after the 23:55 candle


def test_short_mirror():
    w = run(mirror(build(BREAK + [BT, TRIG])))
    d1 = w[DAY:]
    assert np.all(d1[:18] == 0)
    assert np.all(d1[18:287] == -0.5)
    assert d1[287] == 0.0


def test_cancel_below_vwap_drops_acceptance():
    rows = BREAK + [
        (101.9, 101.9, 99.5, 99.6, SMALL),    # 17: close < V -> cancel
        (99.6, 101.95, 99.6, 101.6, SMALL),   # 18
        (101.6, 102.1, 101.5, 102.05, SMALL),  # 19
    ]
    w = run(build(rows))
    assert np.all(w == 0)


def test_window_lapse_and_inside_window():
    b = (101.9, 103.0, 101.3, 101.6, SMALL)
    wait = (102.3, 102.6, 102.2, 102.5, SMALL)
    pop = (102.5, 103.6, 102.4, 103.5, SMALL)
    # entry on the 6th candle after b: allowed
    w = run(build(BREAK + [b] + [wait] * 5 + [pop]))
    assert w[DAY + 22] == 0 and w[DAY + 23] == 0.5
    # 6 candles pass without an entry: b lapses, the 7th candle closing above high_b is not an entry
    w = run(build(BREAK + [b] + [wait] * 6 + [pop]))
    assert np.all(w == 0)


def test_stop_value_and_exit():
    base = build(BREAK + [BT, TRIG])
    atr = ref_atr(base["h"], base["l"], base["c"], base["real"])
    stop = min(101.3, 101.5) - 0.25 * atr[DAY + 18]
    hold = FILL_LONG
    above = (102.1, 102.2, stop + 0.01, 102.1, SMALL)
    below = (102.1, 102.2, stop - 0.01, 102.1, SMALL)
    w = run(build(BREAK + [BT, TRIG] + [hold] * 6 + [above]))
    assert np.all(w[DAY + 18:DAY + 287] == 0.5)
    w = run(build(BREAK + [BT, TRIG] + [hold] * 6 + [below]))
    assert np.all(w[DAY + 18:DAY + 25] == 0.5)
    assert np.all(w[DAY + 25:] == 0)     # stopped at 25, no second long entry that session


def test_target_2r_only_in_target_mode():
    base = build(BREAK + [BT, TRIG])
    atr = ref_atr(base["h"], base["l"], base["c"], base["real"])
    stop = 101.3 - 0.25 * atr[DAY + 18]
    fill = FILL_LONG[0]                  # open of the candle after the signal
    tgt = fill + 2 * (fill - stop)
    hit = (102.1, tgt + 0.01, 102.0, 102.1, SMALL)
    miss = (102.1, tgt - 0.01, 102.0, 102.1, SMALL)
    a_hit = build(BREAK + [BT, TRIG] + [FILL_LONG] * 6 + [hit])
    a_miss = build(BREAK + [BT, TRIG] + [FILL_LONG] * 6 + [miss])
    w = run(a_hit, P(exit_mode="target_2r"))
    assert w[DAY + 24] == 0.5 and np.all(w[DAY + 25:] == 0)
    w = run(a_miss, P(exit_mode="target_2r"))
    assert np.all(w[DAY + 18:DAY + 287] == 0.5)
    w = run(a_hit, P(exit_mode="session_end"))
    assert np.all(w[DAY + 18:DAY + 287] == 0.5)


def test_aggressive_first_cross():
    a = build(BREAK + [BT, TRIG])
    w = run(a, P(entry_mode="aggressive"))
    assert np.all(w[DAY:DAY + 14] == 0)
    assert np.all(w[DAY + 14:DAY + 287] == 0.5)
    assert w[DAY + 287] == 0


def _single_cross(i):
    rows = [alt_candle(j) for j in range(DAY)]
    rows[i] = (101.0, 102.0, 101.0, 101.8, SMALL)
    return build(rows[i:], filler=None, start=i)


def test_warmup_and_cutoff_aggressive():
    agg = P(entry_mode="aggressive")
    w = run(_single_cross(11), agg)     # closes at 01:00: warm-up candle (the first 12 of the session)
    assert np.all(w == 0)
    w = run(_single_cross(12), agg)
    assert w[DAY + 12] == 0.5
    w = run(_single_cross(263), agg)    # closes at 22:00: last candle allowed
    assert w[DAY + 263] == 0.5
    w = run(_single_cross(264), agg)    # closes at 22:05: after the cutoff
    assert np.all(w == 0)


def test_unreal_rows_are_skipped():
    garbage = (50.0, 200.0, 1.0, 50.0, 1e9)
    w = run(build(BREAK + [BT, garbage, TRIG], unreal=(18,)))
    assert np.all(w[DAY:DAY + 19] == 0)
    assert np.all(w[DAY + 19:DAY + 287] == 0.5)


def test_k_changes_output():
    a = build(BREAK + [BT, TRIG])
    assert np.any(run(a) != 0)
    assert np.all(run(a, P(k=2.0)) == 0)
    # a setup around U = 101 (k = 1) that never closes above U = 101.5 (k = 1.5)
    rows = [
        (101.0, 101.3, 101.0, 101.2, SMALL),
        (101.2, 101.3, 101.1, 101.25, SMALL),
        (101.25, 101.35, 101.15, 101.3, SMALL),   # third close above 101 -> acceptance (k = 1)
        (101.3, 101.35, 100.9, 101.1, SMALL),     # backtest of 101
        (101.1, 101.5, 101.0, 101.4, SMALL),      # close > 101.35 -> entry (k = 1)
    ]
    b = build(rows)
    w1 = run(b, P(k=1.0))
    assert w1[DAY + 17] == 0 and w1[DAY + 18] == 0.5
    assert not np.array_equal(w1, run(b))


def test_point_in_time_truncation():
    a = build(BREAK + [BT, TRIG])
    full = run(a)
    for cut in (DAY + 15, DAY + 18, DAY + 19, DAY + 100):
        part = run({k: v[:cut] for k, v in a.items()})
        np.testing.assert_array_equal(full[:cut], part)


class _View:
    def __init__(self, arrs, names):
        n = len(arrs["c"])
        self.instruments = tuple(names)
        self.times = arrs["times_min"].astype("datetime64[m]")
        stackit = lambda x: np.stack([x] * len(names), axis=1)
        self.o = stackit(arrs["o"]).astype(np.float32)
        self.h = stackit(arrs["h"]).astype(np.float32)
        self.l = stackit(arrs["l"]).astype(np.float32)
        self.c = stackit(arrs["c"]).astype(np.float32)
        self.qv = stackit(arrs["qv"]).astype(np.float32)
        self.real = stackit(arrs["real"])
        self.tradable = stackit(arrs["tradable"])
        self.universe = np.ones((n, len(names)), dtype=bool)

    def col(self, name):
        return self.instruments.index(name)


def test_target_positions_gross_and_columns():
    a = build(BREAK + [BT, TRIG])
    view = _View(a, ["XXXUSDT", "BTCUSDT", "ETHUSDT"])
    w = target_positions(view, PARAMS)
    assert w.shape == (len(a["c"]), 3)
    assert np.all(w[:, 0] == 0)
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)
    assert np.any(w[:, 1] == 0.5) and np.any(w[:, 2] == 0.5)
    with pytest.raises(Exception):
        target_positions(_View(a, ["BTCUSDT"]), PARAMS)


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        run(build([]), P(entry_mode="foo"))
