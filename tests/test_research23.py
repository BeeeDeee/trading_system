import numpy as np
import pytest

from qlab.research23 import engine as G, evaluate as E, setups as SU, vwap as VW
from qlab.research23.data import PER_DAY


def synth(days: int = 12, seed: int = 0, t0: int = 50 * PER_DAY) -> dict:
    """Random-walk 5m candles with volume, a few missing candles and funding at 00/08/16 UTC."""
    rng = np.random.default_rng(seed)
    n = days * PER_DAY
    lr = rng.standard_t(4, n) * 0.0015 + np.sin(np.arange(n) / 40.0) * 0.0004
    c = 100.0 * np.exp(np.cumsum(lr))
    o = np.r_[100.0, c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.002, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.002, n))
    v = rng.gamma(2.0, 50.0, n)
    real = np.ones(n, bool)
    real[rng.choice(n, 20, replace=False)] = False
    v[~real] = 0.0
    t = t0 + np.arange(n, dtype=np.int64)
    fund = np.where(t % 96 == 0, 1e-4, 0.0)
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v, "qv": v * (h + l + c) / 3, "real": real, "fund": fund}


def cut(d: dict, n: int) -> dict:
    return {k: x[:n].copy() for k, x in d.items()}


def test_session_vwap_matches_naive():
    d = synth(3)
    p = VW.bar_price(d["h"], d["l"], d["c"], d["v"], d["qv"])
    V, S = VW.session_vwap(d["t"], p, d["v"])
    for t in (5, PER_DAY + 7, 2 * PER_DAY + 200):
        s0 = t - d["t"][t] % PER_DAY
        w, x = d["v"][s0:t + 1], p[s0:t + 1]
        m = (w * x).sum() / w.sum()
        assert V[t] == pytest.approx(m, rel=1e-12)
        assert S[t] == pytest.approx(np.sqrt((w * (x - m) ** 2).sum() / w.sum()), rel=1e-6)


@pytest.mark.parametrize("v", SU.grid(), ids=lambda v: v.id)
def test_signals_are_point_in_time(v):
    d = synth(12, seed=3)
    full = SU.signals(E.prepare(d)["mk"], v)
    assert len(full) > 0, "synthetic data should produce signals for every variant"
    for n in (3 * PER_DAY + 101, 7 * PER_DAY + 5, 11 * PER_DAY + 250):
        dd = d.copy()
        rng = np.random.default_rng(n)                       # perturb the future: past signals must not move
        for k in ("o", "h", "l", "c", "v"):
            dd[k] = d[k].copy()
            dd[k][n:] *= rng.uniform(0.9, 1.1, len(d[k]) - n)
        dd["qv"] = dd["v"] * (dd["h"] + dd["l"] + dd["c"]) / 3
        for src in (cut(d, n), dd):
            s = SU.signals(E.prepare(src)["mk"], v)
            m, mf = s.k < n, full.k < n
            np.testing.assert_array_equal(s.k[m], full.k[mf])
            np.testing.assert_array_equal(s.side[m], full.side[mf])
            np.testing.assert_allclose(s.stop[m], full.stop[mf])
            np.testing.assert_allclose(s.band[m], full.band[mf])


@pytest.mark.parametrize("v", SU.grid()[:16], ids=lambda v: v.id)
def test_mirror_swaps_sides(v):
    d = synth(8, seed=5)
    m = {k: d[k] for k in d}
    m["o"], m["c"], m["h"], m["l"] = 1000 - d["o"], 1000 - d["c"], 1000 - d["l"], 1000 - d["h"]
    m["qv"] = m["v"] * (m["h"] + m["l"] + m["c"]) / 3
    a, b = SU.signals(E.prepare(d)["mk"], v), SU.signals(E.prepare(m)["mk"], v)
    ka = sorted(zip(a.k.tolist(), a.side.tolist()))
    kb = sorted(zip(b.k.tolist(), (-b.side).tolist()))
    assert ka == kb


def flat(n: int = PER_DAY, price: float = 100.0) -> dict:
    z = np.full(n, price)
    return {"o": z.copy(), "h": z.copy(), "l": z.copy(), "c": z.copy(), "real": np.ones(n, bool),
            "fund": np.zeros(n), "pos": np.arange(n, dtype=np.int64) % PER_DAY}


def book(a, k, side, stop, tg=np.nan, rm=np.nan, delay=0, cost=0.0, end=None):
    return G.Book(*G._book(a["o"], a["h"], a["l"], a["c"], a["real"], a["fund"], a["pos"], np.array([k]),
                           np.array([side]), np.array([stop]), np.array([tg]), rm, 0, end or len(a["c"]),
                           delay, cost))


def test_engine_target_stop_gap_and_session_end():
    a = flat()
    a["h"][12] = 102.5                                       # target 102 hit at row 12
    b = book(a, 10, 1, 99.0, tg=102.0)
    assert (b.x[0], b.why[0]) == (12, G.TARGET) and b.ret[0] == pytest.approx(0.02)
    a = flat()
    a["o"][11], a["l"][11], a["c"][11] = 98.0, 97.5, 98.0    # gap through the stop fills at the open
    b = book(a, 10, 1, 99.0, tg=102.0)
    assert (b.x[0], b.why[0]) == (11, G.STOP) and b.ret[0] == pytest.approx(-0.02)
    a = flat()
    a["h"][11], a["l"][11] = 103.0, 98.0                     # both in one candle: stop first
    assert book(a, 10, 1, 99.0, tg=102.0).why[0] == G.STOP
    a = flat()
    a["c"][PER_DAY - 1] = 101.0                              # no exit: closed at the session's last close
    b = book(a, 10, 1, 99.0, cost=7.5e-4)
    assert (b.x[0], b.why[0]) == (PER_DAY - 1, G.END) and b.ret[0] == pytest.approx(0.01 - 1.5e-3)
    a = flat()
    a["h"][15] = 104.0                                       # 2R: entry 100, stop 99 -> target 102
    b = book(a, 10, 1, 99.0, rm=2.0)
    assert b.why[0] == G.TARGET and b.ret[0] == pytest.approx(0.02)
    b = book(flat(), 10, -1, 99.0)                           # stop on the wrong side: skipped
    assert len(b) == 0
    a = flat()
    a["l"][11] = 98.0                                        # delay 1: stop touched before the fill -> skipped
    assert len(book(a, 10, 1, 99.0, delay=1)) == 0


def test_engine_funding_and_short():
    a = flat()
    a["fund"][96] = 1e-3                                     # 08:00 settlement while short: receives
    a["c"][PER_DAY - 1] = 99.0
    b = book(a, 50, -1, 101.0)
    assert b.ret[0] == pytest.approx(0.01 + 1e-3)


def test_one_position_at_a_time():
    a = flat()
    a["h"][20] = 103.0
    ks = np.array([10, 15, 20, 21])
    b = G.Book(*G._book(a["o"], a["h"], a["l"], a["c"], a["real"], a["fund"], a["pos"], ks, np.ones(4, np.int64),
                        np.full(4, 99.0), np.full(4, 102.0), np.nan, 0, PER_DAY, 0, 0.0))
    assert b.e.tolist() == [10, 21]                          # 15 and 20 fall inside the first trade (exit at 20)


def test_null_keeps_count_side_and_day():
    d = synth(10, seed=7)
    assets = {"X": E.prepare(d)}
    v = SU.Variant("S2", "TST", "VWAP")
    r = E.run(assets, v, E.signals(assets, v), 0, len(d["c"]))
    assert len(r.trades()) > 5
    a, b = assets["X"], r.books["X"]
    sums, cnts = G.null_sums(a, b, 0, len(d["c"]), n_rep=20, seed=1, cost=0.0)
    assert (cnts == len(b)).all()
    nl = E.null(assets, r, 0, len(d["c"]), n=20)
    assert 0 < nl["p"] <= 1 and nl["placed"] == pytest.approx(1.0)


def test_daily_is_sum_of_trades():
    d = synth(10, seed=9)
    assets = {"X": E.prepare(d), "Y": E.prepare(synth(10, seed=10))}
    v = SU.Variant("S1", "BRK", "2R")
    r = E.run(assets, v, E.signals(assets, v), 0, len(d["c"]))
    assert len(r.daily) == 10
    assert r.daily.sum() == pytest.approx(0.5 * r.trades().sum())


def test_grid_has_twenty_variants_and_four_primaries():
    g = SU.grid()
    assert len(g) == 20 and len({v.id for v in g}) == 20
    assert sum(v.primary and v.k == 1.0 for v in g) == 4
