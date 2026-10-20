import numpy as np
import pytest

from strategy import target_positions, PARAMS

T0 = 3300        # break candle
QUIET = 3200     # first quiet (compressed) candle
TT = 3400


class FakeView:
    def __init__(self, d, bpd=96):
        self.o = d["c"].astype(np.float32)
        self.h = (d["c"] * (1 + d["sp"])).astype(np.float32)
        self.l = (d["c"] * (1 - d["sp"])).astype(np.float32)
        self.c = d["c"].astype(np.float32)
        self.qv = d["qv"].astype(np.float32)
        self.tbq = (d["qv"] * d["share"]).astype(np.float32)
        self.nt = np.ones_like(self.c)
        self.funding = np.zeros_like(self.c)
        self.real = d["real"]
        self.tradable = d["real"].copy()
        self.listed = np.ones_like(d["real"])
        self.universe = d["uni"]
        self.bars_per_day = bpd
        self.instruments = tuple("A%d" % j for j in range(d["c"].shape[1]))

    def col(self, name):
        return self.instruments.index(name)


def make(n=1, seed=0):
    rng = np.random.default_rng(seed)
    c = np.empty((TT, n))
    c[:QUIET] = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal((QUIET, n)), axis=0))
    cq = c[QUIET - 1].copy()
    c[QUIET:] = cq
    sp = np.full((TT, n), 0.005)
    sp[QUIET:] = 0.0005
    d = dict(c=c, sp=sp, qv=np.full((TT, n), 1000.0), share=np.full((TT, n), 0.5),
             real=np.ones((TT, n), dtype=bool), uni=np.ones((TT, n), dtype=bool))
    return d, cq


def long_break(d, cq, j=0, qv_mult=3.0):
    d["c"][T0, j] = cq[j] * 1.003
    d["c"][T0 + 1:, j] = cq[j] * 1.004
    d["qv"][T0, j] = 1000.0 * qv_mult
    d["share"][T0, j] = 0.7


def run(d, **kw):
    p = dict(PARAMS)
    p.update(kw)
    return target_positions(FakeView(d), p)


def test_long_entry_and_time_stop():
    d, cq = make()
    long_break(d, cq)
    P = run(d)
    H = PARAMS["hold_hours"] * 4
    assert np.all(P[:T0] == 0)
    assert np.all(P[T0:T0 + H, 0] == 0.25)
    assert np.all(P[T0 + H:, 0] == 0)


def test_hold_hours_param():
    d, cq = make()
    long_break(d, cq)
    P = run(d, hold_hours=3)
    assert np.all(P[T0:T0 + 12, 0] == 0.25)
    assert np.all(P[T0 + 12:, 0] == 0)


def test_short_entry():
    d, cq = make()
    d["c"][T0, 0] = cq[0] * 0.997
    d["c"][T0 + 1:, 0] = cq[0] * 0.996
    d["qv"][T0, 0] = 3000.0
    d["share"][T0, 0] = 0.3
    P = run(d)
    assert np.all(P[T0:T0 + 24, 0] == -0.25)
    assert np.all(P[T0 + 24:, 0] == 0)
    assert np.all(P[:T0] == 0)


def test_failed_break_exit():
    d, cq = make()
    long_break(d, cq)
    d["c"][T0 + 5:, 0] = cq[0] * 0.999          # back below the range midpoint (= cq)
    P = run(d)
    assert np.all(P[T0:T0 + 5, 0] == 0.25)
    assert np.all(P[T0 + 5:, 0] == 0)


def test_universe_exit_and_entry():
    d, cq = make()
    long_break(d, cq)
    d["uni"][T0 + 3, 0] = False
    P = run(d)
    assert np.all(P[T0:T0 + 3, 0] == 0.25)
    assert np.all(P[T0 + 3:, 0] == 0)
    d, cq = make()
    long_break(d, cq)
    d["uni"][T0, 0] = False
    assert np.all(run(d) == 0)


@pytest.mark.parametrize("field,value", [("share", 0.5), ("qv", 1500.0)])
def test_flow_confirmation_required(field, value):
    d, cq = make()
    long_break(d, cq)
    d[field][T0, 0] = value
    assert np.all(run(d) == 0)


def test_gap_in_window_skips():
    d, cq = make()
    long_break(d, cq)
    d["real"][3250, 0] = False
    assert np.all(run(d) == 0)
    d, cq = make()
    long_break(d, cq)
    d["real"][T0, 0] = False                    # a gap candle never triggers
    assert np.all(run(d) == 0)


def test_compression_param():
    d, cq = make()
    long_break(d, cq)
    assert np.any(run(d, compression_ratio=0.5) != 0)
    assert np.all(run(d, compression_ratio=0.0) == 0)


def test_range_hours_param():
    d, cq = make()
    long_break(d, cq)
    assert np.any(run(d, range_hours=12) != 0)
    # 48h window reaches into the noisy period, whose highs are above the break close
    assert np.all(run(d, range_hours=48) == 0)


def test_exclusion_of_extended_moves():
    d, cq = make()
    long_break(d, cq)
    d["c"][T0:, 0] = cq[0] * 1.5                 # a 40 % hour, far beyond 4 sigma
    assert np.all(run(d) == 0)


def test_slots_ranked_by_volume_ratio():
    mults = [2.5, 7.0, 3.0, 6.0, 5.0, 4.0]
    d, cq = make(n=6, seed=1)
    for j, m in enumerate(mults):
        long_break(d, cq, j=j, qv_mult=m)
    P = run(d)
    entered = [j for j in range(6) if P[T0, j] != 0]
    assert entered == [1, 3, 4, 5]
    assert np.all(np.abs(P).sum(axis=1) <= 1.0 + 1e-12)


def test_cooldown_blocks_reentry():
    d, cq = make()
    long_break(d, cq)
    d["c"][T0 + 1, 0] = cq[0] * 1.002
    d["c"][T0 + 2, 0] = cq[0] * 0.999          # failed break exit at T0+2
    d["c"][T0 + 3:, 0] = cq[0] * 1.01           # new break inside the cooldown
    d["qv"][T0 + 3, 0] = 3000.0
    d["share"][T0 + 3, 0] = 0.7
    P = run(d)
    assert np.all(P[T0:T0 + 2, 0] == 0.25)
    assert np.all(P[T0 + 2:, 0] == 0)
    # without the first entry the second break is taken
    d["share"][T0, 0] = 0.5
    P = run(d)
    assert np.all(P[:T0 + 3, 0] == 0)
    assert np.all(P[T0 + 3:T0 + 27, 0] == 0.25)
    assert np.all(P[T0 + 27:, 0] == 0)


def test_short_history_no_signal():
    d, cq = make()
    long_break(d, cq)
    d["real"][:T0 - 2880 + 1, 0] = False        # first real candle less than 30 days before T0
    d["c"][:T0 - 2880 + 1, 0] = np.nan
    assert np.all(run(d) == 0)


def test_truncation_is_causal():
    d, cq = make(n=6, seed=1)
    for j, m in enumerate([2.5, 7.0, 3.0, 6.0, 5.0, 4.0]):
        long_break(d, cq, j=j, qv_mult=m)
    full = run(d)
    cut = T0 + 7
    dt = {k: v[:cut].copy() for k, v in d.items()}
    assert np.array_equal(run(dt), full[:cut])


def test_synthetic_view_gross_and_values():
    from lab.framework.intraday import synthetic_view
    v = synthetic_view()
    P = target_positions(v, PARAMS)
    assert P.shape == (len(v.times), len(v.instruments))
    assert np.all(np.abs(P).sum(axis=1) <= 1.0 + 1e-12)
    assert set(np.unique(np.abs(P))) <= {0.0, 0.25}
