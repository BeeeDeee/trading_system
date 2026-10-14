from collections import Counter
from datetime import date, datetime, timezone

import numpy as np
import pytest

from qlab.research15 import strategy as S
from qlab.research15.setup import epoch_15m
from qlab.research16 import engine as G, evaluate as E, families as F

T0 = epoch_15m(date(2020, 11, 2))                  # a Monday, multiple of 96


def _market(T=12000, seed=0, gap=0.01, lead=None):
    """Regime-switching fat-tailed random walk with occasional shocks, missing candles and funding records.
    `lead`: returns of another market that this one follows with a 4-candle lag (half the size)."""
    rng = np.random.default_rng(seed)
    mu = np.repeat(rng.choice([-1, 0, 1], T // 1500 + 1) * 5e-4, 1500)[:T]
    vol = np.repeat(rng.choice([0.0015, 0.004], T // 2000 + 1), 2000)[:T]
    lr = mu + vol * rng.standard_t(4, T) / np.sqrt(2)
    lr[rng.random(T) < 0.004] *= 12
    if lead is not None:
        lr[4:] += 0.5 * lead[:-4]
    c = 100 * np.exp(np.cumsum(lr))
    o = np.concatenate([[100.0], c[:-1]])
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0008, T)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0008, T)))
    real = rng.random(T) > gap
    for i in range(1, T):
        if not real[i]:
            o[i] = h[i] = l[i] = c[i] = c[i - 1]
    t = T0 + np.arange(T, dtype=np.int64)
    f = np.full(T, np.nan)
    sett = np.flatnonzero(t % 32 == 0)
    f[sett] = np.where(rng.random(len(sett)) < 0.5, 1e-4, rng.normal(1e-4, 3e-4, len(sett)))
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "real": real, "funding": f}, lr


def _pair(T=12000):
    b, lr = _market(T, 0)
    e, _ = _market(T, 1, lead=lr)
    return {"BTCUSDT": b, "ETHUSDT": e}


def _assets(raw):
    return {s: G.Asset.build(d) for s, d in raw.items()}


def _cut(raw, n):
    return {s: {k: v[:n] for k, v in d.items()} for s, d in raw.items()}


def _perturb(raw, n, seed=9):
    rng = np.random.default_rng(seed)
    out = {}
    for s, d in raw.items():
        d = {k: v.copy() for k, v in d.items()}
        m = np.exp(rng.normal(0, 0.05, len(d["c"]) - n))
        for k in ("o", "h", "l", "c"):
            d[k][n:] *= m
        d["funding"][n:] = np.where(np.isfinite(d["funding"][n:]), rng.normal(0, 1e-3, len(m)), np.nan)
        d["real"][n:] = rng.random(len(m)) > 0.3
        out[s] = d
    return out


def _same(a: G.Cand, b: G.Cand):
    for f in ("k", "side", "stop", "target", "rmult", "deadline"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f), err_msg=f)


RAW = _pair()
FULL = _assets(RAW)


@pytest.mark.parametrize("v", F.grid(), ids=lambda v: v.id)
def test_every_variant_is_point_in_time_and_fires(v):
    full = F.signals(FULL, v)
    assert sum(len(c) for c in full.values()) > 0, "vacuous: no candidates on the synthetic market"
    for n in (9000, 10001, 11507):
        trunc = F.signals(_assets(_cut(RAW, n)), v)
        pert = F.signals(_assets(_perturb(RAW, n)), v)
        for s in RAW:
            _same(trunc[s], full[s].before(n))
            _same(pert[s].before(n), full[s].before(n))


def test_grid_is_30_unique_variants():
    g = F.grid()
    assert len(g) == 30 and len({v.id for v in g}) == 30
    assert [v.id for v in g[:4]] == ["ORB_H1_LS", "ORB_H1_LONG", "ORB_H4_LS", "ORB_H4_LONG"]
    assert {v.id for v in g if v.family == "CME"} == {"CME_G1", "CME_G2"}
    assert F.Variant("RANGE", 2.5, "LS").id == "RANGE_M25_LS" and F.Variant("FUND", 0.95, "LS").id == "FUND_Q95_LS"


def test_long_only_variants_never_short():
    for v in F.grid():
        if v.side == "LONG":
            assert all((c.side == 1).all() for c in F.signals(FULL, v).values())


# ---------------------------------------------------------------- engine by hand

def _bars(o, h, l, c, real=None, t0=T0):
    n = len(c)
    return G.Asset.build({"t": t0 + np.arange(n, dtype=np.int64), "o": np.array(o, float), "h": np.array(h, float),
                          "l": np.array(l, float), "c": np.array(c, float),
                          "real": np.ones(n, bool) if real is None else np.array(real, bool),
                          "funding": np.full(n, np.nan)})


def test_trade_book_by_hand_rmult_target_and_costs():
    # entry close[0]=100, stop 98 (risk 2 %), target 2R = 104 reached at candle 3 (open 103, high 105)
    a = _bars([100, 100, 101, 103, 104], [100, 101, 102, 105, 105], [100, 99, 100, 102, 103],
              [100, 101, 102, 104, 104])
    c = G.Cand.make([0], [1], [4], stop=[98.0], rmult=[2.0])
    b = G.trade_book(a, c, 0, 5, use_funding=False)
    assert (b.entries[0], b.exits[0], b.reason[0]) == (0, 3, G.TARGET)
    exp = (101 / 100 - 1 - G.COST, 102 / 101 - 1, 104 / 102 - 1 - G.COST)
    np.testing.assert_allclose(b.net[1:4], exp)
    np.testing.assert_allclose(b.ret[0], np.prod(1 + np.array(exp)) - 1)
    assert b.risk[0] == pytest.approx(0.02)


def test_trade_book_deadline_no_stop_and_funding():
    # no stop/target: held to the deadline (candle 33), the short receives the rate at settlement candle 32
    n = 40
    c = 100 + np.arange(n, dtype=float)
    a = _bars(c, c + 0.5, c - 0.5, c)
    a.fund[:] = 0.0
    a.fund[32] = 3e-4
    b = G.trade_book(a, G.Cand.make([30], [-1], [33]), 0, n)
    assert (b.exits[0], b.reason[0]) == (33, G.TIME) and np.isnan(b.risk[0])
    exp = [-(c[31] / c[30] - 1) - G.COST, -(c[32] / c[31] - 1) + 3e-4, -(c[33] / c[32] - 1) - G.COST]
    np.testing.assert_allclose(b.net[31:34], exp)


def test_trade_book_gap_through_stop_risk_bounds_and_blocking():
    a = _bars([100, 100, 95, 96, 96, 96], [100, 100, 96, 97, 97, 97], [100, 99, 94, 95, 95, 95],
              [100, 99.5, 95, 96, 96, 96])
    c = G.Cand.make([0, 1, 3, 4], [1, 1, 1, 1], [5, 5, 5, 5], stop=[98, 99, 95.999, 90], rmult=[2, 2, 2, 2])
    b = G.trade_book(a, c, 0, 6, use_funding=False)
    # trade 1: gap through the stop at candle 2 -> filled at the open 95; candle 1 is blocked; candle 3 has a
    # risk below 0.15 % (skipped); candle 4 enters with risk 6.25 % > 5 % (skipped)
    assert list(b.entries) == [0] and b.exits[0] == 2 and b.reason[0] == G.STOP
    assert b.net[2] == pytest.approx(95 / 99.5 - 1 - G.COST)


def test_delay_fills_one_candle_later_and_skips_touched_stop():
    a = _bars([100] * 6, [101, 101, 101, 101, 101, 101], [99.5, 99.5, 97, 99.5, 99.5, 99.5], [100, 100.5, 100, 100, 100, 100])
    c = G.Cand.make([0], [1], [5], stop=[99.0], rmult=[2.0])
    b = G.trade_book(a, c, 0, 6, delay=1, use_funding=False)
    assert list(b.entries) == [1]
    c2 = G.Cand.make([1], [1], [5], stop=[98.0], rmult=[2.0])
    assert len(G.trade_book(a, c2, 0, 6, delay=1).entries) == 0     # candle 2 touched the stop before the fill


# ---------------------------------------------------------------- families by hand / reference recomputation

def test_orb_first_breakout_stop_and_deadline():
    n = 96 * 2
    c = np.full(n, 100.0)
    h, l = c + 0.2, c - 0.2
    c[10], h[10] = 99.0, 99.2          # inside? no: below the range low 99.8 -> first break is short at 10
    l[10] = 98.9
    c[20], h[20] = 101.0, 101.1
    a = _bars(c, h, l, c)
    ls = F.orb({"X": a}, F.Variant("ORB", 1, "LS"))["X"]
    assert list(ls.k) == [10] and ls.side[0] == -1 and ls.stop[0] == pytest.approx(100.2) and ls.deadline[0] == 95
    lo = F.orb({"X": a}, F.Variant("ORB", 1, "LONG"))["X"]
    assert list(lo.k) == [20] and lo.stop[0] == pytest.approx(99.8)
    assert len(F.orb({"X": a}, F.Variant("ORB", 4, "LS"))["X"]) == 1    # range of 4 h includes candle 10


def test_cme_times_follow_chicago_dst():
    t = epoch_15m(date(2021, 1, 1)) + np.arange(96 * 220, dtype=np.int64)
    Fc, K, D = F.cme_weeks(t)
    utc = lambda i: datetime.fromtimestamp(int(t[i]) * 900, timezone.utc)
    w = {str(d): (utc(f), utc(k)) for f, k, d in zip(Fc, K, D)}
    fj, kj = w["2021-01-08"]                       # CST: close 22:00 UTC (candle 21:45), reopen Sun 23:00 UTC
    assert (fj.weekday(), fj.hour, fj.minute, kj.weekday(), kj.hour) == (4, 21, 45, 6, 23)
    fs, ks = w["2021-07-09"]                       # CDT: one hour earlier
    assert (fs.hour, fs.minute, ks.hour) == (20, 45, 22)


def test_cme_trades_towards_friday_close_and_respects_listing():
    c = FULL["ETHUSDT"]
    cm = F.cme({"ETHUSDT": c, "BTCUSDT": FULL["BTCUSDT"]}, F.Variant("CME", 0.01, "BOTH"))
    Fc, K, D = F.cme_weeks(c.t)
    friday = dict(zip(K, Fc))
    for sym, x in cm.items():
        a = FULL[sym]
        for k, s, st, tg in zip(x.k, x.side, x.stop, x.target):
            f = friday[k]
            g = a.c[k] / a.c[f] - 1
            assert abs(g) >= 0.01 and s == -np.sign(g) and tg == a.c[f]
            assert st == pytest.approx(a.c[k] + np.sign(g) * 0.5 * abs(a.c[k] - a.c[f]))
    first_eth = (cm["ETHUSDT"].k.min() + c.t[0]) * 900
    assert np.datetime64(int(first_eth), "s") >= np.datetime64("2021-02-14")
    assert len(cm["BTCUSDT"]) > len(cm["ETHUSDT"])


def test_shock_matches_reference():
    a = FULL["BTCUSDT"]
    v = F.Variant("SHOCK", 4, "LS")
    x = F.shock({"X": a}, v)["X"]
    lr = np.log(a.c[1:] / a.c[:-1])
    ref = []
    for k in range(F.SIGMA_N + 4, len(a.c)):
        w = [lr[i - 1] for i in range(k - F.SIGMA_N + 1, k + 1) if a.real[i] and a.real[i - 1]]
        if len(w) < 0.75 * F.SIGMA_N or not a.real[k - 4:k + 1].all():
            continue
        m, sg = np.log(a.c[k] / a.c[k - 4]), np.std(w, ddof=1)
        if abs(m) > 4 * 2 * sg:
            ref.append((k, 1 if m < 0 else -1))
    assert list(zip(x.k.tolist(), x.side.tolist())) == ref and len(ref) > 5
    i = 0
    assert x.target[i] == a.c[x.k[i] - 4] and x.deadline[i] == x.k[i] + 16


def test_fund_matches_reference_with_strict_inequalities():
    a = FULL["BTCUSDT"]
    v = F.Variant("FUND", 0.9, "LS")
    x = F.fund({"X": a}, v)["X"]
    idx = np.flatnonzero(np.isfinite(a.fraw))
    ref = []
    for i in range(F.FUND_N, len(idx)):
        prev, r, k = a.fraw[idx[i - F.FUND_N:i]], a.fraw[idx[i]], idx[i]
        if not a.real[k]:
            continue
        if r > np.quantile(prev, 0.9):
            ref.append((k, -1))
        elif r < np.quantile(prev, 0.1):
            ref.append((k, 1))
    assert list(zip(x.k.tolist(), x.side.tolist())) == ref and len(ref) > 5
    assert (x.deadline == x.k + 288).all() and np.isnan(x.stop).all()


def test_lead_matches_reference_and_trades_eth_only():
    v = F.Variant("LEAD", 2, "LS")
    x = F.lead(FULL, v)
    assert len(x["BTCUSDT"]) == 0 and len(x["ETHUSDT"]) > 5
    B, Et = FULL["BTCUSDT"], FULL["ETHUSDT"]
    sg = F.sigma(B)
    for k, s in zip(x["ETHUSDT"].k, x["ETHUSDT"].side):
        rb, re = np.log(B.c[k] / B.c[k - 4]), np.log(Et.c[k] / Et.c[k - 4])
        assert abs(rb) >= 4 * sg[k] and re / rb < 0.5 and s == np.sign(rb)
        assert B.real[k - 4:k + 1].all() and Et.real[k - 4:k + 1].all()


def test_range_and_sqz_levels_follow_the_last_closed_1h_band():
    a = FULL["ETHUSDT"]
    for v, fn in ((F.Variant("RANGE", 2.0, "LS"), F.range_), (F.Variant("SQZ", 0.2, "LS"), F.sqz)):
        x = fn({"X": a}, v)["X"]
        assert len(x) > 5
        for i in range(0, len(x), max(len(x) // 40, 1)):
            k, s = int(x.k[i]), int(x.side[i])
            j = (k - 3) // 4
            w = a.c1[j - 19:j + 1]
            mid, sd = w.mean(), w.std()
            m, d = (v.p, s) if v.family == "RANGE" else (2.0, -s)      # RANGE fades the band, SQZ breaks it
            assert (a.c[k] < mid - m * sd) if d == 1 else (a.c[k] > mid + m * sd)
            if v.family == "RANGE":
                assert a.reg15[k] == 0 and x.target[i] == pytest.approx(mid) and x.deadline[i] == k + 16
                assert x.stop[i] == pytest.approx(2 * a.c[k] - mid)
            else:
                wd = 4 * np.array([a.c1[i_ - 19:i_ + 1].std() / a.c1[i_ - 19:i_ + 1].mean()
                                   for i_ in range(j - 720, j + 1)])
                assert wd[-1] <= np.nanquantile(wd[:-1], 0.2) + 1e-15
                assert x.stop[i] == pytest.approx(mid) and x.rmult[i] == 2.0


def test_pull_requires_recent_rsi_setup_regime_and_confirmation():
    a = FULL["BTCUSDT"]
    v = F.Variant("PULL", 40, "LS")
    x = F.pull({"X": a}, v)["X"]
    rs = F.rsi(a.c1)
    assert len(x) > 5
    for k, s, st in zip(x.k, x.side, x.stop):
        k = int(k)
        js = [j for j in range(len(rs)) if 4 * j + 4 <= k <= 4 * j + 19]
        ok = [j for j in js if (rs[j] <= 40 if s == 1 else rs[j] >= 60) and a.reg15[4 * j + 3] == s]
        assert ok and a.reg15[k] == s and (a.up[k] if s == 1 else a.dn[k])
        w = slice(k - 31, k + 1)
        assert st == (np.where(a.real[w], a.l[w], np.inf).min() if s == 1 else np.where(a.real[w], a.h[w], -np.inf).max())


# ---------------------------------------------------------------- null, portfolio, holm

@pytest.mark.parametrize("fam", ["ORB", "FUND", "CME", "PULL", "RANGE", "SHOCK"])
def test_random_book_respects_context_counts_and_no_overlap(fam):
    v = next(v for v in F.grid() if v.family == fam)
    cands = F.signals(FULL, v)
    n = len(FULL["BTCUSDT"].c)
    ref = E.run(FULL, cands, 0, n)
    rr = E.run_random(FULL, v, ref, 0, n, seed=3)
    for s, a in FULL.items():
        rb, pb, ctx = rr.books[s], ref.books[s], F.context(a, v)
        assert (rb.entries[1:] > rb.exits[:-1]).all()
        if fam == "CME":
            assert set(rb.entries) <= set(pb.entries)
            continue
        for e, sd, x in zip(rb.entries, rb.sides, rb.exits):
            assert (ctx.long if sd == 1 else ctx.short)[e]
            if fam == "ORB":
                assert a.t[x] // 96 == a.t[e] // 96
        key = lambda b: Counter(zip(a.month[b.entries].tolist(), b.sides.tolist()))
        if fam == "FUND":                             # 72 h holds on 3 candles a day: not always feasible
            assert key(rb) <= key(pb) and rb.entries.size >= 0.8 * pb.entries.size
        else:
            assert key(rb) == key(pb)


def test_holm_step_down():
    h = G.holm({"a": 0.001, "b": 0.02, "c": 0.04})
    assert h["a"]["pass"] and h["a"]["threshold"] == pytest.approx(0.05 / 3)
    assert h["b"]["pass"] and h["b"]["threshold"] == pytest.approx(0.025)
    assert h["c"]["pass"]
    h = G.holm({"a": 0.001, "b": 0.03, "c": 0.04})
    assert h["a"]["pass"] and not h["b"]["pass"] and not h["c"]["pass"]


def test_portfolio_null_and_fund_start():
    v = F.Variant("SHOCK", 4, "LS")
    n = len(FULL["BTCUSDT"].c)
    r = E.run(FULL, F.signals(FULL, v), 0, n)
    assert len(r.dates) == n // 96 and len(r.all_trades()) > 5
    out = E.null(FULL, v, r, 0, n, n=20)
    assert 1 / 21 <= out["p"] <= 1
    assert E.start_index(FULL, F.Variant("FUND", 0.9, "LS"), date(2018, 2, 1)) == 0      # 2020-04-01 < T0
    assert E.start_index(FULL, F.Variant("FUND", 0.9, "LS"), date(2020, 12, 1)) == epoch_15m(date(2020, 12, 1)) - T0
    assert E.start_index(FULL, F.Variant("ORB", 1, "LS"), date(2020, 12, 1)) == epoch_15m(date(2020, 12, 1)) - T0


def test_basket_enters_only_in_member_months():
    v = F.Variant("SHOCK", 4, "LS")
    a = FULL["BTCUSDT"]
    months = sorted({f"{1970 + m // 12}-{m % 12 + 1:02d}" for m in np.unique(a.month)})
    members = {months[1]: ["BTCUSDT"]}
    dates, d, tr = E.run_basket(iter([("BTCUSDT", a)]), members, v, date(2020, 11, 2), 0.1)
    full = G.trade_book(a, F.signals({"X": a}, v)["X"], 0, len(a.t))
    inside = full.entries[a.month[full.entries] == np.unique(a.month)[1]]
    assert 0 < len(tr) <= len(inside) + 1
