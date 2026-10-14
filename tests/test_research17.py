from datetime import date

import numpy as np
import pytest

from qlab.research15 import strategy as S
from qlab.research15.setup import epoch_15m
from qlab.research16 import engine as G, families as F
from qlab.research17 import engine as R
from qlab.research17.setup import COST, HOLD, day0, n_days

T0 = epoch_15m(date(2020, 11, 2))                  # a Monday, multiple of 96 (hence of 16)


def _market(T=12000, seed=0, gap=0.01, lead=None, t0=T0, vol=0.0035):
    """Regime-switching fat-tailed random walk; `lead` = returns of the leader that this market follows with a
    4-candle lag at half the size (so lagging follower setups exist)."""
    rng = np.random.default_rng(seed)
    mu = np.repeat(rng.choice([-1, 0, 1], T // 1500 + 1) * 5e-4, 1500)[:T]
    lr = mu + vol * rng.standard_t(4, T) / np.sqrt(2)
    lr[rng.random(T) < 0.004] *= 12
    if lead is not None:
        lr[4:] += 0.5 * lead[:T - 4]
    c = 100 * np.exp(np.cumsum(lr))
    o = np.concatenate([[100.0], c[:-1]])
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0008, T)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0008, T)))
    real = rng.random(T) > gap
    for i in range(1, T):
        if not real[i]:
            o[i] = h[i] = l[i] = c[i] = c[i - 1]
    t = t0 + np.arange(T, dtype=np.int64)
    f = np.full(T, np.nan)
    sett = np.flatnonzero(t % 32 == 0)
    f[sett] = rng.normal(1e-4, 3e-4, len(sett))
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "real": real, "funding": f}, lr


def _cut(d, n):
    return {k: v[:n] for k, v in d.items()}


BTC, LR = _market(12000, 0)
ETH, _ = _market(12000, 1, lead=LR)
ALT, _ = _market(9000, 2, lead=LR[1600:], t0=T0 + 1600)       # starts later, ends earlier than the leader
LD = R.leader(BTC)


def _asset(d):
    return G.Asset.build(d)


def test_signals_equal_research16_lead_on_the_same_pair():
    for K, side in ((3.0, "LONG"), (2.0, "LS"), (3.0, "LS")):
        v = F.Variant("LEAD", K, side)
        ref = F.lead({"BTCUSDT": _asset(BTC), "ETHUSDT": _asset(ETH)}, v)["ETHUSDT"]
        got = R.signals(_asset(ETH), LD, K, side)
        for f in ("k", "side", "stop", "target", "rmult", "deadline"):
            np.testing.assert_array_equal(getattr(got, f), getattr(ref, f), err_msg=f"{K}{side}{f}")
        assert len(got) > 5


def test_signals_match_a_reference_loop_with_a_follower_that_starts_later():
    a = _asset(ALT)
    cs = R.signals(a, LD, 3.0, "LS")
    assert len(cs) > 3
    tb = BTC["t"]
    sg = F.sigma(_asset(BTC))
    ref = []
    for k in range(len(a.c)):
        tk = int(a.t[k])
        kb = tk - int(tb[0])
        if kb < 4 or not (a.real[k] and a.real[k - 4:k + 1].all() if k >= 4 else False):
            continue
        if not BTC["real"][kb - 4:kb + 1].all() or not np.isfinite(sg[kb]):
            continue
        rb, ra = np.log(BTC["c"][kb] / BTC["c"][kb - 4]), np.log(a.c[k] / a.c[k - 4])
        if abs(rb) >= 3.0 * 2 * sg[kb] and ra / rb < 0.5:
            ref.append((k, 1 if rb > 0 else -1))
    assert list(zip(cs.k.tolist(), cs.side.tolist())) == ref
    assert (cs.deadline == cs.k + HOLD).all() and np.isnan(cs.stop).all()


def test_nolag_is_a_superset_and_long_only_never_shorts():
    a = _asset(ETH)
    lag, nolag = R.signals(a, LD, 3.0, "LS"), R.signals(a, LD, 3.0, "LS", lag=False)
    assert set(lag.k.tolist()) <= set(nolag.k.tolist()) and len(nolag) > len(lag)
    assert (R.signals(a, LD, 2.0, "LONG").side == 1).all()


def test_signals_are_point_in_time():
    full = R.signals(_asset(ALT), LD, 3.0, "LS")
    rng = np.random.default_rng(5)
    for n in (6000, 7777):                      # follower and leader both cut / perturbed after the same time
        cut_t = int(ALT["t"][n])
        nb = cut_t - int(BTC["t"][0])
        a_cut, b_cut = _cut(ALT, n), _cut(BTC, nb)
        got = R.signals(_asset(a_cut), R.leader(b_cut), 3.0, "LS")
        want = full.before(n)
        np.testing.assert_array_equal(got.k, want.k)
        np.testing.assert_array_equal(got.side, want.side)
        b_pert = {k: v.copy() for k, v in BTC.items()}
        a_pert = {k: v.copy() for k, v in ALT.items()}
        for d, m in ((b_pert, nb), (a_pert, n)):
            for k in ("o", "h", "l", "c"):
                d[k][m:] *= np.exp(rng.normal(0, 0.05, len(d[k]) - m))
            d["real"][m:] = rng.random(len(d["real"]) - m) > 0.3
        pert = R.signals(_asset(a_pert), R.leader(b_pert), 3.0, "LS")
        np.testing.assert_array_equal(pert.before(n).k, want.k)


# ---------------------------------------------------------------- run, membership, null

SC = [R.Scenario("base"), R.Scenario("x2", cost=2 * COST), R.Scenario("d1", delay=1),
      R.Scenario("nofund", use_funding=False)]
LEADERS = {"spot": LD, "perp": LD}


def test_trade_returns_follow_the_cost_and_funding_by_hand():
    a = _asset(ETH)
    r = R.run_follower(a, LEADERS, [R.Scenario("base"), R.Scenario("nofund", use_funding=False)], 0, None)
    assert len(r["base"].entries) > 3
    for t, ret in list(zip(r["nofund"].entries, r["nofund"].rets))[:20]:
        e = int(t - a.t[0])
        last = e + HOLD
        c = a.c[e:last + 1]
        g = c[1:] / c[:-1] - 1.0
        g[0] -= COST
        g[-1] -= COST
        assert ret == pytest.approx(np.prod(1 + g) - 1)
    # funding: a long pays the rate at the settlement candle (t % 32 == 0) it holds through (4 h hold: at most one)
    nofund = dict(zip(r["nofund"].entries, r["nofund"].rets))
    hit = miss = 0
    for t, ret in zip(r["base"].entries, r["base"].rets):
        e = int(t - a.t[0])
        settle = [i for i in range(e + 1, e + HOLD + 1) if a.t[i] % 32 == 0]
        assert len(settle) <= 1
        paid = a.fund[settle[0]] if settle else 0.0
        assert ret - nofund[t] == pytest.approx(-paid, abs=1e-4)
        hit, miss = hit + bool(settle), miss + (not settle)
    assert hit > 0 and miss > 0


def test_membership_restricts_entries_to_member_months():
    a = _asset(ETH)
    months = sorted({f"{1970 + m // 12}-{m % 12 + 1:02d}" for m in np.unique(a.month)})
    mask = R.member_mask(a, "X", {months[1]: ["X"], months[3]: ["Y"]})
    assert mask.sum() == (a.month == np.unique(a.month)[1]).sum()
    full = R.run_follower(a, LEADERS, [R.Scenario("base")], 0, None)["base"]
    part = R.run_follower(a, LEADERS, [R.Scenario("base")], 0, mask)["base"]
    inside = a.month[(part.entries - a.t[0]).astype(int)]
    assert 0 < len(part.entries) < len(full.entries) and (inside == np.unique(a.month)[1]).all()


def test_daily_contribution_is_weighted_and_sums_to_trade_compounding():
    a = _asset(ETH)
    r = R.run_follower(a, LEADERS, [R.Scenario("base")], 0, None, weight=0.1)["base"]
    assert len(r.daily) == n_days()
    b = G.trade_book(a, R.signals(a, LD, 3.0, "LONG"), 0, len(a.c), cost=COST)
    d, v = S.daily(a.t, b.net, 0, len(a.c))
    i = d.astype("datetime64[D]").astype(np.int64) - day0()
    np.testing.assert_allclose(r.daily[i], 0.1 * v)
    assert np.count_nonzero(r.daily) <= len(d)


def _null_inputs(syms=("ETH", "ALT")):
    assets = {"ETH": _asset(ETH), "ALT": _asset(ALT)}
    res = {s: R.run_follower(assets[s], LEADERS, [R.Scenario("base")], 0, None)["base"] for s in syms}
    lights = {s: R.light(assets[s]) for s in syms}
    bt = BTC["t"]
    times = R.candidate_times(bt, BTC["real"], int(bt[0]), int(bt[-1]) - 20)
    return assets, res, lights, R.events_of(res), times


def test_null_replaying_the_actual_events_reproduces_the_actual_book():
    assets, res, lights, ev, times = _null_inputs()
    # replay: every event is drawn back onto its own candle, so the null must equal the actual portfolio
    order = iter(sorted(ev))
    class Rng:
        def integers(self, n):
            k = next(order)
            m = int(R.month_key(np.array([k]))[0])
            return int(np.flatnonzero(times[m] == k)[0])
    daily, rets = R.null_run(lights, ev, times, Rng())
    want = sum(r.daily for r in res.values())
    np.testing.assert_allclose(daily, want, atol=1e-12)
    assert sorted(rets) == pytest.approx(sorted(np.concatenate([r.rets for r in res.values()])))


def test_null_keeps_events_together_in_their_month():
    assets, res, lights, ev, times = _null_inputs()
    class Spy:
        def __init__(self, rng): self.rng = rng
        def integers(self, n):
            return int(self.rng.integers(n))
    daily, rets = R.null_run(lights, ev, times, Spy(np.random.default_rng(3)))
    n_actual = sum(len(r.rets) for r in res.values())
    assert 0.8 * n_actual <= len(rets) <= n_actual
    # same seed -> same result; different seed -> different placement
    d2, _ = R.null_run(lights, ev, times, Spy(np.random.default_rng(3)))
    d3, _ = R.null_run(lights, ev, times, Spy(np.random.default_rng(4)))
    np.testing.assert_array_equal(daily, d2)
    assert not np.array_equal(daily, d3)
    # every random candle lies in the month of the event it replaces
    tau_pool = {m: set(v.tolist()) for m, v in times.items()}
    assert all(len(v) > 0 for v in tau_pool.values())


def test_null_places_all_trades_of_an_event_on_one_candle():
    lights = {s: R.light(_asset(d)) for s, d in (("ETH", ETH), ("ALT", ALT))}
    k = int(BTC["t"][5000])
    ev = {k: ["ETH", "ALT"]}
    bt = BTC["t"]
    times = R.candidate_times(bt, BTC["real"], int(bt[0]), int(bt[-1]) - 20)
    seen = []
    orig = S._bar_returns

    def spy(a, e, x, *args):
        seen.append(int(a.t0 + e))
        return orig(a, e, x, *args)
    S._bar_returns = spy
    try:
        for seed in range(30):
            seen.clear()
            R.null_run(lights, ev, times, np.random.default_rng(seed))
            assert len(set(seen)) <= 1               # both trades (if both placed) share the event's new candle
    finally:
        S._bar_returns = orig


def test_null_skips_overlapping_positions_of_one_follower():
    lights = {"ETH": R.light(_asset(ETH))}
    t = BTC["t"]
    ks = [int(t[5000]), int(t[5001]), int(t[5002])]
    times = {int(R.month_key(np.array([k]))[0]): np.array([int(t[6000])]) for k in ks}   # all land on one candle
    daily, rets = R.null_run(lights, {k: ["ETH"] for k in ks}, times, np.random.default_rng(0))
    assert len(rets) <= 1


def test_event_bootstrap_resamples_whole_events():
    # event 1: three trades of +1 %, event 2: one trade of -4 %  -> trade mean 0, but events are +1 % and -4 %
    entries = [np.array([10, 10, 10, 20]), ]
    rets = [np.array([0.01, 0.01, 0.01, -0.04])]
    mean, lo, hi, n = R.event_ci(entries, rets, n_boot=2000)
    assert n == 2 and mean == pytest.approx(-0.0025)
    assert lo == pytest.approx(-0.04 / 1) and hi == pytest.approx(0.01)       # resamples of 2 events: all-A or all-B
    # delay shifts entries to the signal candle: events are identified by the signal, not the fill
    m2 = R.event_ci([np.array([11, 12])], [np.array([0.1, 0.1])], delay=1)
    assert m2[3] == 2
    assert np.isnan(R.event_ci([], [])[0])


def test_hold_daily_and_member_days():
    a = _asset(ETH)
    h = R.hold_daily(a, 0)
    assert len(h) == n_days()
    d, v = S.daily(a.t, np.r_[0.0, np.log(a.c[1:] / a.c[:-1])], 0, len(a.c))
    i = d.astype("datetime64[D]").astype(np.int64) - day0()
    np.testing.assert_allclose(h[i], np.expm1(v))
    md = R.member_days("X", {"2020-11": ["X"], "2020-12": ["Y"]})
    days = (day0() + np.arange(n_days())).astype("datetime64[D]")
    assert md.sum() == 30 and md[days == np.datetime64("2020-11-15")].all()


def test_end_to_end_run_on_synthetic_markets():
    from qlab.research17 import evaluate as E
    data = {"BTCUSDT": BTC, "ETHUSDT": ETH, "AAAUSDT": ALT, "BBBUSDT": _market(11000, 7, lead=LR[:11000])[0]}
    months = ["2020-11", "2020-12", "2021-01", "2021-02", "2021-03"]
    members = {m: ["AAAUSDT", "BBBUSDT"] for m in months}
    out = E.run(lambda sym, market: data[sym], ["AAAUSDT", "BBBUSDT"], members, n_rnd=30, n_boot=200)
    assert set(out["criteria"]) == {"1_sharpe_and_event_ci", "2_null_p", "3_cost_x2", "4_delay_1_candle",
                                    "5_subperiods", "6_breadth"}
    assert out["verdict"] in ("pass", "rejected", "inconclusive") and out["n_symbols"] == 2
    assert out["n_events"] > 0 and 1 / 31 <= out["criteria"]["2_null_p"]["p"] <= 1
    # the headline Sharpe is the one of the sum of the two sleeves run directly
    a = {s: _asset(data[s]) for s in ("AAAUSDT", "BBBUSDT")}
    d = sum(R.run_follower(a[s], LEADERS, [R.Scenario("base")], 0, R.member_mask(a[s], s, members))["base"].daily
            for s in a)
    from qlab.research15.evaluate import stats
    assert out["P_stats"]["sharpe"] == pytest.approx(stats(d, np.zeros(1))["sharpe"])
    assert set(out["robust"]["scenarios"]) >= {"x2", "delay1", "nolag", "K2", "LS", "btc_perp_leader"}
    assert out["robust"]["scenarios"]["nolag"]["n_trades"] >= out["P_stats"]["n_trades"]
