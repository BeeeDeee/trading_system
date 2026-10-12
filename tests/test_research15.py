from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from qlab.research15 import evaluate as E, strategy as S
from qlab.research15.setup import epoch_15m

T0 = epoch_15m(date(2020, 1, 1))                 # a multiple of 96, hence of 16


def _market(T=24000, seed=0, gap=0.01):
    """Regime-switching random walk on a 15m grid with missing (carried) candles."""
    rng = np.random.default_rng(seed)
    mu = np.repeat(rng.choice([-1, 1], T // 1500 + 1) * 4e-4, 1500)[:T]
    lr = rng.normal(mu, 0.003)
    c = 100 * np.exp(np.cumsum(lr))
    o = np.concatenate([[100.0], c[:-1]])
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0008, T)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0008, T)))
    real = rng.random(T) > gap
    for i in range(1, T):
        if not real[i]:
            o[i] = h[i] = l[i] = c[i] = c[i - 1]
    return {"t": T0 + np.arange(T, dtype=np.int64), "o": o, "h": h, "l": l, "c": c, "real": real,
            "funding": np.full(T, np.nan)}


def _uptrend(T=3200, step=8e-4):
    """Deterministic steady uptrend: every 1h bar breaks out and every candle confirms."""
    c = 100 * (1 + step) ** np.arange(1, T + 1)
    o = np.concatenate([[100.0], c[:-1]])
    return {"t": T0 + np.arange(T, dtype=np.int64), "o": o, "h": c * 1.0001, "l": o * 0.9999, "c": c,
            "real": np.ones(T, bool), "funding": np.full(T, np.nan)}


CFG = S.Config("BRK", 12, 1.5, "LS")


def _slice(d, n):
    return {k: v[:n] for k, v in d.items()}


def test_aggregate_uses_real_candles_and_75_percent_rule():
    o, h, l, c = (np.arange(8, dtype=float) + x for x in (0, 2, -1, 1))
    real = np.array([1, 1, 1, 0, 1, 1, 0, 0], bool)
    O, H, L, C = S.aggregate(o, h, l, c, real, 4)
    assert (O[0], H[0], L[0], C[0]) == (0, 4, -1, 3)        # candle 3 is not real -> last real close is candle 2
    assert C[0] == c[2]
    assert np.isnan(C[1])                                  # 2 of 4 real: incomplete


def test_regime_uses_only_closed_4h_bars():
    d = _market()
    a = S.Asset.build(d)
    c4 = S.aggregate(d["o"], d["h"], d["l"], d["c"], d["real"], 16)[3]
    reg4 = S.regime_4h(c4)
    for k in (15, 16, 17, 31, 5000, 5001, 5014, 5015):
        b = (k + 1) // 16 - 1
        assert a.reg15[k] == (reg4[b] if b >= 0 else 0)
    assert (a.reg15[:15] == 0).all()


@pytest.mark.parametrize("which", ["regime", "confirm", "entries"])
def test_point_in_time(which):
    d = _market()
    full = S.Asset.build(d)
    n = len(d["c"])
    rng = np.random.default_rng(1)
    book_full = S.trade_book(full, CFG, 0, n)
    assert len(book_full.entries) > 20
    for t in rng.integers(8000, n - 5, 6):
        d2 = {k: v.copy() for k, v in d.items()}
        tr = S.Asset.build(_slice(d, t + 1))
        d2["c"][t + 1:] *= rng.uniform(0.6, 1.5, n - t - 1)
        d2["h"][t + 1:] = d2["c"][t + 1:] * 1.01
        d2["l"][t + 1:] = d2["c"][t + 1:] * 0.99
        d2["o"][t + 1:] = d2["c"][t + 1:]
        d2["real"][t + 1:] = rng.random(n - t - 1) > 0.5
        pert = S.Asset.build(d2)
        if which == "regime":
            for x in (tr, pert):
                np.testing.assert_array_equal(x.reg15[:t + 1], full.reg15[:t + 1])
        elif which == "confirm":
            for x in (tr, pert):
                np.testing.assert_array_equal(x.up[:t + 1], full.up[:t + 1])
                np.testing.assert_array_equal(x.dn[:t + 1], full.dn[:t + 1])
        else:                                              # decisions at k <= t-1 never change
            want = book_full.entries <= t - 1
            for x, end in ((tr, t + 1), (pert, n)):
                b = S.trade_book(x, CFG, 0, end)
                got = b.entries <= t - 1
                np.testing.assert_array_equal(b.entries[got], book_full.entries[want])
                np.testing.assert_array_equal(b.sides[got], book_full.sides[want])
                np.testing.assert_allclose(b.risk[got], book_full.risk[want])


def _bars(o, h, l, c, real=None):
    n = len(c)
    return SimpleNamespace(o=np.array(o, float), h=np.array(h, float), l=np.array(l, float),
                           c=np.array(c, float), real=np.ones(n, bool) if real is None else np.array(real, bool),
                           fund=np.zeros(n))


def test_exit_stop_beats_target_in_same_candle():
    a = _bars(o=[100, 100, 100], h=[100, 110, 100], l=[100, 94, 100], c=[100, 100, 100])
    x, price, why = S._exit(a, 0, 1, 95.0, 105.0, 3)       # candle 1 touches both stop 95 and target 105
    assert (x, price, why) == (1, 95.0, S.STOP)


def test_exit_gap_through_stop_and_target():
    a = _bars(o=[100, 90, 100], h=[100, 91, 100], l=[100, 89, 100], c=[100, 90, 100])
    assert S._exit(a, 0, 1, 95.0, 105.0, 3) == (1, 90.0, S.STOP)           # long fills at the open, below the stop
    a = _bars(o=[100, 110, 100], h=[100, 111, 100], l=[100, 109, 100], c=[100, 110, 100])
    assert S._exit(a, 0, 1, 95.0, 105.0, 3) == (1, 110.0, S.TARGET)        # gap above the target fills at the open
    a = _bars(o=[100, 110, 100], h=[100, 111, 100], l=[100, 109, 100], c=[100, 110, 100])
    assert S._exit(a, 0, -1, 105.0, 95.0, 3) == (1, 110.0, S.STOP)         # short: stop above, gap up -> open


def test_exit_short_target_time_and_missing_candle():
    a = _bars(o=[100, 100, 100], h=[100, 101, 100], l=[100, 94, 100], c=[100, 95, 100])
    assert S._exit(a, 0, -1, 105.0, 96.0, 3) == (1, 96.0, S.TARGET)
    a = _bars(o=[100, 100, 100, 100], h=[100, 120, 101, 101], l=[100, 80, 99, 99], c=[100, 100, 100, 100],
              real=[1, 0, 1, 1])                           # the extreme candle is a carried (not real) one
    assert S._exit(a, 0, 1, 95.0, 105.0, 4) == (3, 100.0, S.END)           # forced exit at end-1
    n = S.MAX_HOLD + 3
    a = _bars(o=[100] * n, h=[100.5] * n, l=[99.5] * n, c=[100] * n)
    assert S._exit(a, 0, 1, 90.0, 110.0, n) == (S.MAX_HOLD, 100.0, S.TIME)


def test_bar_returns_by_hand_costs_and_funding():
    a = _bars(o=[100, 100, 101], h=[100, 102, 103], l=[100, 99, 100], c=[100, 101, 102])
    a.fund = np.array([0.0, 0.001, 0.0])
    hr = S._bar_returns(a, 0, 2, 103.0, 1, 7.5e-4, True)   # exit at 103, not at the carried close 102
    assert hr[0] == pytest.approx((101 / 100 - 1) - 0.001 - 7.5e-4)
    assert hr[1] == pytest.approx((103 / 101 - 1) - 7.5e-4)
    hs = S._bar_returns(a, 0, 2, 103.0, -1, 7.5e-4, True)
    assert hs[0] == pytest.approx(-(101 / 100 - 1) + 0.001 - 7.5e-4)
    assert S._bar_returns(a, 0, 2, 103.0, 1, 0.0, False)[0] == pytest.approx(0.01)


def test_funding_default_before_first_record():
    t = T0 + np.arange(200)
    f = np.full(200, np.nan)
    f[96] = 2e-4
    r = S.funding_rate(t, f)
    assert r[0] == S.FUNDING_DEFAULT and r[32] == S.FUNDING_DEFAULT and r[1] == 0.0
    assert r[96] == 2e-4 and r[128] == 0.0                 # after the first record only real data count


def test_trade_book_matches_hand_calculation():
    d = _uptrend()
    a = S.Asset.build(d)
    start = 2600
    b = S.trade_book(a, CFG, start, len(d["c"]))
    assert len(b.entries) > 5
    # independent re-derivation of the first trade
    k0 = next(k for k in range(start, len(d["c"])) if k % 4 == 3 and a.reg15[k] == 1)
    j = k0 // 4
    anchor = d["l"][4 * j:4 * j + 4].min()                 # low of the setup's 1h bar
    e = k0 + 1                                             # every candle confirms -> first candle of the window
    assert b.entries[0] == e and b.sides[0] == 1
    p = d["c"][e]
    assert b.risk[0] == pytest.approx((p - anchor) / p)
    T = p + 1.5 * (p - anchor)
    x = next(i for i in range(e + 1, len(d["c"])) if d["h"][i] >= T)
    assert b.exits[0] == x and b.reason[0] == S.TARGET
    prev = d["c"][e:x].copy()
    cur = d["c"][e + 1:x + 1].copy()
    cur[-1] = max(d["o"][x], T)
    hr = cur / prev - 1
    hr[0] -= S.COST
    hr[-1] -= S.COST
    assert b.ret[0] == pytest.approx(np.prod(1 + hr) - 1)
    assert b.net[e + 1:x + 1].sum() == pytest.approx(hr.sum())
    # B_NO15 enters at the setup close itself
    b0 = S.trade_book(a, CFG, start, len(d["c"]), confirm=False)
    assert b0.entries[0] == k0


def test_no_overlap_and_ordered():
    d = _market()
    a = S.Asset.build(d)
    for cfg in (CFG, S.Config("SWP", 12, 3.0, "LS")):
        b = S.trade_book(a, cfg, 5000, len(d["c"]))
        assert len(b.entries) > 5
        assert (b.entries[1:] > b.exits[:-1]).all()
        assert (b.exits > b.entries).all()
        assert ((b.risk >= S.R_MIN) & (b.risk <= S.R_MAX)).all()


def test_regime_filter_and_long_only():
    d = _market()
    a = S.Asset.build(d)
    b = S.trade_book(a, CFG, 5000, len(d["c"]))
    assert (a.reg15[b.entries] == b.sides).all()
    bl = S.trade_book(a, S.Config("BRK", 12, 1.5, "LONG"), 5000, len(d["c"]))
    assert (bl.sides == 1).all()
    nb = S.trade_book(a, CFG, 5000, len(d["c"]), use_regime=False)
    assert len(nb.entries) > len(b.entries)                # without the filter more trades, both sides anywhere


def test_setup_is_dropped_when_the_anchor_breaks():
    d = _uptrend()
    a = S.Asset.build(d)
    start = 2600
    k0 = next(k for k in range(start, len(d["c"])) if k % 4 == 3 and a.reg15[k] == 1)
    anchor = d["l"][4 * (k0 // 4):k0 + 1].min()
    d2 = {k: v.copy() for k, v in d.items()}
    d2["l"][k0 + 1] = anchor * 0.999                       # the candle after the setup pierces the anchor
    a2 = S.Asset.build(d2)
    b = S.trade_book(a2, CFG, start, len(d["c"]))
    assert b.entries[0] != k0 + 1 and (b.entries > k0).all()
    # entering at the setup close (no confirmation) does not look at the later pierce
    b0 = S.trade_book(a2, CFG, start, len(d["c"]), confirm=False)
    assert b0.entries[0] == k0


def test_delay_skips_trade_if_anchor_pierced_meanwhile():
    d = _uptrend()
    a = S.Asset.build(d)
    start = 2600
    k0 = next(k for k in range(start, len(d["c"])) if k % 4 == 3 and a.reg15[k] == 1)
    anchor = d["l"][4 * (k0 // 4):k0 + 1].min()
    b1 = S.trade_book(a, CFG, start, len(d["c"]), delay=1)
    assert b1.entries[0] == k0 + 2                         # fill one candle after the confirmation
    d2 = {k: v.copy() for k, v in d.items()}
    d2["l"][k0 + 2] = anchor * 0.999                       # pierced between confirmation and the delayed fill
    b2 = S.trade_book(S.Asset.build(d2), CFG, start, len(d["c"]), delay=1)
    assert b2.entries[0] != k0 + 2


def test_allowed_mask_restricts_entries_only():
    d = _market()
    a = S.Asset.build(d)
    n = len(d["c"])
    full = S.trade_book(a, CFG, 5000, n)
    allowed = (a.t // 96) % 2 == 0                         # every other UTC day
    b = S.trade_book(a, CFG, 5000, n, allowed=allowed)
    assert allowed[b.entries].all() and len(b.entries) < len(full.entries)
    assert (b.exits > b.entries).any() and (b.entries[1:] > b.exits[:-1]).all()
    np.testing.assert_array_equal(S.trade_book(a, CFG, 5000, n, allowed=np.ones(n, bool)).entries, full.entries)


def test_random_book_matches_counts_regime_and_no_overlap():
    d = _market()
    a = S.Asset.build(d)
    n = len(d["c"])
    ref = S.trade_book(a, CFG, 5000, n)
    rb = S.random_book(a, CFG, ref, 5000, n, np.random.default_rng(3))
    key = lambda b: sorted(zip(a.month[b.entries].tolist(), b.sides.tolist()))
    assert len(rb.entries) >= 0.9 * len(ref.entries)
    assert (a.reg15[rb.entries] == rb.sides).all()
    assert (rb.entries[1:] > rb.exits[:-1]).all()
    assert set(rb.risk.round(12)) <= set(ref.risk.round(12))
    from collections import Counter
    need, got = Counter(key(ref)), Counter(key(rb))
    assert all(got[k] <= v for k, v in need.items())
    rb2 = S.random_book(a, CFG, ref, 5000, n, np.random.default_rng(3))
    np.testing.assert_array_equal(rb.entries, rb2.entries)                # seeded


def test_portfolio_and_4h_benchmark_run():
    d1, d2 = _market(seed=1), _market(seed=2)
    assets = {"A": S.Asset.build(d1), "B": S.Asset.build(d2)}
    n = len(d1["c"])
    r = E.run(assets, CFG, 5000, n)
    assert len(r.dates) == len(r.daily) and np.isfinite(r.daily).all()
    s = E.stats(r.daily, r.all_trades())
    assert s["n_trades"] == sum(len(b.entries) for b in r.books.values())
    # daily sum equals the sum of the 15m sleeves weighted 1/2
    tot = 0.5 * sum(b.net[5000:n].sum() for b in r.books.values())
    assert r.daily.sum() == pytest.approx(tot)
    _, h4 = E.run_4h(assets, "LS", 5000, n)
    assert np.isfinite(h4).all()
    _, hold = E.run_hold(assets, 5000, n)
    assert len(hold) == len(r.daily)
    rr = E.run_random(assets, CFG, r, 5000, n, seed=0)
    assert np.isfinite(rr.daily).all()
    assert set(E.trade_breakdown(r)["reasons"]) == {"stop", "target", "time", "end"}


def test_basket_enters_only_in_member_months_with_fixed_weights():
    ds = {"A": _market(seed=5), "B": _market(seed=6)}
    first = int(np.datetime64(int(ds["A"]["t"][0]) * 900, "s").astype("datetime64[M]").astype(int))
    months = [f"{1970 + m // 12}-{m % 12 + 1:02d}" for m in range(first, first + 12)]
    members = {m: (["A"] if i % 2 == 0 else ["B"]) for i, m in enumerate(months)}
    gen = ((s, S.Asset.build(d)) for s, d in ds.items())
    dates, daily, trades = E.run_basket(gen, members, CFG, date(2020, 1, 1))
    assert len(dates) == len(daily) and np.isfinite(daily).all() and len(trades) > 5
    a = S.Asset.build(ds["A"])
    allowed = np.array([f"{1970 + m // 12}-{m % 12 + 1:02d}" in [k for k, v in members.items() if "A" in v]
                        for m in a.month])
    b = S.trade_book(a, CFG, 0, len(a.t), allowed=allowed)
    assert allowed[b.entries].all()
    # weight 0.1 per sleeve: scaling the weight scales the daily series
    _, d2, _ = E.run_basket(((s, S.Asset.build(d)) for s, d in ds.items()), members, CFG, date(2020, 1, 1),
                            weight=0.2)
    np.testing.assert_allclose(d2, 2 * daily)


def test_grid_is_16_unique_configs():
    g = S.grid()
    assert len(g) == 16 and len({c.id for c in g}) == 16
