"""Research 19: signals (F4)."""

import numpy as np
import pytest

from qlab.data.panel import Panel
from qlab.data.synthetic import generate_market
from qlab.data.normalize import normalize_prices
from qlab.data.panel import panel_from_bars
from qlab.features.basic import tr_index
from qlab.research19.signals import (donchian_state, ensemble, realized_vol, rsi_wilder, sma, tom_window)
from qlab.validation.leakage import assert_point_in_time, check_point_in_time


def col(x):
    return np.asarray(x, dtype=float)[:, None]


def ones(x):
    return np.ones((len(x), 1), dtype=bool)


def test_donchian_hand_example():
    P = col([1, 2, 3, 4, 3.4, 3.6, 2, 5])
    s, st = donchian_state(P, ones(P), 3)
    # t=3: 4 > max(1,2,3)=3 -> enter, stop = mid(2..4) = 3; t=4: stop -> 3.5 and 3.4 < 3.5 -> exit
    # t=7: 5 > max(3.4,3.6,2)... window P[4..6] = 3.4,3.6,2 -> 3.6 -> enter, stop = mid(3.6,2,5) = 3.5
    assert s[:, 0].tolist() == [0, 0, 0, 1, 0, 0, 0, 1]
    np.testing.assert_allclose(st[[3, 7], 0], [3.0, 3.5])
    assert np.isnan(st[[0, 1, 2, 4, 5, 6], 0]).all()


def test_donchian_needs_strict_break_and_stop_never_falls():
    P = col([1, 2, 3, 3])                                  # equals the maximum -> no entry
    assert donchian_state(P, ones(P), 3)[0].sum() == 0
    rng = np.random.default_rng(1)
    P = np.exp(np.cumsum(rng.normal(0.002, 0.02, (600, 5)), axis=0))
    s, st = donchian_state(P, np.ones(P.shape, bool), 20)
    assert s.sum() > 0
    for j in range(5):
        run = s[:, j] == 1
        d = np.diff(st[:, j])
        both = run[1:] & run[:-1]
        assert (d[both] >= -1e-12).all()


def test_donchian_invalid_window_resets():
    P = col(np.arange(1.0, 13.0))
    v = ones(P)
    v[6, 0] = False
    s, st = donchian_state(P, v, 3)
    assert s[5, 0] == 1 and s[6:10, 0].sum() == 0          # flat until 4 valid days after the gap
    assert np.isnan(st[6:10, 0]).all() and s[10, 0] == 1


def test_ensemble_levels():
    P = col(np.arange(1.0, 40.0))
    e = ensemble(P, ones(P), (3, 10))
    assert e[3, 0] == 0.5 and e[-1, 0] == 1.0 and e[0, 0] == 0.0


def test_rsi_hand_table():
    P = col([10, 11, 10, 9, 10, 12])
    r = rsi_wilder(P, ones(P), 2)[:, 0]
    assert np.isnan(r[:2]).all()
    np.testing.assert_allclose(r[2:], [50.0, 25.0, 62.5, 87.5])
    Q = col([1, 2, 3])
    assert rsi_wilder(Q, ones(Q), 2)[2, 0] == 100.0        # AL = 0


def test_rsi_resets_over_gap():
    P = col([10, 11, 10, 9, 10, 12, 11, 12])
    v = ones(P)
    v[4, 0] = False
    r = rsi_wilder(P, v, 2)[:, 0]
    assert np.isnan(r[4:7]).all() and not np.isnan(r[7])


def test_sma_and_vol():
    P = col([1, 2, 3, 4, 5])
    np.testing.assert_allclose(sma(P, ones(P), 3)[:, 0], [np.nan, np.nan, 2, 3, 4])
    v = ones(P)
    v[2, 0] = False
    assert np.isnan(sma(P, v, 3)[:5, 0]).all()
    Q = col([100, 110, 99, 108.9])
    rv = realized_vol(Q, ones(Q), 3, 252)[:, 0]
    r = np.array([0.1, -0.1, 0.1])
    np.testing.assert_allclose(rv[3], r.std(ddof=1) * np.sqrt(252))
    assert np.isnan(rv[:3]).all()


def test_tom_window_weekend_and_holiday():
    d = np.array(["2024-02-29", "2024-03-01", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-26",
                  "2024-03-27", "2024-03-28", "2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04"],
                 dtype="datetime64[D]")
    # Feb 29 = last day of Feb (day -1); Mar 1, 4, 5 = +1..+3; Mar 29 is a holiday so Thu Mar 28 is day -1
    # of March; Apr 1, 2, 3 = +1..+3; Apr 4 is outside the window
    assert tom_window(d).tolist() == [True, True, True, True, False, False, False, True, True, True, True, False]
    assert not tom_window(d[:8])[7]                              # last row of the calendar: month end unknown


@pytest.fixture(scope="module")
def panel():
    m = generate_market(n_assets=20, seed=19, start="2010-01-01", end="2012-12-31")
    return panel_from_bars(normalize_prices(m.prices, m.delistings, m.calendar), m.calendar)


def _vp(p: Panel):
    return tr_index(p), np.asarray(p.listed & p.tradable)


def test_signals_are_point_in_time(panel):
    fns = {"ensemble": lambda p: ensemble(*_vp(p), (5, 20, 60))[:, :5],
           "rsi2": lambda p: rsi_wilder(*_vp(p), 2),
           "vol": lambda p: realized_vol(*_vp(p), 30, 252),
           "sma": lambda p: sma(*_vp(p), 50)}
    for name, fn in fns.items():
        assert_point_in_time(fn, panel, name, min_day=100)


# ================================ F5: strategies, controls, evaluation ======================================

from qlab.research10.sim import monthly_mix  # noqa: E402
from qlab.research19 import combine, crypto_trend as CT, etf_trend as ET, evaluate as EV, overlays as OV  # noqa: E402
from qlab.research19 import rsi2 as R2, setup as S  # noqa: E402
from qlab.research5.sim import Candidates  # noqa: E402


def crypto_panel(T=700, seed=3, rising=True):
    """3 assets: A rises steadily, B falls steadily, C is noise. Daily calendar, all listed and tradable."""
    rng = np.random.default_rng(seed)
    r_oc = np.column_stack([0.003 + rng.normal(0, 1e-4, T), -0.003 + rng.normal(0, 1e-4, T),
                            rng.normal(0, 0.03, T)])
    z = np.zeros((T, 3))
    dates = np.arange(np.datetime64("2018-01-01"), np.datetime64("2018-01-01") + T)
    ok = np.ones((T, 3), bool)
    qv = np.full((T, 3), 1e9)
    p = Panel(dates, np.arange(3), z.copy(), r_oc, ok, ok.copy(), np.zeros((T, 3), bool),
              np.exp(np.cumsum(r_oc, axis=0)), qv.copy())
    return p, qv, ["AUSDT", "BUSDT", "CUSDT"]


SMALL = dict(lookbacks=(5, 10, 20, 30, 60), universe_n=3, min_age=30)


def test_etf_cost_is_three_bps_and_pre_decimal_multiplier():
    d = np.array(["2001-04-06", "2001-04-09", "2010-01-04"], dtype="datetime64[D]")
    c = ET.etf_cost(d, (3, 2))
    np.testing.assert_allclose(c[:, 0], [0.00075, 0.0003, 0.0003])
    np.testing.assert_allclose(ET.etf_cost(d, (3, 2), 2.0)[2], 0.0006)


def test_h1a_known_trends_and_gross_below_one():
    p, qv, syms = crypto_panel()
    cfg = CT.Config(**SMALL)
    E, sig = CT.signals(p, cfg)
    assert E[-1, 0] == 1.0 and E[-1, 1] == 0.0
    members = CT.monthly_members(p, qv, 3, 30, CT.excluded_mask(syms))
    w = CT.target_weights(E, sig, members, cfg)
    assert w.sum(axis=1).max() <= 1.0 + 1e-12
    assert w[-1, 0] > 0 and w[-1, 1] == 0
    res = CT.run_panel(p, qv, syms, cfg, first=100)
    assert res.nav[-1] > 1.0 and res.exposure.max() <= 1.0 + 1e-9


def test_h1a_exclusion_and_age_apply_before_the_cut():
    p, qv, syms = crypto_panel()
    qv[:, 0] = 5e9                                          # A is the most liquid
    excl = np.array([True, False, False])
    cum = np.cumsum(np.asarray(p.listed), axis=0)
    u = CT.universe_aged(p, qv, 400, 2, 30, cum, excl)
    assert u.tolist() == [1, 2] or sorted(u.tolist()) == [1, 2]      # A excluded, next two take its place
    assert len(CT.universe_aged(p, qv, 400, 3, 500, cum, np.zeros(3, bool))) == 0   # nobody is 500 days old


def test_h1a_band_changes_turnover_not_entries():
    p, qv, syms = crypto_panel()
    a = CT.run_panel(p, qv, syms, CT.Config(band=None, **SMALL), first=100)
    b = CT.run_panel(p, qv, syms, CT.Config(band=0.2, **SMALL), first=100)
    assert np.array_equal(a.n_positions, b.n_positions)
    assert b.turnover.sum() < a.turnover.sum()


def test_h1a_btc_replication_and_delay():
    p, qv, syms = crypto_panel()
    cfg = CT.Config(btc_only=True, **SMALL)
    syms2 = ["BTCUSDT", "BUSDT", "CUSDT"]
    r0 = CT.run_panel(p, qv, syms2, cfg, first=100)
    r1 = CT.run_panel(p, qv, syms2, CT.Config(btc_only=True, delay=1, **SMALL), first=100)
    assert r0.nav[-1] > 1.0 and r1.exposure[101] == 0.0 and r0.exposure[101] > 0.0   # fill one day later


def test_h1a_whole_pipeline_is_point_in_time(panel):
    syms = [f"S{i}USDT" for i in range(panel.shape[1])]
    cfg = CT.Config(lookbacks=(5, 20), universe_n=5, min_age=20, band=0.2)

    def nav(p):
        qv = np.nan_to_num(np.asarray(p.dollar_volume))
        return CT.run_panel(p, qv, syms, cfg, first=70).nav[:, None]

    assert np.nanstd(nav(panel)) > 0
    assert check_point_in_time(nav, panel, "h1a", n_cuts=6, min_day=80).ok


@pytest.fixture(scope="module")
def clean_panel():
    m = generate_market(n_assets=15, seed=21, start="2010-01-01", end="2012-12-31", p_halt=0, p_no_open=0,
                        p_delist=0)
    return panel_from_bars(normalize_prices(m.prices, m.delistings, m.calendar), m.calendar)


def test_h2_runs_holds_at_most_ten_days_and_is_point_in_time(clean_panel):
    panel = clean_panel
    tickers = [f"T{i}" for i in range(panel.shape[1])]
    cfg = R2.Config(tickers=tuple(tickers[:12]), max_positions=4)
    out = R2.run_panel(panel, tickers, cfg, start=210, cash_ret=None)
    tr = out.trades
    assert tr.height > 0 and tr["held"].max() <= 10
    assert (tr["entry_day"] == tr["signal_day"]).all()                       # MOC entry
    assert (tr["entry_value"] <= 0.25 * out.nav.max() * 1.0001).all()                  # adv cap never binds

    def nav(p):
        o = R2.run_panel(p, tickers, cfg, start=210, cash_ret=None)
        x = np.full((p.shape[0], 1), np.nan)
        x[210:, 0] = o.nav
        return x

    assert check_point_in_time(nav, panel, "h2", n_cuts=6, min_day=220).ok


def test_h2_entry_rules_and_ties():
    T = 260
    rng = np.random.default_rng(0)
    r = rng.normal(0.003, 0.01, (T, 2))
    r[-3:, 0] = -0.03                                        # asset 0 sells off at the end, above SMA200
    p = Panel(np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-01") + T), np.arange(2),
              np.zeros((T, 2)), r, np.ones((T, 2), bool), np.ones((T, 2), bool), np.zeros((T, 2), bool),
              np.exp(np.cumsum(r, axis=0)), np.full((T, 2), 1e9))
    ind = R2.indicators(p, np.arange(2))
    cond, score = R2.entry_mask(ind, R2.Config())
    assert cond[-1, 0] and ind.rsi[-1, 0] < 10 and ind.P[-1, 0] > ind.sma200[-1, 0]
    c = R2.candidates(ind, R2.Config(), ["B", "A"], start=210)
    assert (np.diff(c.t) >= 0).all()


def test_c2_keeps_holding_lengths_and_assets():
    regime = np.ones((300, 3), bool)
    rng = np.random.default_rng(19)
    a, h = np.array([0, 1, 2, 0]), np.array([3, 10, 1, 7])
    c = R2.c2_candidates(a, h, regime, 20, 290, rng)
    assert sorted(zip(c.a.tolist(), c.hold.tolist())) == sorted(zip(a.tolist(), h.tolist()))
    assert c.t.min() >= 20 and c.t.max() < 289 and (np.diff(c.t) >= 0).all()


def test_c1_regime_long_weights_sum_to_one(clean_panel):
    panel = clean_panel
    tickers = [f"T{i}" for i in range(panel.shape[1])]
    res = R2.c1_regime_long(panel, tickers, R2.Config(tickers=tuple(tickers[:6])), 250, None)
    assert res.exposure.max() <= 1.0 + 1e-9 and res.exposure[260:].max() > 0.9


def test_align_to_calendar_weekend_rolls_into_monday():
    cd = np.arange(np.datetime64("2024-03-01"), np.datetime64("2024-03-08"))        # Fri .. Thu
    r = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    us = np.array(["2024-03-01", "2024-03-04", "2024-03-05"], dtype="datetime64[D]")
    out = combine.align_to_calendar(cd, r, us)
    np.testing.assert_allclose(out, [0.01, 1.02 * 1.03 * 1.04 - 1, 0.05])


def test_mix_matches_monthly_mix_without_cost_and_charges_reset_cost():
    rng = np.random.default_rng(2)
    d = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-01") + 90)
    R = rng.normal(0.001, 0.01, (90, 3))
    w = np.array([0.5, 0.3, 0.2])
    np.testing.assert_allclose(combine.mix(R, d, w, 0.0), monthly_mix(R, w, d), atol=1e-12)
    g = combine.mix(R, d, w, 3.0)
    first_of_feb = int(np.flatnonzero(d == np.datetime64("2024-02-01"))[0]) + 1     # reset after the close of the 1st
    drift = w * np.prod(1 + R[1:first_of_feb], axis=0)           # last reset: index 1
    drift /= drift.sum()
    expected = (1 + monthly_mix(R, w, d)[first_of_feb]) * (1 - 3e-4 * np.abs(w - drift).sum()) - 1
    assert g[first_of_feb] == pytest.approx(expected, abs=1e-12)
    assert g[5] == pytest.approx(monthly_mix(R, w, d)[5], abs=1e-12)


def test_inverse_vol_weights():
    rng = np.random.default_rng(0)
    past = np.column_stack([rng.normal(0, 0.01, 400), rng.normal(0, 0.02, 400)])
    w = combine.inverse_vol_weights(252)(past)
    assert w.sum() == pytest.approx(1.0) and w[0] > w[1]
    np.testing.assert_allclose(combine.inverse_vol_weights(252)(past[:10]), [0.5, 0.5])


def test_alpha_hac_recovers_alpha():
    rng = np.random.default_rng(5)
    b = rng.normal(0, 0.03, 3000)
    r = 0.001 + 0.5 * b + rng.normal(0, 0.005, 3000)
    a, beta, t = EV.alpha_hac(r, b, 30)
    assert a == pytest.approx(0.001, abs=4e-4) and beta == pytest.approx(0.5, abs=0.02) and t > 2
    a0, _, t0 = EV.alpha_hac(0.5 * b + rng.normal(0, 0.005, 3000), b, 30)
    assert abs(t0) < 3


def test_bootstrap_indices_and_ci():
    idx = next(EV.boot_indices(100, 10, 50, np.random.default_rng(0)))
    assert idx.shape == (50, 100) and idx.min() >= 0 and idx.max() < 100
    runs = (np.diff(idx, axis=1) == 1) | (np.diff(idx, axis=1) == -99)
    assert 0.8 < runs.mean() < 0.95                                   # mean block 10 -> ~90 % continuations
    r = np.random.default_rng(1).normal(0.001, 0.01, 1500)
    pt, lo, hi = EV.ci_sharpe(r, 252, 21, reps=400)
    assert lo < pt < hi and EV.sharpe_ann(r, 252) == pt
    other = np.random.default_rng(2).normal(0.0, 0.01, 1500)
    d, dlo, dhi = EV.paired_ci(r, other, 252, 21, reps=200)
    assert dlo <= d <= dhi and d > 0


def test_circular_shift_stays_inside_span():
    E = np.zeros((50, 2))
    E[10:20, 0] = 1.0
    E[30:, 1] = np.arange(20)
    out = EV.circular_shift(E, np.array([5, 30]), np.array([30, 49]), np.random.default_rng(0))
    assert out[:5, 0].sum() == 0 and out[31:, 0].sum() == 0 and out[:30, 1].sum() == 0
    assert sorted(out[5:31, 0]) == sorted(E[5:31, 0]) and sorted(out[30:, 1]) == sorted(E[30:, 1])


def test_trade_ci_by_entry_day_groups_trades():
    net = np.r_[np.full(10, 0.02), np.full(10, -0.01)]
    ed = np.r_[np.zeros(10, int), np.ones(10, int)]
    m, lo, hi = EV.trade_ci_by_entry_day(net, ed, 1.5, reps=200)
    assert m == pytest.approx(0.005) and lo < hi


def test_overlays():
    z = np.zeros(10)
    # H3a: free cash 50 % on the TOM days, SPY +1 %, cash 0
    tom = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0], bool)
    r = OV.h3a_overlay(z, np.full(10, 0.5), np.full(10, 0.01), z, tom, cost_bps=3.0)
    np.testing.assert_allclose(r[1], 0.5 * 0.01 - 3e-4 * 0.5)
    np.testing.assert_allclose(r[2], 0.5 * 0.01)
    np.testing.assert_allclose(r[3], -3e-4 * 0.5)                      # window closes: sell the SPY part
    # H3b: calm series -> k = 1; first k is 1; k <= 1 always
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.001, 200)
    out, k = OV.h3b_vol_target(calm, np.zeros(200))
    np.testing.assert_allclose(out, calm)
    assert (k <= 1).all()
    wild = rng.normal(0, 0.03, 200)
    out, k = OV.h3b_vol_target(wild, np.zeros(200))
    assert k[100] < 0.5 and out[101] == pytest.approx(k[100] * wild[101] - 3e-4 * abs(k[100] - k[99]))
    # H3c: bear market with a volatility spike triggers; a steady bull never
    T = 700
    P = np.r_[np.linspace(100, 200, 450), np.linspace(200, 120, 250)]
    r_ = np.r_[rng.normal(0, 0.005, 450), rng.normal(0, 0.03, 250)]
    P2 = 100 * np.cumprod(1 + r_)
    st = OV.crash_state(P2, np.ones(T, bool), 365)
    assert st.sum() > 0 and not OV.crash_state(P, np.ones(T, bool), 365)[:450].any()
    np.testing.assert_allclose(OV.crash_scale(np.array([True, False])), [0.5, 1.0])


def test_criteria_are_mechanical():
    base = dict(sharpe_lo=0.1, sharpe=1.0, sharpe_btc=0.8, alpha=0.001, alpha_t=2.5, sharpe_recent=0.5,
                sharpe_cost25=0.2, sharpe_cost2x=0.6, null_p95=0.7, dsr=0.97)
    assert EV.passed(EV.criteria_h1a(base))
    assert not EV.passed(EV.criteria_h1a({**base, "dsr": 0.94}))
    assert not EV.passed(EV.criteria_h1a({**base, "alpha_t": 1.99}))
    h4 = dict(sharpe=1.0, best_sleeve=0.9, diff_lo=-0.1, maxdd=0.2, maxdd_6040=0.3, sharpe_cost2x=0.9, sharpe_6040=0.8)
    c = EV.criteria_h4(h4)
    assert not EV.passed(c) and not c["2 paired diff vs 60/40 CI lower > 0"][2] and c["3 maxDD < 60/40"][2]
