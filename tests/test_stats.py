"""Statistiky ověřené ručním výpočtem na malém vzorku (SPEC §17 P4)."""

import math

import numpy as np
import pytest

from arena import stats
from arena.stats import T

D = 86_400_000
D0 = 20_000 * D


def t(variant, sym, day, net_bps, *, gross=None, fee=10.0, es=1.0, xs=2.0, fund=0.0, hold_min=60, reason="trailing",
      tainted=0):
    notional = 1000.0
    exit_t = D0 + day * D + 3_600_000
    net_pnl = net_bps * notional / 1e4
    return T(f"{variant}.{sym}", sym, variant, variant.split(".")[0], "LONG", exit_t - hold_min * 60_000, exit_t,
             reason, notional, net_pnl, gross if gross is not None else net_bps + fee - fund, net_bps, fee, es, xs, fund,
             50.0, -20.0, tainted)


SAMPLE = [t("brk.v00", "BTCUSDT", 0, 30.0), t("brk.v00", "ETHUSDT", 0, -10.0),
          t("brk.v00", "BTCUSDT", 1, 20.0, fund=-1.0), t("brk.v00", "SOLUSDT", 2, -20.0)]


def test_group_stats_by_hand():
    g = stats.group_stats("brk.v00", SAMPLE, n_symbols=5, period_ms=3 * D)
    assert g.n == 4 and g.days == 3 and g.trades_per_day == pytest.approx(4 / 3)
    assert g.net_mean == pytest.approx(5.0) and g.net_median == pytest.approx(5.0)
    assert g.net_sd == pytest.approx(math.sqrt((625 + 225 + 225 + 625) / 3))
    assert g.win_rate == pytest.approx(50.0)
    # náklady: fee 10, entry 1, exit 2, funding: jeden obchod −1 bps → průměrný náklad +0.25
    assert g.cost_bps["fee"] == 10 and g.cost_bps["entry_slip"] == 1 and g.cost_bps["exit_slip"] == 2
    assert g.cost_bps["funding"] == pytest.approx(0.25) and g.cost_bps["total"] == pytest.approx(13.25)
    assert g.edge_mean == pytest.approx(5 + 13.25) and g.break_even_edge == pytest.approx(13.25)
    assert g.net_at_1_5x == pytest.approx(18.25 - 1.5 * 13.25) and g.net_at_2x == pytest.approx(18.25 - 26.5)
    # denní P&L (USDT): den 0: 3 − 1 = 2, den 1: 2, den 2: −2
    dp = stats.daily_pnl(SAMPLE)
    np.testing.assert_allclose(dp, [2.0, 2.0, -2.0])
    sd = np.std([2, 2, -2], ddof=1)
    assert g.sharpe == pytest.approx((2 / 3) / sd * math.sqrt(365))
    assert g.max_dd == pytest.approx(2.0)
    assert g.calmar == pytest.approx(365 * (2 / 3) / 2.0)
    # expozice: 4 × 60 min / (3 dny × 5 symbolů)
    assert g.exposure_pct == pytest.approx(4 * 3_600_000 / (3 * D * 5) * 100)
    assert g.avg_hold_min == pytest.approx(60)
    assert g.by_symbol["BTCUSDT"]["net_mean"] == pytest.approx(25.0)
    assert g.exit_mix == {"trailing": 4}


def test_n_req_matches_spec_example():
    # SPEC §13: sd 60, hrana 5 bps → ~580 obchodů na t=2, ~1300 na t=3
    trades = [t("x.v00", "BTCUSDT", i // 10, 5 + 60 * (1 if i % 2 else -1)) for i in range(100)]
    g = stats.group_stats("x.v00", trades)
    assert g.n_req["z2"] == pytest.approx((2 * g.net_sd / g.net_mean) ** 2)
    assert 570 < (2 * 60 / 5) ** 2 < 590 and 1290 < (3 * 60 / 5) ** 2 < 1300


def test_block_bootstrap_degenerate_and_deterministic():
    v = np.array([7.0, 7.0, 7.0, 7.0])
    idx = np.array([0, 0, 1, 2])
    b = stats.block_bootstrap_mean(v, idx, 3)
    assert np.allclose(b, 7.0)
    v2 = np.array([1.0, 5.0, 9.0, -3.0])
    assert np.array_equal(stats.block_bootstrap_mean(v2, idx, 3), stats.block_bootstrap_mean(v2, idx, 3))


def test_holm_by_hand():
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": math.nan})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert math.isnan(adj["d"])


def test_dsr_reduces_to_psr_when_no_trial_variance():
    # sr_var = 0 → SR* = 0 → DSR = Φ(SR √(T−1) / √(1 − γ3 SR + (γ4−1)/4 SR²))
    sr, n = 0.1, 101
    got = stats.deflated_sharpe(sr, n, 0.0, 3.0, 0.0, 30)
    assert got == pytest.approx(stats._N.cdf(0.1 * 10 / math.sqrt(1 + 0.5 * 0.01)))
    # s rozptylem mezi variantami je DSR nižší
    assert stats.deflated_sharpe(sr, n, 0.0, 3.0, 0.01, 30) < got


def test_vs_control_and_badges():
    var = [t("obi.v00", "BTCUSDT", d, 20.0 + (d % 3)) for d in range(40)]
    ctl = [t("ctl.obi", "BTCUSDT", d, -5.0 + (d % 3)) for d in range(40)]
    r = stats.vs_control(var, ctl)
    assert r["delta"] == pytest.approx(25.0) and r["p"] < 0.01 and r["ci_lo"] > 20
    lb = stats.leaderboard(var + ctl, {"obi.v00": ["BTCUSDT"], "ctl.obi": ["BTCUSDT"]})
    assert lb["obi.v00"].badge == "PŘÍLIŠ BRZY" and lb["ctl.obi"].badge == "KONTROLA"
    assert lb["obi.v00"].p_holm == pytest.approx(lb["obi.v00"].vs_control["p"])


def test_badge_rules():
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
    good = [t("brk.v00", syms[i % 5], i // 10, 30.0 + (i % 7), fee=2, es=0, xs=0) for i in range(600)]
    g = stats.group_stats("brk.v00", good, 5)
    g.p_holm = 0.01
    assert stats.badge(g, good, False, calibrated=True, live=False) == "KANDIDÁT"
    assert stats.badge(g, good, False, calibrated=False, live=False) == "NEJASNÁ"   # model nekalibrovaný
    assert stats.badge(g, good, False, calibrated=True, live=True) == "LIVE"
    bad = [t("brk.v01", "BTCUSDT", i // 10, -30.0 + (i % 5)) for i in range(300)]
    gb = stats.group_stats("brk.v01", bad)
    assert stats.badge(gb, bad, False, True, False) == "ZÁPORNÁ"
    tainted = good[:-1] + [t("brk.v00", "BTCUSDT", 70, 30.0, tainted=1)]
    assert stats.badge(g, tainted, False, True, False) == "NEJASNÁ"           # tainted obchod
