"""Orchestration of the research 17 test (prereg §3-9): followers, scenarios, criteria, null, robustness.

`run(loader, syms, members)` takes `loader(sym, market)` -> panel dict, so the real script feeds vault-guarded data
and the tests feed synthetic markets through exactly the same code.
"""

from dataclasses import asdict

import numpy as np

from qlab.research15.evaluate import by_year, stats, window
from qlab.research15.setup import epoch_15m
from qlab.research16 import engine as G
from qlab.research17 import engine as R
from qlab.research17.setup import COST, N_BOOT, N_RND, START, SUBPERIODS, day0, n_days
from qlab.validation.stats import deflated_sharpe

N_CUM = 47                                  # 16 (research 15) + 30 (research 16) + 1
SR_VAR = 0.0043749486723779615              # variance of daily Sharpe across the research 16 grid (proxy)
MIN_EVENTS = 30
P = R.Scenario("base")
SCEN = [P, R.Scenario("x2", cost=2 * COST), R.Scenario("delay1", delay=1), R.Scenario("cost75", cost=7.5e-4),
        R.Scenario("nofund", use_funding=False), R.Scenario("K2", K=2.0), R.Scenario("LS", side="LS"),
        R.Scenario("nolag", lag=False), R.Scenario("btc_perp_leader", leader="perp")]


def run(loader, syms: list[str], members: dict, n_rnd: int = N_RND, n_boot: int = N_BOOT, verbose: bool = False) -> dict:
    btc = loader("BTCUSDT", "spot")
    leaders = {"spot": R.leader(btc), "perp": R.leader(loader("BTCUSDT", "perp"))}
    res, lights, hold_sum, hold_cnt = {}, {}, np.zeros(n_days()), np.zeros(n_days())
    for i, sym in enumerate(syms):
        a = G.Asset.build(loader(sym, "perp"))
        s0 = max(int(epoch_15m(START) - a.t[0]), 0)
        res[sym] = R.run_follower(a, leaders, SCEN, s0, R.member_mask(a, sym, members))
        lights[sym] = R.light(a)
        md = R.member_days(sym, members)
        hold_sum += np.where(md, R.hold_daily(a, s0), 0.0)
        hold_cnt += md
        if verbose:
            print(f"{i + 1}/{len(syms)} {sym} trades {len(res[sym]['base'].rets)}", flush=True)

    eth = G.Asset.build(loader("ETHUSDT", "perp"))
    eth_ref = R.run_follower(eth, leaders, [P], max(int(epoch_15m(START) - eth.t[0]), 0), None, weight=1.0)["base"]


    def portfolio(name):
        return sum(r[name].daily for r in res.values())


    def trades(name):
        return np.concatenate([r[name].rets for r in res.values()])


    dates = (day0() + np.arange(n_days())).astype("datetime64[D]")
    base = {s: r["base"] for s, r in res.items()}
    daily = portfolio("base")
    sp = stats(daily, trades("base"))
    mean, lo, hi, n_events = R.event_ci([r.entries for r in base.values() if len(r.entries)],
                                        [r.rets for r in base.values() if len(r.entries)], n_boot=n_boot)
    ev = R.events_of(base)
    times = R.candidate_times(btc["t"], btc["real"], int(epoch_15m(START)), int(btc["t"][-1]) - R.HOLD - 1)
    rnd = np.array([stats(R.null_run(lights, ev, times, np.random.default_rng(sd))[0])["sharpe"] for sd in range(n_rnd)])
    p_null = float((1 + (rnd >= sp["sharpe"]).sum()) / (1 + n_rnd))
    s_x2, s_d1 = stats(portfolio("x2"))["sharpe"], stats(portfolio("delay1"))["sharpe"]
    sub = {k: stats(window(dates, daily, *w))["sharpe"] for k, w in SUBPERIODS.items()}
    contrib = {s: float(r.daily.sum()) for s, r in base.items()}
    top = max(contrib, key=contrib.get)
    s_wo_top = stats(daily - base[top].daily)["sharpe"]
    per_sym = {s: {"n": int(len(r.rets)), "mean": float(r.rets.mean()) if len(r.rets) else None,
                   "contribution": contrib[s]} for s, r in base.items()}
    big = [v for v in per_sym.values() if v["n"] >= 10]
    share_pos = float(np.mean([v["mean"] > 0 for v in big])) if big else float("nan")

    criteria = {
        "1_sharpe_and_event_ci": {"sharpe": sp["sharpe"], "mean_trade_ci90": [mean, lo, hi], "events": n_events,
                                  "pass": bool(sp["sharpe"] > 0 and mean > 0 and lo > 0)},
        "2_null_p": {"p": p_null, "B_RND_EVT": {"median": float(np.median(rnd)), "p95": float(np.quantile(rnd, 0.95))},
                     "pass": bool(p_null < 0.05)},
        "3_cost_x2": {"sharpe": s_x2, "pass": bool(s_x2 > 0)},
        "4_delay_1_candle": {"sharpe": s_d1, "pass": bool(s_d1 > 0)},
        "5_subperiods": {"sharpe": sub, "pass": bool(all(v > 0 for v in sub.values()))},
        "6_breadth": {"sharpe_without_top": s_wo_top, "top": top, "share_positive_symbols": share_pos,
                      "symbols_with_10_trades": len(big), "pass": bool(s_wo_top > 0 and share_pos >= 0.5)},
    }
    n_pass = int(sum(c["pass"] for c in criteria.values()))
    verdict = "inconclusive" if n_events < MIN_EVENTS else ("pass" if n_pass == 6 else "rejected")

    n_ev = np.array([len(v) for v in ev.values()])
    hold = np.where(hold_cnt > 0, hold_sum / np.maximum(hold_cnt, 1), 0.0)
    out = {
        "verdict": verdict, "n_pass": n_pass,
        "criteria": criteria, "P_stats": sp, "n_symbols": len(syms), "n_events": n_events,
        "trades_per_event": {"mean": float(n_ev.mean()), "max": int(n_ev.max()), "dist": np.bincount(n_ev).tolist()},
        "dsr_N1": deflated_sharpe(daily, 1, SR_VAR), "dsr_N47": deflated_sharpe(daily, N_CUM, SR_VAR),
        "B_HOLD_equal_weight_members": stats(hold),
        "robust": {
            "scenarios": {sc.name: stats(portfolio(sc.name), trades(sc.name)) | {"spec": asdict(sc)} for sc in SCEN},
            "by_year": by_year(dates, daily),
            "eth_reference_weight1": stats(eth_ref.daily, eth_ref.rets),
            "per_symbol": per_sym,
            "exposure_share_of_days": float((daily != 0).mean()),
        },
    }
    return out
