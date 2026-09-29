#!/usr/bin/env python3
"""Invariant tests for the engine (no pytest needed):  python tests/test_engine.py"""
import json, pathlib, sys, copy

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import simulate  # noqa: E402

CFG = json.load(open(ROOT / "config" / "config.json"))
E = simulate.load_engine()
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)
        print("FAIL", msg)


sim = simulate.run(CFG, days=14, seed=3)
vmax = {v["id"]: v for v in CFG["variants"]}
vmax["nahoda"] = CFG["random_baseline"]

prev_costs = {}
for n, d in enumerate(sim["days"], 1):
    check(d["run_no"] == n, f"run_no {d['run_no']} != {n}")
    for vid, v in d["variants"].items():
        check(v["cash"] >= -0.05, f"{d['date']} {vid}: negative cash {v['cash']}")
        mtm = v["cash"] + sum(p["shares"] * p["last"] for p in v["positions"])
        check(abs(mtm - v["equity"]) < 0.05, f"{d['date']} {vid}: equity {v['equity']} != cash+positions {mtm:.2f}")
        check(len(v["positions"]) <= vmax[vid]["max_positions"], f"{d['date']} {vid}: too many positions")
        tk = vmax[vid].get("tickers")
        if tk:
            check(all(p["ticker"] in tk for p in v["positions"]), f"{d['date']} {vid}: ticker outside subset")
        check(v["costs_total"] >= prev_costs.get(vid, 0) - 1e-6, f"{d['date']} {vid}: costs_total decreased")
        prev_costs[vid] = v["costs_total"]
        check(all(p["stop"] is None or p["stop"] > 0 for p in v["positions"]), f"{d['date']} {vid}: bad stop")
        if vmax[vid].get("stop_pct") is None and not vmax[vid].get("trailing_stop_pct") and vmax[vid]["mode"] != "free":
            check(all(p["stop"] is None for p in v["positions"]), f"{d['date']} {vid}: stop set on a no-stop variant")

# fees and slippage are really charged: first day with fills has costs > 0
first_fill = next(d for d in sim["days"] if d["variants"]["zaklad"]["fills"])
check(first_fill["variants"]["zaklad"]["costs_today"] > 0, "no costs charged on fills")

# every closed trade nets fees: pnl_usd < gross move for a winning trade
for t in sim["trades"]:
    gross = (t["exit_price"] - t["entry_price"]) * t["shares"]
    check(t["pnl_usd"] <= gross + 0.01, f"{t['ticker']}: pnl above gross (fees missing)")

# contrarian holds the opposite end of the ranking from 'plne'
last = sim["days"][-1]["variants"]
check({p["ticker"] for p in last["kontrarian"]["positions"]} != {p["ticker"] for p in last["plne"]["positions"]}, "kontrarian == plne")

# determinism
again = simulate.run(CFG, days=14, seed=3)
check(json.dumps(again["days"], sort_keys=True) == json.dumps(sim["days"], sort_keys=True), "simulation not deterministic")

# data checks catch bad prices
hist, bad = sim["history"], copy.deepcopy(simulate.run(CFG, days=1, seed=5)["history"])
today = {"date": "2099-01-01", "open": {}, "high": {}, "low": {}, "close": {}}
for t in CFG["universe"]:
    c = hist[t][-1][4]
    today["open"][t], today["high"][t], today["low"][t], today["close"][t] = c, c * 1.01, c * 0.99, c
res = E.check(CFG, hist, today)
check(not res["errors"], f"clean data flagged: {res['errors']}")
today["close"]["AAPL"] *= 1.5
today["high"]["AAPL"] = today["close"]["AAPL"] * 1.01
for t in ["XLE", "XLV", "XLP", "XLU", "TLT", "GLD"]:
    today["close"].pop(t)
res = E.check(CFG, hist, today)
check(any("AAPL" in e for e in res["errors"]), "bad AAPL jump not caught")
check(any("Chybí ceny" in e for e in res["errors"]), "missing tickers not caught")

# config sanity
ids = [v["id"] for v in CFG["variants"]]
check(len(ids) == len(set(ids)), "duplicate variant ids")
check(CFG["random_baseline"]["mirror"] in ids, "random baseline mirrors an unknown variant")
check(all(t in CFG["universe"] for v in CFG["variants"] for t in v.get("tickers", [])), "variant ticker not in universe")

if fails:
    sys.exit(f"{len(fails)} test(s) failed")
print(f"OK: {len(sim['days'])} days x {len(last)} portfolios, {len(sim['trades'])} trades, all invariants hold")
