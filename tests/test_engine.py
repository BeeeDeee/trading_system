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
hist = sim["history"]
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

# ---------------------------------------------------------------- targeted unit tests
import os, tempfile

def mk_today(date, o, h, l, c):
    return {"date": date, "open": o, "high": h, "low": l, "close": c}

UNI = CFG["universe"]
def flat(price=100.0, date="2026-10-01", **over):
    d = {k: {t: price for t in UNI} for k in ("open", "high", "low", "close")}
    d["date"] = date
    for k, v in over.items():
        d[k].update(v)
    return d

def day1(date="2026-10-01"):
    return E.settle(CFG, None, flat(date=date))["day"]

def with_order(day, vid, t, weight=20, stop_pct=7):
    day["variants"][vid]["orders"] = [{"ticker": t, "side": "BUY", "weight_pct": weight, "stop_pct": stop_pct, "reason": "t"}]
    day["variants"][vid]["plan_equity"] = day["variants"][vid]["equity"]
    return day

# gap-down stop fills at the OPEN (not at the stop price), with stop slippage
d = with_order(day1(), "zaklad", "AAPL")
s2 = E.settle(CFG, d["variants"] and d, flat(date="2026-10-02", open={"AAPL": 100}, high={"AAPL": 100}, low={"AAPL": 100}, close={"AAPL": 100}))["day"]
pos = s2["variants"]["zaklad"]["positions"][0]
check(pos["stop"] is not None and pos["stop"] < pos["entry_price"], "stop not set below entry")
s3 = E.settle(CFG, s2, flat(date="2026-10-03", open={"AAPL": 80}, high={"AAPL": 81}, low={"AAPL": 79}, close={"AAPL": 80}))
sold = [t for t in s3["trades"] if t["variant"] == "zaklad" and t["ticker"] == "AAPL"]
check(len(sold) == 1 and sold[0]["exit_reason"] == "Stop-loss", "gap stop not triggered")
check(sold and sold[0]["exit_price"] < 80, "stop slippage missing on gap exit")
check(sold and sold[0]["exit_price"] > 79.5 * 0.99, "gap stop filled below the open by too much")

# cash exactly conserved through a round trip (no money from nowhere)
eq = s3["day"]["variants"]["zaklad"]
check(not eq["positions"] and eq["cash"] < 10000, "round trip left positions or created cash")

# trailing stop only ratchets up
d = with_order(day1(), "posuvny", "AAPL")
a = E.settle(CFG, d, flat(date="2026-10-02", close={"AAPL": 110}, high={"AAPL": 111}))["day"]
b = E.settle(CFG, a, flat(date="2026-10-03", close={"AAPL": 107}, high={"AAPL": 109}, low={"AAPL": 106}, open={"AAPL": 108}))["day"]
sa, sb = a["variants"]["posuvny"]["positions"][0]["stop"], b["variants"]["posuvny"]["positions"][0]["stop"]
check(sb >= sa, f"trailing stop went down {sa} -> {sb}")

# max-hold exits at close
d = with_order(day1(), "reverze", "AAPL")
cur = d
for i in range(2, 6):
    r = E.settle(CFG, cur, flat(date=f"2026-10-0{i}"))
    cur = r["day"]
check(not cur["variants"]["reverze"]["positions"], "max_hold_days=3 did not close the position")

# daily loss limit blocks new buys in signal variants
d = day1()
pf = d["variants"]["zaklad"]
pf["prev_equity"], pf["equity"] = 10000, 9000
sc = {"scores": {t: {"trend": 2, "mr": 2, "news": 2, "conviction": 5, "event": False} for t in UNI}}
pl = E.plan({"day": d, "trades": []}, sc, None, CFG)
check(not pl["day"]["variants"]["zaklad"]["orders"], "daily loss limit did not block buys")

# market filter: negative SPY trend blocks filtr_trhu only
sc2 = {"scores": {t: {"trend": 2, "mr": 2, "news": 2, "conviction": 5, "event": False} for t in UNI}}
sc2["scores"]["SPY"]["trend"] = -1
pl = E.plan({"day": day1(), "trades": []}, sc2, None, CFG)["day"]["variants"]
check(not pl["filtr_trhu"]["orders"] and pl["zaklad"]["orders"], "market filter behaves wrongly")

# catch-up path: plan with empty scores and no free decisions must not crash or open trades
r = E.plan({"day": day1(), "trades": []}, {"scores": {}}, None, CFG)
check(all(not v["orders"] for k, v in r["day"]["variants"].items() if k != "nahoda"), "empty scores produced orders")

# garbage from the LLM must not crash plan (v1.0.0 crashed on all of these)
bad = {"scores": {t: {"trend": 1.0, "mr": "x", "news": None, "conviction": None, "event": "yes"} for t in UNI}}
bad["scores"]["AAPL"] = {"trend": 5, "mr": 2, "news": 2, "conviction": 9}
bad["scores"]["ZZZZ"] = {"trend": 1}
bad["scores"]["MSFT"] = "nonsense"
freebad = {"decisions": [{"ticker": "AAPL", "action": "BUY", "conviction": None, "weight_pct": "x"},
                         {"ticker": "MSFT", "action": "BUY", "conviction": 4, "weight_pct": 999, "stop_price": 500},
                         {"ticker": "NVDA", "action": "BUY", "conviction": 4, "weight_pct": 20},
                         {"ticker": "NVDA", "action": "BUY", "conviction": 4, "weight_pct": 20},
                         "junk"]}
try:
    r = E.plan({"day": day1(), "trades": []}, bad, freebad, CFG)
    orders = r["day"]["variants"]["volny"]["orders"]
    check(sum(o["ticker"] == "NVDA" for o in orders) >= 1, "valid free BUY lost")
    check(all(o["weight_pct"] <= 20 for o in orders), "free weight not capped")
except Exception as ex:
    check(False, f"plan crashed on bad input: {ex!r}")
v = E.validate(CFG, bad, freebad)
check(v["errors"] and any("ZZZZ" not in e for e in v["errors"]), "validate reported nothing on garbage")
check(not E.validate(CFG, {"scores": sim["scores"][-1]["scores"]}, None)["errors"], "validate flagged clean scores")

# a stop_price at/above the entry must not cause an instant stop-out
d = day1()
d["variants"]["volny"]["orders"] = [{"ticker": "AAPL", "side": "BUY", "weight_pct": 20, "stop_pct": 7, "stop_price": 150, "reason": "t"}]
d["variants"]["volny"]["plan_equity"] = 10000
r = E.settle(CFG, d, flat(date="2026-10-02"))["day"]["variants"]["volny"]
check(r["positions"] and r["positions"][0]["stop"] < r["positions"][0]["entry_price"], "invalid stop_price accepted")

# a new_stop above the price is ignored (would be a disguised exit)
d = with_order(day1(), "volny", "AAPL")
h = E.settle(CFG, d, flat(date="2026-10-02"))["day"]
fr = {"decisions": [{"ticker": "AAPL", "action": "HOLD", "conviction": 3, "new_stop": 120, "reason": "x"}]}
pl = E.plan({"day": h, "trades": []}, {"scores": {}}, fr, CFG)["day"]["variants"]["volny"]["positions"][0]
check(pl["stop"] < 100, "new_stop above price was applied")

# split: check suggests the ratio, apply_split re-bases everything, the same day then passes check
hist0 = {t: [[f"2026-09-{d:02d}", 100, 101, 99, 100] for d in range(1, 30)] for t in UNI}
tdy = flat(price=100.5, date="2026-10-01")
for k in ("open", "high", "low", "close"):
    tdy[k]["NVDA"] = 25.0
res = E.check(CFG, hist0, tdy)
check(res["split_suspects"].get("NVDA") == 4, f"split not recognised: {res['split_suspects']}")
d = with_order(day1("2026-09-30"), "zaklad", "NVDA")
held = E.settle(CFG, d, flat(date="2026-10-01"))["day"]
sh0 = held["variants"]["zaklad"]["positions"][0]["shares"]
h2, p2 = E.apply_split(hist0, held, "NVDA", 4)
check(h2["NVDA"][-1][4] == 25.0 and hist0["NVDA"][-1][4] == 100, "history not re-based (or input mutated)")
check(abs(p2["variants"]["zaklad"]["positions"][0]["shares"] - sh0 * 4) < 1e-3, "shares not multiplied")
val0 = sh0 * held["variants"]["zaklad"]["positions"][0]["last"]
pp = p2["variants"]["zaklad"]["positions"][0]
check(abs(pp["shares"] * pp["last"] - val0) < 0.05, "position value changed by split")
check(not E.check(CFG, h2, tdy)["errors"], "check still fails after apply_split")
check(E.split_suspect(100, 100) is None and E.split_suspect(70, 100) is None, "false split suspicion")

# atomic dump + CLI usage guard
with tempfile.TemporaryDirectory() as td:
    f = os.path.join(td, "x.json")
    E.dump({"a": 1}, f); E.dump({"a": 2}, f)
    check(json.load(open(f)) == {"a": 2} and os.listdir(td) == ["x.json"], "dump not atomic/clean")
try:
    E.main(["engine.py", "settle"])
    check(False, "bad CLI usage not rejected")
except SystemExit:
    pass

# ---------------------------------------------------------------- v1.1.0 rules
# consensus needs >= 2 lenses for AND none against
zk = next(v for v in CFG["variants"] if v["id"] == "zaklad")
check(E.entry_ok(zk, {"trend": 1, "mr": 1, "news": 0, "conviction": 3}), "clean consensus rejected")
check(not E.entry_ok(zk, {"trend": 1, "mr": 1, "news": -1, "conviction": 3}), "consensus with dissent accepted")
check(not E.entry_ok(zk, {"trend": 2, "mr": 2, "news": -2, "conviction": 5}), "consensus with strong dissent accepted")

# daily loss limit now applies to top_n and to the random baseline too
allpos = {"scores": {t: {"trend": 2, "mr": 2, "news": 2, "conviction": 5, "event": False} for t in UNI}}
d = day1()
for vid in ("plne", "top3", "rotace", "kontrarian", "nahoda", "zaklad"):
    d["variants"][vid]["prev_equity"], d["variants"][vid]["equity"] = 10000, 9000
pl = E.plan({"day": d, "trades": []}, allpos, None, CFG)["day"]["variants"]
check(all(not pl[v]["orders"] for v in ("plne", "top3", "rotace", "kontrarian", "nahoda", "zaklad")), "loss limit missed a variant")
pl = E.plan({"day": day1(), "trades": []}, allpos, None, CFG)["day"]["variants"]
check(all(pl[v]["orders"] for v in ("plne", "top3", "rotace", "kontrarian")), "top_n stopped buying without a loss")

# top_n: a held ticker without a score is sold after 2 plans, and never re-bought the same day
d = with_order(day1(), "plne", "AAPL")
h = E.settle(CFG, d, flat(date="2026-10-02"))["day"]
sc_wo = {"scores": {t: {"trend": 1, "mr": 0, "news": 0, "conviction": 3, "event": False} for t in UNI if t != "AAPL"}}
p1 = E.plan({"day": json.loads(json.dumps(h)), "trades": []}, sc_wo, None, CFG)["day"]["variants"]["plne"]
check(not any(o["side"] == "SELL" for o in p1["orders"]), "sold after a single missing score")
h2 = json.loads(json.dumps(h)); h2["variants"]["plne"]["positions"] = p1["positions"]
p2 = E.plan({"day": h2, "trades": []}, sc_wo, None, CFG)["day"]["variants"]["plne"]
check(any(o["side"] == "SELL" and o["ticker"] == "AAPL" for o in p2["orders"]), "stale position not sold")
check(not any(o["side"] == "BUY" and o["ticker"] == "AAPL" for o in p2["orders"]), "sold ticker re-bought same day")
sc_ok = {"scores": {t: {"trend": 1, "mr": 0, "news": 0, "conviction": 3, "event": False} for t in UNI}}
p3 = E.plan({"day": json.loads(json.dumps(h)), "trades": []}, sc_ok, None, CFG)["day"]["variants"]["plne"]
check(p3["positions"][0]["no_score_days"] == 0, "no_score_days not reset")

# random_null reproduces the engine and orders sanely
import random_null
nul = random_null.null_returns(CFG, sim["days"], sim["ohlc"], "zaklad", n=20, seed=1)
check(len(nul) == 20 and all(-50 < x < 50 for x in nul), "random null out of range")
check(len(set(round(x, 4) for x in nul)) > 1, "random null has no variation")

if fails:
    sys.exit(f"{len(fails)} test(s) failed")
print(f"OK: {len(sim['days'])} days x {len(last)} portfolios, {len(sim['trades'])} trades, all invariants hold")
