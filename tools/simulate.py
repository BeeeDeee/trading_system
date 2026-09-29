#!/usr/bin/env python3
"""Synthetic market simulation driven through the real engine.

Used by tests (invariants) and to build the demo data shown in the dashboard
before the first real evening run.  Nothing here touches the network.

  python tools/simulate.py --days 16 --seed 11 --out dashboard/sample.json
"""
import argparse, datetime, importlib.util, json, pathlib, random

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_engine():
    spec = importlib.util.spec_from_file_location("engine", ROOT / "engine" / "engine.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REGIMES = [
    ("Risk-on, trend", "Index nad 20denním průměrem, VIX pod 16, vedou technologie."),
    ("Neutrální, konsolidace", "Trh v úzkém pásmu před CPI, klesající objemy."),
    ("Risk-off, zvýšená volatilita", "VIX nad 20, rostou výnosy, defenziva překonává cyklické sektory."),
]


def trading_days(start, n):
    d, out = start, []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return out


def run(cfg, days=16, seed=11, start=datetime.date(2026, 9, 7), warmup=60):
    """Return dict(days, trades, scores, prices, runs, history) for a synthetic market."""
    E = load_engine()
    rng = random.Random(seed)
    U = cfg["universe"]
    px = {t: rng.uniform(60, 500) for t in U}

    # warm-up history so features() has enough rows
    hist = {}
    wd = trading_days(start - datetime.timedelta(days=warmup * 2), warmup)
    for d in wd:
        for t in U:
            o = px[t]
            c = o * (1 + rng.gauss(0.0004, 0.012))
            hist.setdefault(t, []).append([d, round(o, 2), round(max(o, c) * 1.004, 2), round(min(o, c) * 0.996, 2), round(c, 2)])
            px[t] = c

    dates = trading_days(start, days)
    prev, out_days, trades, scores_l, prices_l, runs_l, ohlc_l = None, [], [], [], [], [], []
    drift = {t: 0.0 for t in U}
    for i, dt in enumerate(dates):
        mkt = rng.gauss(0.0005, 0.008)
        o, h, l, c = {}, {}, {}, {}
        for t in U:
            op = px[t] * (1 + rng.gauss(0, 0.003))
            cl = op * (1 + mkt + drift[t] + rng.gauss(0, 0.013))
            o[t], c[t] = round(op, 2), round(cl, 2)
            h[t] = round(max(op, cl) * (1 + abs(rng.gauss(0, 0.004))), 2)
            l[t] = round(min(op, cl) * (1 - abs(rng.gauss(0, 0.006))), 2)
            px[t] = cl
        today = {"date": dt, "open": o, "high": h, "low": l, "close": c}
        chk = E.check(cfg, hist, today)
        hist = E.history_append(hist, today)
        settled = E.settle(cfg, prev, today)

        drift = {t: rng.gauss(0, 0.004) for t in U}
        sc = {}
        for t in U:
            z = drift[t] / 0.004
            clip = lambda v, a, b: max(a, min(b, v))
            tr = clip(round(z * 0.8 + rng.gauss(0, 1)), -2, 2)
            mr = clip(round(rng.gauss(0, 1.1)), -2, 2)
            nw = clip(round(z * 0.4 + rng.gauss(0, 0.8)), -2, 2)
            sc[t] = {"trend": tr, "mr": mr, "news": nw,
                     "conviction": clip(3 + round((tr + nw) / 2 + rng.gauss(0, 0.7)), 1, 5),
                     "event": rng.random() < 0.06, "note": "Ukázkové zdůvodnění."}
        reg = REGIMES[0 if i < 6 else 1 if i < 11 else 2]
        scores_doc = {"date": dt, "regime": {"label": reg[0], "summary": reg[1]}, "scores": sc}

        vf = settled["day"]["variants"]["volny"]
        held = {p["ticker"] for p in vf["positions"]}
        best = sorted((t for t in U if t not in held), key=lambda t: -(sc[t]["trend"] + sc[t]["news"] + sc[t]["conviction"]))[:1]
        dec = [{"ticker": p["ticker"], "action": "HOLD", "conviction": 3, "reason": "Trend trvá, držet.", "new_stop": p["last"] * 0.95} for p in vf["positions"]]
        dec += [{"ticker": t, "action": "BUY", "conviction": 4, "weight_pct": 20,
                 "reason": "Relativní síla vůči SPY, průraz z konsolidace.", "invalidation": "Zavření pod 20denním průměrem."} for t in best]
        dec += [{"ticker": "TSLA", "action": "SKIP", "conviction": 2, "reason": "Blíží se binární událost."}]

        planned = E.plan(settled, scores_doc, {"decisions": dec}, cfg)
        day = planned["day"]
        day["regime"], day["notes"] = scores_doc["regime"], "—"
        prev = day
        out_days.append(day)
        trades += planned["trades"]
        scores_l.append(scores_doc)
        prices_l.append({"date": dt, "close": c})
        ohlc_l.append(today)
        warn = ["NVDA: velký pohyb +13,2 %, ověř"] if i == 12 else []
        runs_l.append({"date": dt, "status": "warning" if warn else "ok", "finished_at": dt + "T20:31:00Z",
                       "source": "simulace", "checks": {**chk, "warnings": chk["warnings"] + warn},
                       "spy_crosscheck": {"diff_pct": 0.02},
                       "message": "Simulovaný běh."})
    return {"days": out_days, "trades": trades, "scores": scores_l, "prices": prices_l, "runs": runs_l, "history": hist, "ohlc": ohlc_l}


def dashboard_sample(sim, cfg):
    """Trim a simulation to what the dashboard needs (keeps the HTML small)."""
    days = json.loads(json.dumps(sim["days"]))
    for d in days[:-1]:
        for k, v in d["variants"].items():
            v["fills"], v["orders"] = [], []
            if k != "volny":
                v["decisions"] = []
            v["positions"] = [{kk: p[kk] for kk in ("ticker", "shares", "entry_price", "last")} for p in v["positions"]]
    trades = [{k: v for k, v in t.items() if k not in ("thesis", "shares")} for t in sim["trades"]]
    scores = json.loads(json.dumps(sim["scores"]))
    for s in scores:
        for x in s["scores"].values():
            x.pop("note", None)
    return {"runs": sim["runs"], "days": days, "trades": trades, "scores": scores, "prices": sim["prices"], "config": cfg}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=16)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--config", default=str(ROOT / "config" / "config.json"))
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "sample.json"))
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    sim = run(cfg, a.days, a.seed)
    json.dump(dashboard_sample(sim, cfg), open(a.out, "w"), ensure_ascii=False, separators=(",", ":"))
    last = sim["days"][-1]
    print(f"{a.out}: {a.days} days, {len(sim['trades'])} trades")
    for k, v in last["variants"].items():
        print(f"  {k:14} {v['equity']:>10}")
