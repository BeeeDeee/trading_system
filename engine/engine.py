#!/usr/bin/env python3
"""Paper-trading engine for the evening bot (multi-variant).

Subcommands (all read/write JSON files, stdlib only):
  history-append HISTORY TODAY_PRICES OUT     merge today's OHLC into rolling history (80 rows)
  features HISTORY OUT                        indicators for every ticker (+ prints a table)
  settle CONFIG PREV_DAY|- TODAY_PRICES OUT   execute yesterday's orders at today's open, stops, max-hold, mark to market
  plan CONFIG SETTLED SCORES FREE|- OUT       orders for tomorrow's open for every variant + random baseline
  check CONFIG HISTORY TODAY_PRICES OUT       sanity-check today's prices before anything else ("errors" must be empty)

TODAY_PRICES = {"date": "YYYY-MM-DD", "open": {T: x}, "high": {...}, "low": {...}, "close": {...}}
SCORES       = {"date": ..., "scores": {T: {"trend": -2..2, "mr": -2..2, "news": -2..2,
                "conviction": 1..5, "event": bool, "note": str}}}
FREE         = {"decisions": [{"ticker", "action": BUY|SELL|HOLD|SKIP, "conviction", "weight_pct",
                "stop_price"?, "new_stop"?, "reason", "invalidation"}]}
OUT of plan  = {"day": <days/<date> document>, "trades": [<trades documents>]}
"""
import json, math, random, sys

LENSES = ("trend", "mr", "news")
HIST_LEN = 80


def load(p):
    if p in (None, "-"):
        return None
    with open(p) as f:
        return json.load(f)


def dump(obj, p):
    with open(p, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def r2(x):
    return None if x is None else round(float(x), 2)


def r4(x):
    return None if x is None else round(float(x), 4)


# ---------------------------------------------------------------- history & features
def history_append(hist, today):
    hist = hist or {}
    d = today["date"]
    for t, c in today["close"].items():
        if c is None:
            continue
        row = [d, today["open"].get(t), today["high"].get(t), today["low"].get(t), c]
        rows = [r for r in hist.get(t, []) if r[0] != d]
        rows.append(row)
        rows.sort(key=lambda r: r[0])
        hist[t] = rows[-HIST_LEN:]
    return hist


def _rsi(closes, n=14):
    if len(closes) <= n:
        return None
    gains, losses = [], []
    for a, b in zip(closes[:-1], closes[1:]):
        ch = b - a
        gains.append(max(ch, 0))
        losses.append(max(-ch, 0))
    ag = sum(gains[:n]) / n
    al = sum(losses[:n]) / n
    for g, l in zip(gains[n:], losses[n:]):
        ag = (ag * (n - 1) + g) / n
        al = (al * (n - 1) + l) / n
    if al == 0:
        return 100.0
    return 100 - 100 / (1 + ag / al)


def _ret(c, k):
    return (c[-1] / c[-1 - k] - 1) * 100 if len(c) > k else None


def features(hist):
    out = {}
    spy = [r[4] for r in hist.get("SPY", [])]
    spy20 = _ret(spy, 20)
    for t, rows in hist.items():
        c = [r[4] for r in rows]
        h = [r[2] if r[2] is not None else r[4] for r in rows]
        l = [r[3] if r[3] is not None else r[4] for r in rows]
        if not c:
            continue
        f = {"close": c[-1], "rows": len(c)}
        f["ret1"], f["ret5"], f["ret20"] = _ret(c, 1), _ret(c, 5), _ret(c, 20)
        for n in (20, 50):
            f[f"dist_sma{n}"] = (c[-1] / (sum(c[-n:]) / n) - 1) * 100 if len(c) >= n else None
        f["rsi14"] = _rsi(c)
        if len(c) >= 15:
            trs = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(len(c) - 14, len(c))]
            f["atr14_pct"] = sum(trs) / 14 / c[-1] * 100
        else:
            f["atr14_pct"] = None
        if len(c) >= 20:
            f["dist_20d_high"] = (c[-1] / max(h[-20:]) - 1) * 100
            f["dist_20d_low"] = (c[-1] / min(l[-20:]) - 1) * 100
        else:
            f["dist_20d_high"] = f["dist_20d_low"] = None
        f["rs20_vs_spy"] = f["ret20"] - spy20 if f["ret20"] is not None and spy20 is not None else None
        out[t] = {k: (r2(v) if isinstance(v, float) else v) for k, v in f.items()}
    return out


def print_features(feat):
    cols = ["close", "ret1", "ret5", "ret20", "dist_sma20", "dist_sma50", "rsi14", "atr14_pct", "dist_20d_high", "dist_20d_low", "rs20_vs_spy"]
    print("ticker " + " ".join(f"{c:>12}" for c in cols))
    for t in sorted(feat):
        print(f"{t:<6} " + " ".join(f"{('' if feat[t].get(c) is None else feat[t][c]):>12}" for c in cols))


# ---------------------------------------------------------------- execution helpers
class Costs:
    def __init__(self, cfg):
        c = cfg.get("costs", {})
        self.fee = c.get("fees_bps", 5) / 1e4
        self.slip = c.get("slippage_bps", 5) / 1e4
        self.stop_slip = c.get("stop_slippage_bps", 20) / 1e4
        self.fractional = cfg.get("fractional_shares", True)


def buy(pf, ticker, raw, amount, costs, date, meta, fills, log):
    price = raw * (1 + costs.slip)
    per_share_cash = price * (1 + costs.fee)
    max_amount = pf["cash"] / (1 + costs.fee)
    if amount > max_amount:
        if max_amount < 0.25 * amount:
            log.append(f"{ticker}: nákup přeskočen, málo hotovosti")
            return None
        amount = max_amount
    shares = math.floor(amount / price * 1e4) / 1e4 if costs.fractional else math.floor(amount / price)
    if shares <= 0:
        return None
    notional = shares * price
    fee = notional * costs.fee
    slip_cost = shares * (price - raw)
    pf["cash"] -= notional + fee
    pf["costs_today"] += fee + slip_cost
    stop = meta.get("stop_price")
    if stop is None and meta.get("stop_pct"):
        stop = price * (1 - meta["stop_pct"] / 100)
    pos = {"ticker": ticker, "shares": shares, "entry_price": r4(price), "entry_date": date,
           "entry_fee": r4(fee), "stop": r4(stop), "last": r4(raw), "days_held": 0,
           "thesis": meta.get("reason", ""), "conviction": meta.get("conviction")}
    pf["positions"].append(pos)
    fills.append({"ticker": ticker, "side": "BUY", "price": r4(price), "shares": shares, "fee": r2(fee)})
    return pos


def sell(pf, pos, raw, costs, date, reason, fills, trades, vid, stop_exit=False):
    slip = costs.slip + (costs.stop_slip if stop_exit else 0)
    price = raw * (1 - slip)
    notional = pos["shares"] * price
    fee = notional * costs.fee
    pf["cash"] += notional - fee
    pf["costs_today"] += fee + pos["shares"] * (raw - price)
    pf["positions"] = [p for p in pf["positions"] if p is not pos]
    fills.append({"ticker": pos["ticker"], "side": "SELL", "price": r4(price), "shares": pos["shares"], "fee": r2(fee), "reason": reason})
    cost_basis = pos["entry_price"] * pos["shares"]
    pnl = notional - fee - cost_basis - (pos.get("entry_fee") or 0)
    trades.append({"variant": vid, "ticker": pos["ticker"], "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
                   "exit_date": date, "exit_price": r4(price), "shares": pos["shares"], "pnl_usd": r2(pnl),
                   "pnl_pct": r2(pnl / cost_basis * 100 if cost_basis else 0), "exit_reason": reason,
                   "thesis": pos.get("thesis", ""), "days_held": pos.get("days_held", 0)})


def settle_variant(vid, vcfg, prev, today, costs, start_capital, trades):
    date = today["date"]
    O, H, L, C = today["open"], today["high"], today["low"], today["close"]
    pf = {"cash": prev["cash"] if prev else start_capital,
          "positions": [dict(p) for p in (prev["positions"] if prev else [])],
          "costs_today": 0.0}
    fills, log = [], []
    orders = prev.get("orders", []) if prev else []
    plan_equity = prev.get("plan_equity", prev.get("equity")) if prev else start_capital

    for o in [o for o in orders if o["side"] == "SELL"]:
        pos = next((p for p in pf["positions"] if p["ticker"] == o["ticker"]), None)
        if pos and O.get(o["ticker"]) is not None:
            sell(pf, pos, O[o["ticker"]], costs, date, o.get("exit_reason", "Signál k prodeji"), fills, trades, vid)
    for o in [o for o in orders if o["side"] == "BUY"]:
        t = o["ticker"]
        if O.get(t) is None or any(p["ticker"] == t for p in pf["positions"]):
            continue
        if len(pf["positions"]) >= vcfg.get("max_positions", 5):
            log.append(f"{t}: nákup přeskočen, plný počet pozic")
            continue
        buy(pf, t, O[t], o["weight_pct"] / 100 * plan_equity, costs, date, o, fills, log)

    for pos in list(pf["positions"]):
        t = pos["ticker"]
        if pos.get("stop") and L.get(t) is not None and L[t] <= pos["stop"]:
            raw = min(pos["stop"], O.get(t) or pos["stop"])
            sell(pf, pos, raw, costs, date, "Stop-loss", fills, trades, vid, stop_exit=True)

    for pos in list(pf["positions"]):
        pos["days_held"] = pos.get("days_held", 0) + 1
        mh = vcfg.get("max_hold_days")
        if mh and pos["days_held"] >= mh and C.get(pos["ticker"]) is not None:
            sell(pf, pos, C[pos["ticker"]], costs, date, "Max. doba držení", fills, trades, vid)

    trail = vcfg.get("trailing_stop_pct")
    for pos in pf["positions"]:
        if C.get(pos["ticker"]) is not None:
            pos["last"] = r4(C[pos["ticker"]])
        if trail:
            cand = pos["last"] * (1 - trail / 100)
            if not pos.get("stop") or cand > pos["stop"]:
                pos["stop"] = r4(cand)
    equity = pf["cash"] + sum(p["shares"] * p["last"] for p in pf["positions"])
    prev_eq = prev["equity"] if prev else start_capital
    return {"cash": r2(pf["cash"]), "equity": r2(equity), "prev_equity": r2(prev_eq),
            "positions": pf["positions"], "fills": fills, "log": log,
            "costs_today": r2(pf["costs_today"]),
            "costs_total": r2((prev.get("costs_total", 0) if prev else 0) + pf["costs_today"]),
            "exposure_pct": r2((equity - pf["cash"]) / equity * 100 if equity else 0),
            "orders": [], "decisions": []}


def settle(cfg, prev, today):
    costs = Costs(cfg)
    start = cfg["start_capital"]
    trades = []
    ids = [v["id"] for v in cfg["variants"]] + ["nahoda"]
    vmap = {v["id"]: v for v in cfg["variants"]}
    vmap["nahoda"] = cfg.get("random_baseline", {"max_positions": 5, "weight_pct": 20, "stop_pct": 7, "max_hold_days": 10})
    variants = {}
    for vid in ids:
        pv = (prev or {}).get("variants", {}).get(vid)
        variants[vid] = settle_variant(vid, vmap[vid], pv, today, costs, start, trades)
    C = today["close"]
    bases = (prev or {}).get("bases") or {"spy0": C.get("SPY"), "univ0": {t: c for t, c in C.items() if c}}
    spy_eq = start * C["SPY"] / bases["spy0"] if C.get("SPY") and bases.get("spy0") else None
    ratios = [C[t] / c0 for t, c0 in bases["univ0"].items() if C.get(t) and c0]
    univ_eq = start * sum(ratios) / len(ratios) if ratios else None
    day = {"date": today["date"], "run_no": (prev or {}).get("run_no", 0) + 1, "bases": bases,
           "spy_close": C.get("SPY"), "spy_equity": r2(spy_eq), "universe_equity": r2(univ_eq),
           "variants": variants}
    return {"day": day, "trades": trades}


# ---------------------------------------------------------------- planning
def lens_counts(s):
    vals = [s.get(k, 0) or 0 for k in LENSES]
    return sum(v > 0 for v in vals), sum(v < 0 for v in vals), sum(vals)


def entry_ok(v, s):
    if s.get("conviction", 0) < v.get("min_conviction", 1):
        return False
    if v.get("skip_events", True) and s.get("event"):
        return False
    if v["signal"] == "consensus":
        pos, neg, _ = lens_counts(s)
        return pos >= 2
    if v["signal"] == "conviction":
        return True
    return (s.get(v["signal"]) or 0) >= 1


def exit_signal(v, s):
    if v["signal"] == "consensus":
        pos, neg, _ = lens_counts(s)
        return neg > pos
    if v["signal"] == "conviction":
        return (s.get("conviction") or 3) <= 2
    return (s.get(v["signal"]) or 0) <= -1


def strength(v, s):
    if v["signal"] == "consensus":
        return (lens_counts(s)[2], s.get("conviction", 0))
    if v["signal"] == "conviction":
        return (s.get("conviction", 0), composite(s))
    return (s.get(v["signal"]) or 0, s.get("conviction", 0))


def composite(s):
    return sum((s.get(k) or 0) for k in LENSES) + 0.5 * ((s.get("conviction") or 3) - 3)


def blocked(v, pf, cfg):
    lim = cfg.get("daily_loss_limit_pct")
    return bool(lim and pf["prev_equity"] and pf["equity"] < pf["prev_equity"] * (1 - lim / 100))


def auto_reason(v, s):
    parts = [f"{k} {s.get(k):+d}" for k in LENSES if s.get(k) is not None]
    note = s.get("note") or ""
    return f"Skóre {', '.join(parts)}, přesvědčení {s.get('conviction')}/5. {note}".strip()


def allowed(v, scores):
    t = v.get("tickers")
    return {k: s for k, s in scores.items() if not t or k in t}


def market_blocked(v, scores):
    if v.get("market_filter") == "spy_trend":
        return (scores.get("SPY", {}).get("trend") or 0) < 0
    return False


def plan_signal(v, pf, scores, cfg):
    orders, decisions = [], []
    held = {p["ticker"] for p in pf["positions"]}
    for p in pf["positions"]:
        s = scores.get(p["ticker"])
        if s and exit_signal(v, s):
            orders.append({"ticker": p["ticker"], "side": "SELL", "exit_reason": "Signál k prodeji", "reason": auto_reason(v, s)})
            decisions.append({"ticker": p["ticker"], "action": "SELL", "conviction": s.get("conviction"), "reason": auto_reason(v, s)})
        else:
            decisions.append({"ticker": p["ticker"], "action": "HOLD", "conviction": (s or {}).get("conviction"), "reason": auto_reason(v, s) if s else "Bez skóre, držet."})
    if blocked(v, pf, cfg):
        decisions.append({"ticker": "—", "action": "SKIP", "conviction": 0, "reason": "Denní limit ztráty překročen, žádné nové nákupy."})
        return orders, decisions
    if market_blocked(v, scores):
        decisions.append({"ticker": "SPY", "action": "SKIP", "conviction": 0, "reason": "Tržní filtr: trend SPY je záporný, žádné nové nákupy."})
        return orders, decisions
    slots = v["max_positions"] - (len(held) - sum(o["side"] == "SELL" for o in orders))
    cands = sorted([t for t, s in allowed(v, scores).items() if t not in held and entry_ok(v, s)],
                   key=lambda t: strength(v, scores[t]), reverse=True)
    for t in cands[:max(slots, 0)]:
        s = scores[t]
        o = {"ticker": t, "side": "BUY", "weight_pct": v["weight_pct"], "stop_pct": v.get("stop_pct"),
             "conviction": s.get("conviction"), "reason": auto_reason(v, s)}
        orders.append(o)
        decisions.append({"ticker": t, "action": "BUY", "conviction": s.get("conviction"), "weight_pct": v["weight_pct"], "reason": o["reason"]})
    return orders, decisions


def plan_top_n(v, pf, scores, cfg):
    orders, decisions = [], []
    pool = allowed(v, scores)
    ranked = sorted(pool, key=lambda t: composite(pool[t]), reverse=not v.get("invert"))
    rank = {t: i for i, t in enumerate(ranked)}
    held = [p["ticker"] for p in pf["positions"]]
    keep = []
    for t in held:
        s = scores.get(t, {})
        if t in rank and rank[t] >= v.get("hysteresis_rank", 10):
            orders.append({"ticker": t, "side": "SELL", "exit_reason": "Vypadl z top výběru", "reason": f"Pořadí {rank[t] + 1}."})
            decisions.append({"ticker": t, "action": "SELL", "conviction": s.get("conviction"), "reason": f"Vypadl z top {v.get('hysteresis_rank', 10)}, pořadí {rank[t] + 1}."})
        else:
            keep.append(t)
            decisions.append({"ticker": t, "action": "HOLD", "conviction": s.get("conviction"), "reason": f"Pořadí {rank.get(t, -1) + 1} podle složeného skóre."})
    slots = v["max_positions"] - len(keep)
    for t in [t for t in ranked if t not in keep][:max(slots, 0)]:
        s = scores[t]
        orders.append({"ticker": t, "side": "BUY", "weight_pct": v["weight_pct"], "stop_pct": v.get("stop_pct"),
                       "conviction": s.get("conviction"), "reason": auto_reason(v, s)})
        decisions.append({"ticker": t, "action": "BUY", "conviction": s.get("conviction"), "weight_pct": v["weight_pct"], "reason": f"Pořadí {rank[t] + 1}. " + auto_reason(v, s)})
    return orders, decisions


def plan_free(v, pf, free, cfg):
    orders, decisions = [], []
    held = {p["ticker"]: p for p in pf["positions"]}
    buys = 0
    for d in (free or {}).get("decisions", []):
        a, t = d.get("action"), d.get("ticker")
        dd = {k: d.get(k) for k in ("ticker", "action", "conviction", "weight_pct", "reason", "invalidation") if d.get(k) is not None}
        if a == "HOLD" and t in held and d.get("new_stop"):
            if not held[t].get("stop") or d["new_stop"] > held[t]["stop"]:
                held[t]["stop"] = r4(d["new_stop"])
        if a == "SELL" and t in held:
            orders.append({"ticker": t, "side": "SELL", "exit_reason": "Rozhodnutí bota", "reason": d.get("reason", "")})
        if a == "BUY" and t not in held:
            if blocked(v, pf, cfg) or d.get("conviction", 0) < v.get("min_conviction", 3):
                dd["action"], dd["reason"] = "SKIP", "Zablokováno pravidly: " + d.get("reason", "")
            elif len(held) - sum(o["side"] == "SELL" for o in orders) + buys >= v["max_positions"]:
                dd["action"], dd["reason"] = "SKIP", "Plný počet pozic: " + d.get("reason", "")
            else:
                w = min(d.get("weight_pct") or v["weight_pct"], v["weight_pct"])
                orders.append({"ticker": t, "side": "BUY", "weight_pct": w, "stop_pct": v.get("stop_pct"),
                               "stop_price": d.get("stop_price"), "conviction": d.get("conviction"), "reason": d.get("reason", "")})
                dd["weight_pct"] = w
                buys += 1
        decisions.append(dd)
    return orders, decisions


def plan_random(ref_orders, pf, rcfg, universe, date):
    n = sum(o["side"] == "BUY" for o in ref_orders)
    held = {p["ticker"] for p in pf["positions"]}
    slots = rcfg["max_positions"] - len(held)
    eligible = sorted(t for t in universe if t not in held)
    rng = random.Random("nahoda-" + date)
    picks = rng.sample(eligible, max(0, min(n, slots, len(eligible))))
    orders = [{"ticker": t, "side": "BUY", "weight_pct": rcfg["weight_pct"], "stop_pct": rcfg.get("stop_pct"), "reason": "Náhodný výběr."} for t in picks]
    decisions = [{"ticker": t, "action": "BUY", "weight_pct": rcfg["weight_pct"], "reason": "Náhodný výběr."} for t in picks]
    return orders, decisions


def plan(settled, scores_doc, free, cfg):
    day, trades = settled["day"], settled["trades"]
    scores = scores_doc.get("scores", {})
    for v in cfg["variants"]:
        pf = day["variants"][v["id"]]
        if v["mode"] == "free":
            o, d = plan_free(v, pf, free, cfg)
        elif v["mode"] == "top_n":
            o, d = plan_top_n(v, pf, scores, cfg)
        else:
            o, d = plan_signal(v, pf, scores, cfg)
        pf["orders"], pf["decisions"], pf["plan_equity"] = o, d, pf["equity"]
    rcfg = cfg.get("random_baseline", {"max_positions": 5, "weight_pct": 20, "stop_pct": 7, "max_hold_days": 10, "mirror": "zaklad"})
    rpf = day["variants"]["nahoda"]
    ref = day["variants"].get(rcfg.get("mirror", "zaklad"), {}).get("orders", [])
    rpf["orders"], rpf["decisions"] = plan_random(ref, rpf, rcfg, cfg["universe"], day["date"])
    rpf["plan_equity"] = rpf["equity"]
    return {"day": day, "trades": trades}


def check(cfg, hist, today):
    errors, warnings = [], []
    uni = cfg["universe"]
    C, O, H, L = (today.get(k, {}) for k in ("close", "open", "high", "low"))
    missing = [t for t in uni if C.get(t) is None]
    if len(missing) > 5:
        errors.append(f"Chybí ceny pro {len(missing)} titulů: {', '.join(missing)}")
    elif missing:
        warnings.append(f"Chybí ceny: {', '.join(missing)}")
    if C.get("SPY") is None:
        errors.append("Chybí cena SPY")
    stale = 0
    for t in uni:
        c, o, h, l = C.get(t), O.get(t), H.get(t), L.get(t)
        if c is None:
            continue
        if None in (o, h, l):
            errors.append(f"{t}: neúplné OHLC")
            continue
        if not (l <= min(o, c) + 1e-6 and h >= max(o, c) - 1e-6 and l > 0):
            errors.append(f"{t}: nekonzistentní OHLC (o {o}, h {h}, l {l}, c {c})")
        rows = (hist or {}).get(t, [])
        prev = [r for r in rows if r[0] < today["date"]]
        if prev:
            pc = prev[-1][4]
            ch = (c / pc - 1) * 100
            if abs(ch) > 25:
                errors.append(f"{t}: změna {ch:+.1f} % proti předchozímu zavření, pravděpodobně chybná data nebo split")
            elif abs(ch) > 12:
                warnings.append(f"{t}: velký pohyb {ch:+.1f} %, ověř")
            if abs(c - pc) < 1e-9 and o == prev[-1][1]:
                stale += 1
    if stale > 3:
        errors.append(f"{stale} titulů má stejné ceny jako předchozí den, data jsou nejspíš stará")
    return {"errors": errors, "warnings": warnings, "tickers_ok": len(uni) - len(missing), "tickers_missing": missing}


def main(argv):
    cmd = argv[1]
    if cmd == "history-append":
        dump(history_append(load(argv[2]), load(argv[3])), argv[4])
    elif cmd == "features":
        f = features(load(argv[2]))
        dump(f, argv[3])
        print_features(f)
    elif cmd == "settle":
        dump(settle(load(argv[2]), load(argv[3]), load(argv[4])), argv[5])
    elif cmd == "check":
        res = check(load(argv[2]), load(argv[3]), load(argv[4]))
        dump(res, argv[5])
        print(json.dumps(res, ensure_ascii=False, indent=1))
    elif cmd == "plan":
        dump(plan(load(argv[3]), load(argv[4]), load(argv[5]), load(argv[2])), argv[6])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
