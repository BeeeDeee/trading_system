#!/usr/bin/env python3
"""Paper-trading engine for the evening bot (multi-variant).

Subcommands (all read/write JSON files, stdlib only):
  history-append HISTORY TODAY_PRICES OUT     merge today's OHLC into rolling history (80 rows)
  features HISTORY OUT                        indicators for every ticker (+ prints a table)
  settle CONFIG PREV_DAY|- TODAY_PRICES OUT   execute yesterday's orders at today's open, stops, max-hold, mark to market
  plan CONFIG SETTLED SCORES FREE|- OUT       orders for tomorrow's open for every variant + random baseline
  check CONFIG HISTORY TODAY_PRICES OUT       sanity-check today's prices before anything else ("errors" must be empty)
  validate CONFIG SCORES FREE|- OUT           report problems in scores/free decisions before `plan` ("errors" should be empty)
  apply-split HISTORY PREV|- TICKER RATIO OUTH OUTP
                                              re-base history + open positions after a verified split
                                              (RATIO = new shares per old share: 2 for 2:1, 0.1 for 1:10)

TODAY_PRICES = {"date": "YYYY-MM-DD", "open": {T: x}, "high": {...}, "low": {...}, "close": {...}}
SCORES       = {"date": ..., "scores": {T: {"trend": -2..2, "mr": -2..2, "news": -2..2,
                "conviction": 1..5, "event": bool, "note": str}}}
FREE         = {"decisions": [{"ticker", "action": BUY|SELL|HOLD|SKIP, "conviction", "weight_pct",
                "stop_price"?, "new_stop"?, "reason", "invalidation"}]}
OUT of plan  = {"day": <days/<date> document>, "trades": [<trades documents>]}
"""
import hashlib, json, math, os, random, sys

LENSES = ("trend", "mr", "news")
STALE_SCORE_DAYS = 2   # held ticker without a score for this many consecutive plans is sold
HIST_LEN = 80


def load(p):
    if p in (None, "-"):
        return None
    with open(p) as f:
        return json.load(f)


def dump(obj, p):
    tmp = p + ".tmp"          # atomic: a crash mid-write must not corrupt the input file
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def num(x):
    """Finite real number or None (bools and NaN/inf are not numbers here)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    return x


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
    if stop is not None and not 0 < stop < price:
        log.append(f"{ticker}: stop_price {stop} není pod nákupní cenou, použit standardní stop")
        stop = None
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
# ---------------------------------------------------------------- input validation
def _lens_int(x, lo, hi):
    x = num(x)
    return None if x is None else int(max(lo, min(hi, round(x))))


def clean_scores(scores, universe):
    """Coerce LLM-written scores to what the rules expect. Valid input passes through unchanged.
    Returns (clean, issues); missing tickers are NOT invented, they simply stay absent."""
    clean, issues = {}, []
    if not isinstance(scores, dict):
        return clean, ["scores není objekt"]
    for t, s in scores.items():
        if t not in universe:
            issues.append(f"{t}: mimo univerzum, ignorováno")
            continue
        if not isinstance(s, dict):
            issues.append(f"{t}: skóre není objekt, ignorováno")
            continue
        c = {}
        for k in LENSES:
            if s.get(k) is None:
                continue
            v = _lens_int(s[k], -2, 2)
            if v is None:
                issues.append(f"{t}: {k} není číslo, ignorováno")
            else:
                if v != s[k]:
                    issues.append(f"{t}: {k}={s[k]} upraveno na {v}")
                c[k] = v
        cv = _lens_int(s.get("conviction"), 1, 5)
        if cv is None:
            issues.append(f"{t}: chybí/neplatné přesvědčení, použito 3 (bez názoru)")
            cv = 3
        elif cv != s.get("conviction"):
            issues.append(f"{t}: conviction={s.get('conviction')} upraveno na {cv}")
        c["conviction"] = cv
        c["event"] = bool(s.get("event"))
        c["note"] = s.get("note") if isinstance(s.get("note"), str) else ""
        clean[t] = c
    return clean, issues


def clean_free(free, universe):
    """Same for the 'Claude volně' decisions: drop what cannot be executed safely."""
    out, issues = [], []
    decs = (free or {}).get("decisions", []) if isinstance(free, dict) or free is None else []
    if not isinstance(decs, list):
        return {"decisions": []}, ["decisions není seznam"]
    for d in decs:
        if not isinstance(d, dict) or d.get("action") not in ("BUY", "SELL", "HOLD", "SKIP") or d.get("ticker") not in universe:
            issues.append(f"rozhodnutí ignorováno: {str(d)[:80]}")
            continue
        c = dict(d)
        for k in ("conviction", "weight_pct", "stop_price", "new_stop"):
            if k in c:
                v = num(c[k])
                if v is None or (k != "conviction" and v <= 0):
                    issues.append(f"{d['ticker']}: {k} neplatné, ignorováno")
                    del c[k]
                else:
                    c[k] = v
        for k in ("reason", "invalidation"):
            if k in c and not isinstance(c[k], str):
                c[k] = str(c[k])
        out.append(c)
    return {"decisions": out}, issues


def validate(cfg, scores_doc, free):
    """Problems worth fixing BEFORE plan. plan() itself never crashes on them (it sanitizes)."""
    uni = cfg["universe"]
    raw = (scores_doc or {}).get("scores", {}) if isinstance(scores_doc, dict) else {}
    _, issues = clean_scores(raw, uni)
    errors = [i for i in issues if "mimo univerzum" not in i]
    missing = [t for t in uni if t not in raw]
    if missing:
        errors.append(f"Chybí skóre pro: {', '.join(missing)}")
    if "SPY" in raw and (raw["SPY"] or {}).get("trend") is None:
        errors.append("SPY nemá trend (používá ho tržní filtr)")
    _, fissues = clean_free(free, uni)
    warnings = list(fissues)
    return {"errors": errors, "warnings": warnings}


def tiebreak(date, t):
    """Deterministic, unbiased tie-break for equal scores: independent of JSON key order."""
    return hashlib.sha256(f"{date}|{t}".encode()).hexdigest()


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
        return pos >= 2 and neg <= v.get("max_neg_lenses", 0)
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


def plan_signal(v, pf, scores, cfg, date=""):
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
                   key=lambda t: (tuple(-x for x in strength(v, scores[t])), tiebreak(date, t)))
    for t in cands[:max(slots, 0)]:
        s = scores[t]
        o = {"ticker": t, "side": "BUY", "weight_pct": v["weight_pct"], "stop_pct": v.get("stop_pct"),
             "conviction": s.get("conviction"), "reason": auto_reason(v, s)}
        orders.append(o)
        decisions.append({"ticker": t, "action": "BUY", "conviction": s.get("conviction"), "weight_pct": v["weight_pct"], "reason": o["reason"]})
    return orders, decisions


def plan_top_n(v, pf, scores, cfg, date=""):
    orders, decisions = [], []
    pool = allowed(v, scores)
    sign = 1 if v.get("invert") else -1
    ranked = sorted(pool, key=lambda t: (sign * composite(pool[t]), tiebreak(date, t)))
    rank = {t: i for i, t in enumerate(ranked)}
    keep, sold = [], set()
    for p in pf["positions"]:
        t = p["ticker"]
        s = scores.get(t, {})
        p["no_score_days"] = 0 if t in scores else p.get("no_score_days", 0) + 1
        if p["no_score_days"] >= STALE_SCORE_DAYS:
            orders.append({"ticker": t, "side": "SELL", "exit_reason": "Bez skóre", "reason": f"Titul {p['no_score_days']} dny bez skóre."})
            decisions.append({"ticker": t, "action": "SELL", "conviction": None, "reason": f"Bez skóre {p['no_score_days']} dny, uzavřeno."})
            sold.add(t)
        elif t in rank and rank[t] >= v.get("hysteresis_rank", 10):
            orders.append({"ticker": t, "side": "SELL", "exit_reason": "Vypadl z top výběru", "reason": f"Pořadí {rank[t] + 1}."})
            decisions.append({"ticker": t, "action": "SELL", "conviction": s.get("conviction"), "reason": f"Vypadl z top {v.get('hysteresis_rank', 10)}, pořadí {rank[t] + 1}."})
            sold.add(t)
        else:
            keep.append(t)
            decisions.append({"ticker": t, "action": "HOLD", "conviction": s.get("conviction"), "reason": f"Pořadí {rank.get(t, -1) + 1} podle složeného skóre."})
    if blocked(v, pf, cfg):
        decisions.append({"ticker": "—", "action": "SKIP", "conviction": 0, "reason": "Denní limit ztráty překročen, žádné nové nákupy."})
        return orders, decisions
    slots = v["max_positions"] - len(keep)
    for t in [t for t in ranked if t not in keep and t not in sold][:max(slots, 0)]:
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
            if d["new_stop"] >= held[t]["last"]:
                dd["reason"] = (dd.get("reason") or "") + " (new_stop není pod cenou, ignorován)"
            elif not held[t].get("stop") or d["new_stop"] > held[t]["stop"]:
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
    scores, _ = clean_scores(scores_doc.get("scores", {}), cfg["universe"])
    free, _ = clean_free(free, cfg["universe"])
    for v in cfg["variants"]:
        pf = day["variants"][v["id"]]
        if v["mode"] == "free":
            o, d = plan_free(v, pf, free, cfg)
        elif v["mode"] == "top_n":
            o, d = plan_top_n(v, pf, scores, cfg, day["date"])
        else:
            o, d = plan_signal(v, pf, scores, cfg, day["date"])
        pf["orders"], pf["decisions"], pf["plan_equity"] = o, d, pf["equity"]
    rcfg = cfg.get("random_baseline", {"max_positions": 5, "weight_pct": 20, "stop_pct": 7, "max_hold_days": 10, "mirror": "zaklad"})
    rpf = day["variants"]["nahoda"]
    ref = day["variants"].get(rcfg.get("mirror", "zaklad"), {}).get("orders", [])
    if blocked(rcfg, rpf, cfg):
        rpf["orders"] = []
        rpf["decisions"] = [{"ticker": "—", "action": "SKIP", "conviction": 0, "reason": "Denní limit ztráty překročen, žádné nové nákupy."}]
    else:
        rpf["orders"], rpf["decisions"] = plan_random(ref, rpf, rcfg, cfg["universe"], day["date"])
    rpf["plan_equity"] = rpf["equity"]
    return {"day": day, "trades": trades}


SPLIT_RATIOS = (1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50)


def split_suspect(c, pc):
    """Ratio (new shares per old share) if the price jump looks like a clean split, else None."""
    r = pc / c
    for k in SPLIT_RATIOS:
        if abs(r / k - 1) < 0.04:
            return k
        if abs(r * k - 1) < 0.04:
            return 1 / k
    return None


def apply_split(hist, prev, ticker, ratio):
    """Re-base a ticker after a VERIFIED split: history prices, open positions, stops, baselines."""
    if not ratio or ratio <= 0:
        raise SystemExit("ratio must be > 0")
    hist = {t: [list(r) for r in rows] for t, rows in (hist or {}).items()}
    prev = json.loads(json.dumps(prev)) if prev else prev
    hist[ticker] = [[r[0]] + [None if x is None else round(x / ratio, 4) for x in r[1:]] for r in hist.get(ticker, [])]
    if prev:
        for vf in prev.get("variants", {}).values():
            for pos in vf.get("positions", []):
                if pos["ticker"] == ticker:
                    pos["shares"] = round(pos["shares"] * ratio, 4)
                    for k in ("entry_price", "last", "stop"):
                        if pos.get(k):
                            pos[k] = r4(pos[k] / ratio)
        b = prev.get("bases") or {}
        if ticker in b.get("univ0", {}) and b["univ0"][ticker]:
            b["univ0"][ticker] = b["univ0"][ticker] / ratio
        if ticker == "SPY" and b.get("spy0"):
            b["spy0"] = b["spy0"] / ratio
    return hist, prev


def check(cfg, hist, today):
    errors, warnings, suspects = [], [], {}
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
                sr = split_suspect(c, pc)
                if sr:
                    suspects[t] = sr
                errors.append(f"{t}: změna {ch:+.1f} % proti předchozímu zavření, pravděpodobně chybná data nebo split"
                              + (f" (poměr odpovídá splitu, RATIO {sr:g})" if sr else ""))
            elif abs(ch) > 12:
                warnings.append(f"{t}: velký pohyb {ch:+.1f} %, ověř")
            if abs(c - pc) < 1e-9 and o == prev[-1][1]:
                stale += 1
    if stale > 3:
        errors.append(f"{stale} titulů má stejné ceny jako předchozí den, data jsou nejspíš stará")
    return {"errors": errors, "warnings": warnings, "tickers_ok": len(uni) - len(missing), "tickers_missing": missing,
            "split_suspects": suspects}


ARITY = {"history-append": 5, "features": 4, "settle": 6, "check": 6, "plan": 7, "validate": 6, "apply-split": 8}


def main(argv):
    cmd = argv[1] if len(argv) > 1 else None
    if cmd not in ARITY or len(argv) != ARITY[cmd]:
        raise SystemExit(__doc__)
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
    elif cmd == "validate":
        res = validate(load(argv[2]), load(argv[3]), load(argv[4]))
        dump(res, argv[5])
        print(json.dumps(res, ensure_ascii=False, indent=1))
    elif cmd == "plan":
        dump(plan(load(argv[3]), load(argv[4]), load(argv[5]), load(argv[2])), argv[6])
    elif cmd == "apply-split":
        prev = load(argv[3])
        hist, prev = apply_split(load(argv[2]), prev, argv[4], float(argv[5]))
        dump(hist, argv[6])
        if prev is not None:
            dump(prev, argv[7])


if __name__ == "__main__":
    main(sys.argv)
