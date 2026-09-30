"""Target weights for every variant and benchmark (pure functions of point-in-time inputs).

Returns per portfolio {"targets": {coin: w} | None, "exits": {coin: reason}, "info": {...}}.
targets None = hold everything unchanged (LLM step failed, or not a rebalance day).
"""
import math
import random

from . import canon
from .features import aligned_returns, gate


def composite(s, rules):
    k = rules["composite"]
    return (k["trend"] * s["trend"] + k["news"] * s["news"] + k["mr"] * s["mr"]
            + k["conviction"] * (s["conviction"] - k["conviction_center"]))


def cap_weights(raw, cap, min_w):
    """Normalize to 1, cap each at `cap` redistributing excess to uncapped names (leftover = cash),
    then drop weights below min_w (their weight stays in cash)."""
    raw = {c: v for c, v in raw.items() if v > 0}
    if not raw:
        return {}
    tot = sum(raw.values())
    w = {c: v / tot for c, v in raw.items()}
    fixed = {}
    while True:
        over = [c for c in w if c not in fixed and w[c] > cap + 1e-12]
        if not over:
            break
        for c in over:
            fixed[c] = cap
        free = [c for c in w if c not in fixed]
        rest = 1 - sum(fixed.values())
        ftot = sum(raw[c] for c in free)
        for c in fixed:
            w[c] = cap
        for c in free:
            w[c] = rest * raw[c] / ftot if ftot > 0 else 0.0
        if not free:
            break
    return {c: v for c, v in sorted(w.items()) if v >= min_w - 1e-12}


def _cond(val, op, x):
    if val is None:
        return False
    return {">": val > x, ">=": val >= x, "<": val < x, "<=": val <= x}[op]


class Ctx:
    """Point-in-time inputs for one decision."""

    def __init__(self, cfg, date, universe_coins, feats, scores, hist, base_state, first_run, mcaps):
        self.cfg, self.date, self.U = cfg, date, universe_coins
        self.date_asof = canon.add_days(date, -1)
        self.feats, self.scores, self.hist = feats, scores, hist
        self.state = base_state            # {pid: ledger (base scenario)}
        self.first_run = first_run
        self.mcaps = mcaps                 # {coin: market cap} from the frozen universe
        self.rules = cfg["rules"]

    def row(self, coin):
        """All fields a rule may use: indicators + LLM scores + composite."""
        f = self.feats.get(coin) or {}
        r = dict(f)
        s = (self.scores or {}).get("coins", {}).get(coin)
        if s:
            r.update(s)
            r["composite"] = composite(s, self.rules)
        return r


def _rank(ctx, coins, field, desc):
    rows = [(c, ctx.row(c).get(field)) for c in coins]
    rows = [(c, v) for c, v in rows if v is not None]
    return [c for c, v in sorted(rows, key=lambda cv: ((-cv[1] if desc else cv[1]), canon.hash_rank(ctx.date, cv[0])))]


def _weights(ctx, scheme, coins):
    if scheme == "equal":
        return {c: 1.0 for c in coins}
    raw = {}
    for c in coins:
        r = ctx.row(c)
        atr = r.get("atr14_pct")
        if scheme == "composite":
            raw[c] = r["composite"]
        elif scheme == "shifted_composite":
            lo = min(ctx.row(x)["composite"] for x in coins)
            raw[c] = r["composite"] - lo + 1
        elif scheme == "inv_atr":
            raw[c] = 1 / atr if atr else 0
        elif scheme == "composite_inv_atr":
            raw[c] = r["composite"] / atr if atr else 0
        elif scheme == "p_minus_half":
            raw[c] = r["p_outperform_btc_7d"] - 0.5
        elif scheme == "field":
            raise ValueError
        else:
            raw[c] = r.get(scheme) or 0
    return raw


def portfolio_vol(ctx, w):
    rets = aligned_returns(ctx.hist, sorted(w), ctx.date_asof, 30)
    cs = [c for c in sorted(w) if c in rets]
    if not cs:
        return None
    n = len(rets[cs[0]])
    means = {c: sum(rets[c]) / n for c in cs}
    var = 0.0
    for a in cs:
        for b in cs:
            cv = sum((x - means[a]) * (y - means[b]) for x, y in zip(rets[a], rets[b])) / (n - 1)
            var += w[a] * w[b] * cv
    return math.sqrt(max(var, 0)) * math.sqrt(365)


def variant_plan(v, ctx):
    sel = v["select"]
    led = ctx.state.get(v["id"])
    info = {}
    exits = {}
    blocked = {c for c, d in (led or {}).get("blocked", {}).items() if d >= ctx.date}
    if v.get("max_hold_days") and led:
        for c, p in sorted(led["positions"].items()):
            if canon.days_between(p["entry_date"], ctx.date) >= v["max_hold_days"]:
                exits[c] = "max_hold"
                blocked.add(c)
    if v.get("uses_llm") and ctx.scores is None:
        return {"targets": None, "exits": exits, "info": {"hold": "krok LLM selhal – pozice beze změn"}}
    if v.get("rebalance") == "weekly" and not ctx.first_run and canon.weekday(ctx.date) != 0:
        return {"targets": None, "exits": exits, "info": {"hold": "rebalanc jen v pondělí"}}

    src = sel["source"]
    # only coins with today's data and a TRADING pair are ever candidates (feats is None otherwise)
    blocked |= {c for c in ctx.U if not ctx.feats.get(c)}
    scored = sorted(c for c in ctx.U if c in (ctx.scores or {}).get("coins", {}) and c not in blocked)
    if src == "free":
        fp = (ctx.scores or {}).get("claude_volne", {}).get("positions", [])
        t = {p["coin"]: p["weight_pct"] / 100 for p in fp if p["coin"] not in blocked}
        t = {c: w for c, w in t.items() if w >= ctx.rules["min_weight"]}
        return {"targets": dict(sorted(t.items())), "exits": exits, "info": {"selected": sorted(t)}}
    if src == "random":
        pool = sorted(c for c in ctx.U if ctx.feats.get(c) and c not in blocked)
        rng = random.Random(int(canon.hash_rank("nahodny", ctx.date), 16))
        picks = sorted(rng.sample(pool, min(sel["n"], len(pool))))
        return {"targets": cap_weights({c: 1.0 for c in picks}, v.get("max_weight", ctx.rules["max_weight"]), ctx.rules["min_weight"]),
                "exits": exits, "info": {"selected": picks}}
    top = [t["coin"] for t in ctx.scores.get("top10", [])] if ctx.scores else []
    if src == "llm_top10":
        cands = [c for c in top if c in scored][:10]
    elif src == "llm_top3":
        cands = [c for c in top if c in scored][:3]
    elif src == "universe_scored":
        cands = scored
    elif src == "universe":
        cands = sorted(c for c in ctx.U if ctx.feats.get(c))
    else:
        raise ValueError(f"unknown source {src}")
    cands = [c for c in cands if c not in blocked]
    for field, op, x in sel.get("conditions", []):
        cands = [c for c in cands if _cond(ctx.row(c).get(field), op, x)]
    if sel.get("gate"):
        cands = [c for c in cands if gate(ctx.feats.get(c), ctx.rules) and not ctx.row(c).get("risk_flag")]
    if sel.get("rank_by"):
        cands = _rank(ctx, cands, sel["rank_by"], sel.get("desc", True))[: sel.get("n", 10)]
    raw = _weights(ctx, v["weight"], cands)
    t = cap_weights(raw, v.get("max_weight", ctx.rules["max_weight"]), ctx.rules["min_weight"])
    info["selected"] = cands
    ov = v.get("overlay") or {}
    if ov.get("btc_below_sma"):
        b = ctx.feats.get("BTC") or {}
        s = b.get(f"sma{ov['btc_below_sma']}")
        if s is None or b["close"] < s:
            info["overlay"] = f"BTC pod SMA{ov['btc_below_sma']} → hotovost"
            t = {}
    if ov.get("vol_target") and t:
        pv = portfolio_vol(ctx, t)
        k = min(1.0, ov["vol_target"] / pv) if pv else 1.0
        info["overlay"] = f"vol portfolia {pv * 100:.0f} % → expozice ×{k:.2f}" if pv else "vol nelze spočítat"
        t = {c: w * k for c, w in t.items() if w * k >= ctx.rules["min_weight"]}
    return {"targets": t, "exits": exits, "info": info}


def benchmark_plan(b, ctx):
    led = ctx.state.get(b["id"])
    last = (led or {}).get("last_rebalance")
    kind = b["kind"]
    if kind == "cash":
        return {"targets": None, "exits": {}, "info": {}}
    if kind == "hodl":
        if last:
            return {"targets": None, "exits": {}, "info": {}}
        return {"targets": {b["coin"]: 1.0}, "exits": {}, "info": {"rebalance": True}}
    due = last is None
    if b["rebalance"] == "monthly":
        due = due or last[:7] != ctx.date[:7]
    elif b["rebalance"] == "weekly":
        due = due or canon.weekday(ctx.date) == 0
    if not due:
        return {"targets": None, "exits": {}, "info": {}}
    if kind == "fixed":
        t = dict(b["weights"])
    elif kind == "mcap_top":
        top = sorted((c for c in ctx.U if ctx.feats.get(c) and ctx.mcaps.get(c)), key=lambda c: (-ctx.mcaps[c], c))[: b["n"]]
        tot = sum(ctx.mcaps[c] for c in top)
        t = {c: ctx.mcaps[c] / tot for c in top}
    elif kind == "equal_universe":
        cs = sorted(c for c in ctx.U if ctx.feats.get(c))
        t = {c: 1 / len(cs) for c in cs}
    else:
        raise ValueError(kind)
    return {"targets": dict(sorted(t.items())), "exits": {}, "info": {"rebalance": True}}
