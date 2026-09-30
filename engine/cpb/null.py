"""Per-variant null distribution through the real accounting engine (skill or luck?).

Each random path replays the stored point-in-time days with the variant's own rules and costs, but picks RANDOM
coins: on every decision day it holds as many coins as the variant targeted, with the same weight profile
(shuffled over the random coins), keeps as many of its previous coins as the variant kept, and trades through
ledger.rebalance at the same snapshot bid/ask with the same band, minimum trade, fees and slippage. Days on which
the variant held (LLM failure, weekly variant) are held too; max holding period and stops apply to the random
positions. The variant's actual return is placed as a percentile of these paths.

Approximation (documented in README): stops of random positions are checked on daily candles (5-minute bars are
only stored for coins the real `stop` variant held), placed at the end of their day.
"""
import os
import random

from . import canon, ledger as L
from .pipeline import SCENARIOS  # noqa: F401  (documents that only "base" is simulated here)

DAY_MS = 86_400_000


def pctl(x, xs):
    """Mid-rank percentile (ties count half), so a variant that never trades sits at 50, not 0."""
    return (sum(r < x - 1e-12 for r in xs) + 0.5 * sum(abs(r - x) <= 1e-12 for r in xs)) / len(xs) * 100


def load_days(repo, runs):
    """runs = [(record, dir)] chronological, good ones only -> list of day dicts."""
    out = []
    for rec, d in runs:
        inp = canon.read_json(os.path.join(d, "inputs.json"))
        day = {"date": inp["date"], "asof": inp["asof"], "mode": inp["mode"], "status": inp["status"],
               "cfg": canon.read_json(os.path.join(d, "config.json"))}
        led = canon.read_json(os.path.join(d, "ledger.json"))
        day["close"] = {pid: m["base"]["equity"] for pid, m in led["close"].items()}
        if inp["mode"] == "decision":
            dec = canon.read_json(os.path.join(d, "decisions.json"))
            feats = canon.read_json(os.path.join(d, "features.json"))
            snap = inp["snapshot"]
            day.update(targets={pid: x["targets"] for pid, x in dec["portfolios"].items()},
                       pool=sorted(c for c in inp["universe"]["coins"] if feats.get(c) and inp["status"].get(c) == "TRADING"
                                   and c in snap["prices"]),
                       prices=snap["prices"], quotes=snap.get("quotes"), ts=snap["fetched_ms"])
        out.append(day)
    return out


class Hist:
    """Indexes built once per null run: rows by date, and daily candles as end-of-day bars for the stop check."""

    def __init__(self, hist):
        self.rows = hist
        self.by_date = {c: {r[0]: r for r in rows} for c, rows in hist.items()}
        self.bars = {c: [[canon.date_ms(r[0]) + DAY_MS - 300_000, r[1], r[2], r[3], r[4]] for r in rows] for c, rows in hist.items()}

    def bars_until(self, coins, asof):
        end = canon.date_ms(asof) + DAY_MS
        return {c: [b for b in self.bars.get(c, []) if b[0] < end] for c in coins}


def simulate_path(vdef, days, H, rng, capital, replicate=False):
    """One random path. replicate=True uses the variant's ACTUAL targets instead of random coins (self-check:
    it must reproduce the variant's own equity, up to the daily-candle stop approximation)."""
    vid = vdef["id"]
    stops = vdef.get("stops")
    led, prev_actual, last_close = None, None, {}
    hist, by_date = H.rows, H.by_date
    for day in days:
        cfg, D, asof = day["cfg"], day["date"], day["asof"]
        if led is None:
            if day["mode"] != "decision" or vid not in day["targets"]:
                continue
            led = L.new_ledger(capital)
        closes = {c: by_date.get(c, {}).get(asof, [None] * 5)[4] for c in led["positions"]}
        vol = {c: by_date.get(c, {}).get(asof, [None] * 7)[6] for c in set(led["positions"]) | set(day.get("pool", []))}
        led["blocked"] = {c: d for c, d in led["blocked"].items() if d >= D}
        if stops and led["positions"]:
            L.apply_stops(led, vid, "base", cfg, stops, H.bars_until(list(led["positions"]), asof), vol, D, D, None, "null")
        for c in sorted(led["positions"]):
            if day["status"].get(c, "TRADING") != "TRADING" or closes.get(c) is None:
                rows = [r for r in hist.get(c, []) if r[0] <= asof]
                L.force_sell(led, vid, "base", cfg, c, rows[-1][4], rows[-1][6], D, canon.date_ms(rows[-1][0]) + DAY_MS,
                             "delisting", cfg["costs"]["delist_extra_bps"], None, "null")
        closes = {c: by_date[c][asof][4] for c in led["positions"]}
        last_close[D] = led["cash"] + sum(p["qty"] * closes[c] for c, p in led["positions"].items())
        if day["mode"] != "decision":
            continue
        exits = {}
        if vdef.get("max_hold_days"):
            for c, p in sorted(led["positions"].items()):
                if canon.days_between(p["entry_date"], D) >= vdef["max_hold_days"]:
                    exits[c] = "max_hold"
        actual = day["targets"].get(vid)
        px = day["prices"]
        if any(c not in px for c in led["positions"]):
            actual, exits = None, {}             # cannot price a random holding at the snapshot: hold (rare)
        targets = None
        if actual is not None and replicate:
            targets = {c: w for c, w in actual.items() if c not in led["blocked"] and c not in exits}
        elif actual is not None:
            blocked = set(led["blocked"]) | set(exits)
            pool = [c for c in day["pool"] if c not in blocked]
            n = min(len(actual), len(pool))
            keep_n = len(set(actual) & set(prev_actual or {}))
            held = [c for c in sorted(led["positions"]) if c in pool]
            rng.shuffle(held)
            kept = held[:min(keep_n, n)]
            rest = [c for c in pool if c not in kept]
            chosen = kept + rng.sample(rest, n - len(kept))
            profile = sorted(actual.values(), reverse=True)[:n]
            rng.shuffle(profile)
            targets = dict(zip(chosen, profile))
            prev_actual = actual
        L.rebalance(led, vid, "base", cfg, targets, exits, px, vol, D, day["ts"], "null", None, day.get("quotes"))
        for c in exits:
            led["blocked"][c] = D
    return last_close


def null_percentiles(days, hist, variant_defs, n_paths=1000, seed=11):
    if sum(1 for d in days if d["mode"] == "decision") < 5:
        return {}
    last = days[-1]
    H = Hist(hist)
    out = {}
    for v in variant_defs:
        vid = v["id"]
        first = next((d for d in days if d["mode"] == "decision" and vid in d.get("targets", {})), None)
        if not first or vid not in last["close"]:
            continue
        capital = first["cfg"]["start_capital"]
        actual = last["close"][vid] / capital - 1
        rng = random.Random(f"{seed}|{vid}")
        res = []
        for _ in range(n_paths):
            eq = simulate_path(v, days, H, rng, capital)
            res.append(eq[last["date"]] / capital - 1)
        res.sort()
        out[vid] = {"actual": actual, "p05": res[int(0.05 * n_paths)], "p50": res[n_paths // 2],
                    "p95": res[int(0.95 * n_paths) - 1], "percentile": pctl(actual, res),
                    "paths": n_paths, "as_of": last["date"]}
    return out


# ============================================================ hourly variants

def hourly_null_percentiles(repo, hgood, cfg, n_paths=200, seed=13):
    """Null of the hourly variants: at every successful hourly run a random path holds as many coins as the variant held
    after that run, with the same weight profile, keeping as many coins as the variant kept, traded through
    ledger.rebalance at that run's bid/ask; trailing stops (h_breakout) on hourly candles. Measured at the last run's close."""
    import bisect
    from .hourly import HOUR_MS, load_hh
    hh = load_hh(repo, days=100000)
    closes = {c: {r[0]: r[4] for r in rows} for c, rows in hh.items()}
    opens = {c: [r[0] for r in rows] for c, rows in hh.items()}
    steps = []
    for r in hgood:
        inp = canon.read_json(os.path.join(r["_dir"], "inputs.json"))
        led = canon.read_json(os.path.join(r["_dir"], "ledger.json"))
        snap = inp["snapshot"]
        actual = {vid: {c: p["value"] / m["base"]["equity"] for c, p in m["base"]["positions"].items()} for vid, m in led["after"].items()}
        steps.append({"t_end": inp["t_end"], "label": inp["label"], "date": inp["date"], "status": inp["status"],
                      "cfg": canon.read_json(os.path.join(r["_dir"], "config.json")),
                      "pool": sorted(c for c in inp["universe"] if inp["status"].get(c) == "TRADING" and c in snap["prices"]
                                     and closes.get(c, {}).get(inp["t_end"] - HOUR_MS)),
                      "prices": snap["prices"], "quotes": snap.get("quotes"), "ts": snap["fetched_ms"], "actual": actual,
                      "close_eq": {vid: m["base"]["equity"] for vid, m in led["close"].items()}})

    def recent(c, t_end):
        rows = hh.get(c, [])
        i = bisect.bisect_left(opens.get(c, []), t_end)
        return rows[max(0, i - 48):i]

    def bars(c, t_end):
        return [r[:5] for r in recent(c, t_end)]

    out = {}
    for v in cfg["hourly"]["variants"]:
        vid = v["id"]
        if vid not in steps[-1]["close_eq"]:
            continue
        capital = steps[0]["cfg"]["start_capital"]
        stop = {"trail_pct": v["trail_pct"]} if v.get("trail_pct") else None
        rng = random.Random(f"{seed}|{vid}")
        res = []
        for _ in range(n_paths):
            led, prev = L.new_ledger(capital), set()
            for s in steps:
                cfg_s, t_end = s["cfg"], s["t_end"]
                vol = {c: sum(r[5] for r in recent(c, t_end)[-24:]) for c in set(led["positions"]) | set(s["pool"])}
                if stop and led["positions"]:
                    L.apply_stops(led, vid, "base", cfg_s, stop, {c: bars(c, t_end) for c in led["positions"]}, vol, s["date"], s["label"], None, "null")
                for c in sorted(led["positions"]):
                    if s["status"].get(c, "TRADING") != "TRADING" or c not in s["prices"]:
                        last = bars(c, t_end)[-1]
                        L.force_sell(led, vid, "base", cfg_s, c, last[4], vol.get(c), s["date"], last[0] + HOUR_MS, "delisting",
                                     cfg_s["costs"]["delist_extra_bps"], None, "null")
                act = s["actual"].get(vid, {})
                n = min(len(act), len(s["pool"]))
                keep_n = len(set(act) & prev)
                held_now = [c for c in sorted(led["positions"]) if c in s["pool"]]
                rng.shuffle(held_now)
                kept = held_now[:min(keep_n, n)]
                rest = [c for c in s["pool"] if c not in kept]
                chosen = kept + rng.sample(rest, n - len(kept))
                prof = sorted(act.values(), reverse=True)[:n]
                rng.shuffle(prof)
                L.rebalance(led, vid, "base", cfg_s, dict(zip(chosen, prof)), {}, s["prices"], vol, s["date"], s["ts"], "null", None, s["quotes"])
                prev = set(act)
            last = steps[-1]
            eq = led["cash"] + sum(p["qty"] * closes[c].get(last["t_end"] - HOUR_MS, s["prices"].get(c, 0)) for c, p in led["positions"].items())
            res.append(eq / capital - 1)
        res.sort()
        actual_ret = steps[-1]["close_eq"][vid] / capital - 1
        out[vid] = {"actual": actual_ret, "p05": res[int(0.05 * n_paths)], "p50": res[n_paths // 2], "p95": res[int(0.95 * n_paths) - 1],
                    "percentile": pctl(actual_ret, res), "paths": n_paths, "as_of": steps[-1]["label"]}
    return out
