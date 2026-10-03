#!/usr/bin/env python3
"""Pre-registered evaluation of the slow trend-filter forward test (docs/PREREGISTRATION_TREND.md §4-5).

Written and tested on synthetic data BEFORE the test window starts; its sha256 goes into the
"Změny" section of the pre-registration before the window opens. Do not change it afterwards: any change
is a new, clearly labelled post-hoc analysis.

  python tools/prereg_trend_eval.py [--repo DIR] [--json OUT]

Primary: trend_btc200 vs b_btc. Secondary (descriptive): trend_5050_200 vs b_btc_eth.
Criteria (all): max DD <= 0.6 x BTC HOLD; Sharpe >= BTC HOLD; both also under stress costs; every signal
change executed in the next decision run. Valid only if BTC HOLD drew down >= 25 % in the window, otherwise
the window extends once by 12 months.
"""
import argparse
import datetime
import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))
from cpb import canon  # noqa: E402

PRIMARY, SECONDARY = "trend_btc200", "trend_5050_200"
BENCH, BENCH2 = "b_btc", "b_btc_eth"
START_MIN, START_MAX = "2026-12-01", "2026-12-31"
END, EXT_END = "2028-11-30", "2029-11-30"
VALID_DD = 0.25
DD_RATIO = 0.6
SMA_N = 200
SCENARIOS = ("base", "stress")


# ---------------------------------------------------------------- data

def load_runs(repo):
    """Successful daily runs (ok / warning / catchup), chronological: [{date, mode, asof, ledger}]."""
    out = []
    for p in sorted(glob.glob(os.path.join(repo, "runs", "*", "run.json"))):
        rec = canon.read_json(p)
        if rec.get("status") not in ("ok", "warning", "catchup"):
            continue
        d = os.path.dirname(p)
        inp = canon.read_json(os.path.join(d, "inputs.json")) or {}
        led = canon.read_json(os.path.join(d, "ledger.json"))
        if not led:
            continue
        out.append({"date": rec["date"], "mode": inp.get("mode"), "asof": inp.get("asof"), "ledger": led})
    return out


def btc_closes(repo):
    rows = canon.read_json(os.path.join(repo, "data", "history", "BTC.json")) or []
    return {r[0]: r[4] for r in rows}


def sma_signal(closes, asof, n=SMA_N):
    """True = hold BTC: close(asof) > mean of the n closes ending at asof, all n present on consecutive days.
    Missing history -> False (cash), never filled in (prereg §3)."""
    vals = []
    for k in range(n):
        v = closes.get(canon.add_days(asof, -k))
        if v is None:
            return False
        vals.append(v)
    return vals[0] > sum(vals) / n


# ---------------------------------------------------------------- metrics

def series(runs, pid, scn):
    return [(r["date"], r["ledger"]["close"][pid][scn]["equity"]) for r in runs
            if pid in r["ledger"].get("close", {}) and scn in r["ledger"]["close"][pid]]


def returns(eq):
    return [b / a - 1 for (_, a), (_, b) in zip(eq, eq[1:]) if a > 0]


def max_dd(eq):
    peak, mdd = None, 0.0
    for _, x in eq:
        peak = x if peak is None else max(peak, x)
        mdd = max(mdd, 1 - x / peak)
    return mdd


def sharpe(r):
    if len(r) < 2:
        return None
    m = sum(r) / len(r)
    sd = math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1))
    return m / sd * math.sqrt(365) if sd > 0 else None


def cagr(eq):
    if len(eq) < 2 or eq[0][1] <= 0:
        return None
    days = canon.days_between(eq[0][0], eq[-1][0])
    return (eq[-1][1] / eq[0][1]) ** (365 / days) - 1 if days > 0 else None


def recovery_days(eq):
    """Longest stretch (days) below a previous peak."""
    peak, since, longest = None, None, 0
    for d, x in eq:
        if peak is None or x >= peak:
            if since is not None:
                longest = max(longest, canon.days_between(since, d))
            peak, since = x, None
        elif since is None:
            since = d
    if since is not None:
        longest = max(longest, canon.days_between(since, eq[-1][0]))
    return longest


def worst_month(eq):
    by = {}
    for d, x in eq:
        by.setdefault(d[:7], []).append(x)
    months = sorted(by)
    rets = [(m, by[m][-1] / by[p][-1] - 1) for p, m in zip(months, months[1:])]
    return min(rets, key=lambda t: t[1]) if rets else None


def describe(runs, pid):
    eq = series(runs, pid, "base")
    exp = [r["ledger"]["close"][pid]["base"].get("exposure", 0) for r in runs if pid in r["ledger"].get("close", {})]
    last = runs[-1]["ledger"]["close"].get(pid, {}).get("base", {}) if runs else {}
    c, dd = cagr(eq), max_dd(eq)
    return {"cagr": c, "max_dd": dd, "calmar": (c / dd) if c is not None and dd > 0 else None,
            "sharpe": sharpe(returns(eq)), "days": len(eq), "time_in_cash": (sum(1 for e in exp if e < 1e-9) / len(exp)) if exp else None,
            "costs_usd": (last.get("fees", 0) + last.get("slippage", 0)) if last else None,
            "worst_month": worst_month(eq), "longest_underwater_days": recovery_days(eq) if eq else None}


# ---------------------------------------------------------------- evaluation

def execution_check(runs, closes):
    """Criterion 4: after every decision run the primary holds BTC iff the SMA200 signal (as of that run's
    asof) says so. Catch-up runs make no decisions and are skipped."""
    misses, changes, prev = [], 0, None
    for r in runs:
        if r["mode"] != "decision" or not r["asof"]:
            continue
        after = (r["ledger"].get("after") or {}).get(PRIMARY, {}).get("base")
        if after is None:
            misses.append((r["date"], "chybí stav po plnění"))
            continue
        sig = sma_signal(closes, r["asof"])
        holds = after.get("positions", {}).get("BTC", {}).get("qty", 0) > 0
        if holds != sig:
            misses.append((r["date"], f"signál {'BTC' if sig else 'hotovost'}, drží {'BTC' if holds else 'hotovost'}"))
        if prev is not None and sig != prev:
            changes += 1
        prev = sig
    return misses, changes


def evaluate(repo, today=None):
    today = today or datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    all_runs = load_runs(repo)
    with_primary = [r for r in all_runs if PRIMARY in r["ledger"].get("close", {})]
    res = {"today": today, "notes": []}
    if not with_primary:
        res.update(status="nezačalo", valid=False)
        res["notes"].append(f"žádný běh s variantou {PRIMARY}")
        return res
    start = with_primary[0]["date"]
    res["start"] = start
    if not (START_MIN <= start <= START_MAX):
        res.update(status="neplatné", valid=False)
        res["notes"].append(f"start {start} mimo {START_MIN}..{START_MAX}: podle §4 je nutná nová verze pre-registrace")
        return res

    def window(end):
        return [r for r in all_runs if start <= r["date"] <= end]

    end = END
    runs = window(end)
    bench_dd = max_dd(series(runs, BENCH, "base"))
    extended = False
    if bench_dd < VALID_DD and today > END:              # decided only once the original window is over
        extended = True
        end = EXT_END
        runs = window(end)
        bench_dd = max_dd(series(runs, BENCH, "base"))
        res["notes"].append(f"propad BTC HOLD do {END} pod {VALID_DD:.0%}: okno prodlouženo do {EXT_END} (§4)")
    res.update(end=end, extended=extended, bench_max_dd=bench_dd,
               bear_market=bench_dd >= VALID_DD, complete=today > end)
    if today <= end:
        res["notes"].append(f"dnes {today}: okno končí {end}, výsledek je jen průběžný a nic se podle něj nerozhoduje")

    crit = {}
    for scn in SCENARIOS:
        p_eq, b_eq = series(runs, PRIMARY, scn), series(runs, BENCH, scn)
        p_dd, b_dd = max_dd(p_eq), max_dd(b_eq)
        p_sr, b_sr = sharpe(returns(p_eq)), sharpe(returns(b_eq))
        res[scn] = {"primary_max_dd": p_dd, "bench_max_dd": b_dd, "primary_sharpe": p_sr, "bench_sharpe": b_sr,
                    "dd_ratio": (p_dd / b_dd) if b_dd > 0 else None}
        crit[f"dd_{scn}"] = b_dd > 0 and p_dd <= DD_RATIO * b_dd
        crit[f"sharpe_{scn}"] = p_sr is not None and b_sr is not None and p_sr >= b_sr
    misses, changes = execution_check(runs, btc_closes(repo))
    crit["execution"] = not misses
    res["execution_misses"], res["signal_changes"] = misses, changes
    res["criteria"] = {"1_dd_base": crit["dd_base"], "2_sharpe_base": crit["sharpe_base"],
                       "3_stress": crit["dd_stress"] and crit["sharpe_stress"], "4_execution": crit["execution"]}
    res["pass"] = all(res["criteria"].values())
    res["valid"] = res["complete"]
    res["status"] = ("splněno" if res["pass"] else "zamítnuto") if res["complete"] else "průběžně"
    if res["complete"] and not res["bear_market"]:
        res["notes"].append("ani po prodloužení BTC HOLD nespadl o 25 %: výsledek se označí „bez medvědího trhu“ (§6)")
    res["describe"] = {pid: describe(runs, pid) for pid in (PRIMARY, BENCH, SECONDARY, BENCH2)}
    return res


def pct(x):
    return "   –  " if x is None else f"{x * 100:6.1f} %"


def num(x):
    return "  –  " if x is None else f"{x:5.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=ROOT)
    ap.add_argument("--json")
    ap.add_argument("--today", help="jen pro testy")
    a = ap.parse_args()
    res = evaluate(a.repo, a.today)
    print("=" * 90)
    print(f"FORWARD TEST TRENDOVÉHO FILTRU  stav: {res['status'].upper()}")
    print("=" * 90)
    for n in res["notes"]:
        print(f"  - {n}")
    if "criteria" in res:
        print(f"okno {res['start']} … {res['end']}{' (prodlouženo)' if res['extended'] else ''}, "
              f"propad BTC HOLD {pct(res['bench_max_dd'])}, změn signálu {res['signal_changes']}")
        for scn in SCENARIOS:
            s = res[scn]
            print(f"  {scn:<7} propad {pct(s['primary_max_dd'])} vs {pct(s['bench_max_dd'])} (poměr {num(s['dd_ratio'])}, limit {DD_RATIO})"
                  f"   Sharpe {num(s['primary_sharpe'])} vs {num(s['bench_sharpe'])}")
        for k, v in res["criteria"].items():
            print(f"  {k:<16} {'ano' if v else 'NE'}")
        for d, why in res["execution_misses"]:
            print(f"    nesplněné provedení {d}: {why}")
        print("\npopisně:")
        for pid, d in res["describe"].items():
            print(f"  {pid:<16} CAGR {pct(d['cagr'])}  propad {pct(d['max_dd'])}  Sharpe {num(d['sharpe'])}  Calmar {num(d['calmar'])}  "
                  f"v hotovosti {pct(d['time_in_cash'])}  nejdéle pod vrcholem {d['longest_underwater_days']} d")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)
        print(f"uloženo: {a.json}")


if __name__ == "__main__":
    main()
