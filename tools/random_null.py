#!/usr/bin/env python3
"""Skill or luck? Null distribution of each variant against random stock picking.

A single random portfolio (the "nahoda" baseline) is one draw with huge variance. Here, for each
variant, the real engine replays the SAME trading activity (same number of buys and sells on the
same days) with RANDOM tickers, under the variant's own stop / max-hold / weight / position limit
and the same costs. Repeated N times this gives the return distribution you would get by luck.

  python tools/random_null.py export.json         # {"days": [day docs], "ohlc": [today.json docs]}
  python tools/random_null.py --demo              # synthetic market (scores there are noise for mr)

The variant's actual return is placed as a percentile in its own null distribution.
> 95 = unlikely to be luck (still 19 variants tested: expect ~1 by chance, see README).
"""
import json, pathlib, random, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import simulate  # noqa: E402

E = simulate.load_engine()


def null_returns(cfg, days, ohlc, vid, n=300, seed=1):
    vcfg = {v["id"]: v for v in cfg["variants"]}[vid]
    start = cfg["start_capital"]
    costs = E.Costs(cfg)
    U = cfg["universe"]
    acts = [(sum(o["side"] == "BUY" for o in d["variants"][vid].get("orders", [])),
             sum(o["side"] == "SELL" for o in d["variants"][vid].get("orders", []))) for d in days]
    rets = []
    for k in range(n):
        rng = random.Random(f"{vid}-{seed}-{k}")
        state, trades = None, []
        for i, today in enumerate(ohlc):
            state = E.settle_variant(vid, vcfg, state, today, costs, start, trades)
            nb, ns = acts[i]
            held = [p["ticker"] for p in state["positions"]]
            sells = rng.sample(held, min(ns, len(held)))
            free = sorted(t for t in U if t not in held)
            buys = rng.sample(free, min(nb, len(free)))
            state["orders"] = [{"ticker": t, "side": "SELL", "reason": "náhoda"} for t in sells] + \
                              [{"ticker": t, "side": "BUY", "weight_pct": vcfg["weight_pct"], "stop_pct": vcfg.get("stop_pct"),
                                "reason": "náhoda"} for t in buys]
            state["plan_equity"] = state["equity"]
        rets.append((state["equity"] / start - 1) * 100)
    return rets


def pct_of(x, xs):
    return sum(v < x for v in xs) / len(xs) * 100


def report(cfg, days, ohlc, n=300):
    start = cfg["start_capital"]
    print(f"{'varianta':<14}{'skutečně %':>11}{'náhoda medián':>15}{'náhoda 5–95 %':>20}{'percentil':>11}")
    for v in cfg["variants"]:
        vid = v["id"]
        act = (days[-1]["variants"][vid]["equity"] / start - 1) * 100
        nul = sorted(null_returns(cfg, days, ohlc, vid, n))
        print(f"{vid:<14}{act:>+11.2f}{nul[len(nul) // 2]:>+15.2f}{f'[{nul[int(.05 * n)]:+.1f}, {nul[int(.95 * n) - 1]:+.1f}]':>20}{pct_of(act, nul):>10.0f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cfg = json.load(open(simulate.ROOT / "config" / "config.json"))
    if sys.argv[1] == "--demo":
        sim = simulate.run(cfg, days=30, seed=5)
    else:
        sim = json.load(open(sys.argv[1]))
    report(cfg, sim["days"], sim["ohlc"], n=200)
